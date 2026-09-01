import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.io import decode_image

from misc.face_alignment import (
    VideoFaceExtractor,
    align_faces,
    center_crop_and_resize,
    restore_faces_to_original,
)
from misc.models import ImageInputRange
from misc.models.face_mask import FaceMasker
from misc.models.id_encoder import IDEncoder, IDEncoderProvider, get_alignment_template
from misc.models.retinaface import RetinaFace, extract_landmarks
from models.networks import Generator

random.seed(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False


def add_black_border_(images: Tensor, border: int = 1) -> None:
    """
    images: [N, C, H, W]
    """
    assert images.ndim == 4, f"expected [N, C, H, W], got {images.shape}"
    _, _, h, w = images.shape
    assert 0 <= border <= min(h, w) // 2, "border 太大了"

    images[:, :, :border, :] = -1
    images[:, :, -border:, :] = -1
    images[:, :, :, :border] = -1
    images[:, :, :, -border:] = -1


class FaceSwapper:
    def __init__(self, model_path: str, id_encoder_provider: IDEncoderProvider, device: str = "cuda") -> None:
        super().__init__()

        if not Path(model_path).exists():
            raise FileNotFoundError(f"ckpt file: {model_path} Not found")

        self.device = torch.device(device)

        self.is_onnx = str(model_path).lower().endswith(".onnx")

        if self.is_onnx:
            import onnxruntime as ort
            import tensorrt as trt

            print(trt.__version__)
            assert trt.Builder(trt.Logger())

            trt_engine_cache_path = "./trt_cache"
            Path(trt_engine_cache_path).mkdir(exist_ok=True, parents=True)

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 0

            print(f"Loading ONNX model from {model_path}")

            providers = [
                # (
                #     "TensorrtExecutionProvider",
                #     {
                #         "device_id": 0,
                #         "trt_fp16_enable": True,
                #         "trt_engine_cache_enable": True,
                #         "trt_engine_cache_path": "./trt_cache",
                #     },
                # ),
                ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}),
                "CPUExecutionProvider",
            ]
            self.ort_session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)

            inputs = self.ort_session.get_inputs()

            self.ort_faces_input_name = None
            self.ort_identity_input_name = None

            for inp in inputs:
                shape = inp.shape  # e.g. [1, H, W, C] or [1, N]

                if len(shape) == 4:
                    # 图像输入
                    self.ort_faces_input_name = inp.name
                    _, h, w, c = shape
                    if c != 3:
                        raise ValueError(f"Unexpected image channel: {c}")
                    if h != w:
                        raise ValueError("Only support square input")
                    self.img_resolution = h

                elif len(shape) == 2:
                    self.ort_identity_input_name = inp.name

            if self.ort_faces_input_name is None or self.ort_identity_input_name is None:
                raise RuntimeError("Failed to parse ONNX inputs")

            self.ort_output_name = self.ort_session.get_outputs()[0].name
        else:
            print(f"Loading ckpt from {model_path}")
            checkpoint: dict[str, Any] = torch.load(model_path, map_location="cpu", weights_only=False)

            self.training_iteration = checkpoint["iter"]

            print(f"ckpt Info:\n  {'iter':25}: {self.training_iteration}")
            print("net_g:")
            for k, v in checkpoint["net_g"]["network_cfg"].items():
                print(f"  {k:25}: {v}")
            print("net_d:")
            for k, v in checkpoint["net_d"]["network_cfg"].items():
                print(f"  {k:25}: {v}")

            self.img_resolution = checkpoint["net_g"]["network_cfg"]["img_resolution"]

            net_g = Generator(**checkpoint["net_g"]["network_cfg"])
            net_g.load_state_dict(checkpoint["net_g"]["state_dict"])
            self.net_g = net_g.to(device=self.device).eval()

        self.face_detector = RetinaFace(input_range=ImageInputRange.MINUS_ONE_TO_ONE).to(device=self.device).eval()
        self.id_encoder = IDEncoder(provider=id_encoder_provider).to(device=self.device).eval()
        self.face_masker = FaceMasker(input_range=ImageInputRange.MINUS_ONE_TO_ONE, mode="occlusion").to(device=self.device)  # occ模型在eval模式下无法正常推理

        self.alignment_template = torch.as_tensor(get_alignment_template(self.img_resolution), device=self.device, dtype=torch.float32)

    def extract_identity_embedding(self, image_path: str | Path) -> Tensor:
        """
        返回已经归一化的ID特征，形状为[1, 512]
        """

        image = decode_image(str(image_path)).to(device=self.device, dtype=torch.float)
        image.div_(127.5).sub_(1.0).unsqueeze_(0)  # [-1 ~ 1]

        detections, segment_lengths = self.face_detector.detect(image)
        source_landmarks = extract_landmarks(detections)
        aligned_faces, _ = align_faces(image, source_landmarks, segment_lengths, self.alignment_template, self.img_resolution)

        face_count = aligned_faces.size(0)

        if face_count == 0:
            raise AssertionError(f"未从{image_path}检测到任何面部")
        if face_count > 1:
            print("Warning: 检测到多张面部，仅使用第一张")
            aligned_faces = aligned_faces[:1]

        with torch.inference_mode():
            identity_embedding = self.id_encoder(center_crop_and_resize(aligned_faces, 0.102))

        return identity_embedding

    @staticmethod
    def _display_worker(display_queue: mp.Queue, fps: int = 25) -> None:
        delay = int(1000 / fps)
        while True:
            frames = display_queue.get()
            if frames is None:
                break

            for frame in frames:
                cv2.imshow("swapped", frame)
                cv2.waitKey(delay)

    # def swap_image_directory(self, directory: str | Path, identity_image_path: str | Path):

    @torch.inference_mode()
    def swap_video(self, video_path: str | Path, identity_image_path: str | Path) -> None:

        mp.set_start_method("spawn", force=True)
        display_queue = mp.Queue(maxsize=15)
        display_process = mp.Process(target=self._display_worker, args=(display_queue,), daemon=True)
        display_process.start()

        batch_size = 8

        identity_embedding = self.extract_identity_embedding(identity_image_path)

        for original_frames, faces, grid_theta, segment_lengths, _frame_count in VideoFaceExtractor(video_path, batch_size=batch_size, output_size=self.img_resolution, device=self.device):
            face_count = faces.size(0)
            if face_count == 0:
                continue
            original_frames.div_(127.5).sub_(1.0)
            faces.div_(127.5).sub_(1.0)

            if self.is_onnx:
                # onnx输入为1HWC,RGB,输出为1HWC,RGB
                faces_np = faces.permute(0, 2, 3, 1).detach().cpu().numpy()  # NCHW -> NHWC
                identity_embedding_np = identity_embedding.detach().cpu().numpy()

                face_batches = np.split(faces_np, faces_np.shape[0], axis=0)

                outputs = []
                for face_batch in face_batches:
                    output = self.ort_session.run(
                        [self.ort_output_name],
                        {self.ort_faces_input_name: face_batch, self.ort_identity_input_name: identity_embedding_np},
                    )[0]

                    output = output[..., ::-1]  # RGB -> BGR
                    outputs.append(output)

                frames = np.concatenate(outputs, axis=0)

                frames = np.concatenate([faces_np[..., ::-1], frames], axis=2)

                frames = ((frames + 1.0) * 127.5).astype(np.uint8)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    swapped_faces: Tensor = self.net_g(faces, identity_embedding.expand(face_count, -1))
                    face_mask = self.face_masker(faces)
                    face_mask = F.gaussian_blur(face_mask, [7, 7], [13.0, 13.0])
                    swapped_faces = faces * (1.0 - face_mask) + swapped_faces * face_mask
                    # mask = mask * 2.0 - 1.0
                    # swapped_faces = faces
                    # add_black_border_(swapped_faces)
                    swapped_faces = restore_faces_to_original(original_frames.clone(), swapped_faces, grid_theta, segment_lengths)
                    swapped_faces = torch.cat((original_frames, swapped_faces), dim=3)
                    swapped_faces = torch.nn.functional.interpolate(swapped_faces, scale_factor=0.25, mode="bilinear", align_corners=False)

                    # swapped_faces = torch.cat((faces, swapped_faces), dim=3)
                    swapped_faces = swapped_faces.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)[:, [2, 1, 0], :, :]  # RGB -> BGR
                    swapped_faces = swapped_faces.permute(0, 2, 3, 1)  # NCHW -> NHWC
                    frames = swapped_faces.to(device="cpu", dtype=torch.uint8).numpy()

            display_queue.put(frames)

        display_queue.put(None)
        display_process.join()


if __name__ == "__main__":
    video_path = "/opt/share/deepfake/dataset_1/oneman/1.mp4"
    # video_path = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    # video_path = "/opt/share/deepfake/linshi/录屏素材/2023-02-17(袁忠 曹属馨)/20230217-160724910曹属馨(电脑录屏).mp4"
    # video_path = "/opt/share/deepfake/linshi/录屏素材/2023-06-20(邓叮玲 许馨文)/20230620-155807793邓叮玲(电脑录屏).mp4"
    # video_path = "/opt/share/deepfake/linshi/录屏素材/2023-08-28(李小姐 邱小姐)/20230828-122302李小姐(电脑录屏).mp4"
    # video_path = "/opt/share/deepfake/linshi/录屏素材/2025-07-03(蒋女士 林嘉惠 王钰 符媛媛 石星儿)/王钰(电脑录屏).mp4"

    # identity_image_path = "/home/liaohaixun/swap/IDAssets/wuyanzu.png"
    # identity_image_path = "/home/liaohaixun/swap/IDAssets/周杰伦.png"
    # identity_image_path = "/home/liaohaixun/swap/IDAssets/陈冠希.png"
    # identity_image_path = "/home/liaohaixun/swap/faceset/ljx/arcface_pts/6000_0.png"
    # identity_image_path = "/home/liaohaixun/swap/IDAssets/安妮·海瑟薇.png"
    identity_image_path = "/home/liaohaixun/swap/IDAssets/刘亦菲.jpg"

    swapper = FaceSwapper("onnx_export/512-MS1MV3_ARCFACE_R50_FP16-510000-20260630_141426.onnx", IDEncoderProvider.MS1MV3_ARCFACE_R50_FP16)

    swapper.swap_video(video_path, identity_image_path)
