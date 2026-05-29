from pathlib import Path
from typing import Any
import multiprocessing as mp

import torch
from torch import Tensor
import torchvision.transforms.functional as F
from misc.facealign import face_align_batch, FaceExtractorVideo, restore_faces_to_original, FaceAligner, zoom_in
from misc.models.retinaface import get_pts, RetinaFace
from misc.models.idencoder import IDEncoder, get_align_landmarks, PROVIDER
from misc.models.face_parsing import FaceParsing
from .networks import Generator
from torchvision.io import decode_image
from tqdm import tqdm
from misc.utils import ImageFolder

import cv2
import numpy as np


def add_black_border_batch(imgs: Tensor, border: int = 1):
    """
    imgs: [N, C, H, W]
    """
    assert imgs.ndim == 4, f"expected [N, C, H, W], got {imgs.shape}"
    _, _, h, w = imgs.shape
    assert 0 <= border <= min(h, w) // 2, "border 太大了"

    imgs[:, :, :border, :] = -1
    imgs[:, :, -border:, :] = -1
    imgs[:, :, :, :border] = -1
    imgs[:, :, :, -border:] = -1


class Swap:
    def __init__(self, model_path: str, idencoder_provider: PROVIDER, device: str = "cuda") -> None:
        super().__init__()

        if not Path(model_path).exists():
            raise FileNotFoundError(f"ckpt file: {model_path} Not found")

        self.device = torch.device(device)

        self.is_onnx = str(model_path).lower().endswith(".onnx")

        if self.is_onnx:
            import tensorrt as trt
            import onnxruntime as ort

            print(trt.__version__)
            assert trt.Builder(trt.Logger())

            TRT_ENGINE_CACHE_PATH = "./trt_cache"
            Path(TRT_ENGINE_CACHE_PATH).mkdir(exist_ok=True, parents=True)

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 0

            print(f"Loading ONNX model from {model_path}")

            providers = (
                [
                    (
                        "TensorrtExecutionProvider",
                        {
                            "device_id": 0,
                            "trt_fp16_enable": True,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": "./trt_cache",
                        },
                    ),
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                        },
                    ),
                    "CPUExecutionProvider",
                ]
                if self.device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.ort_session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)

            inputs = self.ort_session.get_inputs()

            self.ort_input_faces = None
            self.ort_input_id = None

            for inp in inputs:
                shape = inp.shape  # e.g. [1, H, W, C] or [1, N]

                if len(shape) == 4:
                    # 图像输入
                    self.ort_input_faces = inp.name
                    _, h, w, c = shape
                    if c != 3:
                        raise ValueError(f"Unexpected image channel: {c}")
                    if h != w:
                        raise ValueError("Only support square input")
                    self.img_resolution = h

                elif len(shape) == 2:
                    self.ort_input_id = inp.name

            if self.ort_input_faces is None or self.ort_input_id is None:
                raise RuntimeError("Failed to parse ONNX inputs")

            self.ort_output = self.ort_session.get_outputs()[0].name
        else:
            print(f"Loading ckpt from {model_path}")
            model_path: dict[str, Any] = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

            self.iter = model_path["iter"]

            print(f"ckpt Info:\n  {'iter':25}: {self.iter}")
            print("net_g:")
            for k, v in model_path["net_g"]["network_cfg"].items():
                print(f"  {k:25}: {v}")
            print("net_d:")
            for k, v in model_path["net_d"]["network_cfg"].items():
                print(f"  {k:25}: {v}")

            self.img_resolution = model_path["net_g"]["network_cfg"]["img_resolution"]

            net_g = Generator(**model_path["net_g"]["network_cfg"])
            net_g.load_state_dict(model_path["net_g"]["state_dict"])
            self.net_g = net_g.to(device=self.device).eval()

        self.facedetch = RetinaFace(from_normalized=True).to(device=self.device).eval()
        self.idencoder = IDEncoder(provider=idencoder_provider).to(device=self.device).eval()
        self.face_mask = FaceParsing(range_norm=True, occ=True).to(device=self.device)  # occ模型在eval模式下无法正常推理

        self.dst_pts = torch.tensor(get_align_landmarks(self.img_resolution), device=self.device)

    def extract_id_feats_from_image(self, fp: str) -> Tensor:
        """
        返回已经归一化的ID特征，形状为[1, 512]
        """

        image = decode_image(fp).to(device=self.device, dtype=torch.float)
        image.div_(127.5).sub_(1.0).unsqueeze_(0)  # [-1 ~ 1]

        detected, offset = self.facedetch.detector(image)
        src_pts = get_pts(detected)
        dst_pts = get_align_landmarks(self.img_resolution)
        dst_pts = torch.tensor(dst_pts, device=self.device)
        align_face, _ = face_align_batch(image, src_pts, offset, dst_pts, self.img_resolution)

        N = align_face.size(0)

        if N == 0:
            raise AssertionError(f"未从{fp}检测到任何面部")
        if N > 1:
            print("Warning: 检测到多张面部，仅使用第一张")
            align_face = align_face[:1]

        with torch.inference_mode():
            id_emb = self.idencoder(zoom_in(align_face, 0.102))

        return id_emb

    @staticmethod
    def _display_worker(q: mp.Queue, fps=25):
        delay = int(1000 / fps)
        while True:
            frames = q.get()
            if frames is None:
                break

            for frame in frames:
                cv2.imshow("swaped", frame)
                cv2.waitKey(delay)

    # def swap_image_folder(self, fp: str, id_fp: str):

    @torch.inference_mode()
    def swap_video(self, vfp: str, id_fp: str):

        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=15)
        p = mp.Process(target=self._display_worker, args=(q,), daemon=True)
        p.start()

        batch_size = 8

        id_emb = self.extract_id_feats_from_image(id_fp)

        for org_frames, faces, norm_theta, offset, nb in FaceExtractorVideo(vfp, batch_size=batch_size, align_size=self.img_resolution, device=self.device):
            N = faces.size(0)
            if N == 0:
                continue
            org_frames.div_(127.5).sub_(1.0)
            faces.div_(127.5).sub_(1.0)

            if self.is_onnx:
                # onnx输入为1HWC,RGB,输出为1HWC,RGB
                faces_np = faces.permute(0, 2, 3, 1).detach().cpu().numpy()  # NCHW -> NHWC
                id_np = id_emb.detach().cpu().numpy()

                faces = np.split(faces_np, faces_np.shape[0], axis=0)

                outputs = []
                for face in faces:
                    output = self.ort_session.run(
                        [self.ort_output],
                        {self.ort_input_faces: face, self.ort_input_id: id_np},
                    )[0]

                    output = output[..., ::-1]  # RGB -> BGR
                    outputs.append(output)

                frames = np.concatenate(outputs, axis=0)

                frames = np.concatenate([faces_np[..., ::-1], frames], axis=2)

                frames = ((frames + 1.0) * 127.5).astype(np.uint8)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    swap_face: Tensor = self.net_g(faces, id_emb.expand(N, -1))
                    mask = self.face_mask(faces)
                    mask = F.gaussian_blur(mask, 7, 13)
                    swap_face = faces * (1.0 - mask) + swap_face * mask
                    # mask = mask * 2.0 - 1.0
                    # swap_face = faces
                    add_black_border_batch(swap_face)
                    swap_face = restore_faces_to_original(org_frames, swap_face, norm_theta, offset)
                    swap_face = torch.nn.functional.interpolate(swap_face, scale_factor=0.25, mode="bilinear", align_corners=False)
                    # swap_face = torch.cat((faces, swap_face), dim=3)
                    swap_face = swap_face.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)[:, [2, 1, 0], :, :]  # RGB -> BGR
                    swap_face = swap_face.permute(0, 2, 3, 1)  # NCHW -> NHWC
                    frames = swap_face.to(device="cpu", dtype=torch.uint8).numpy()

            q.put(frames)

        q.put(None)
        p.join()


if __name__ == "__main__":
    # video = "/opt/share/deepfake/dataset_1/oneman/1.mp4"
    video = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    # video = "/opt/share/deepfake/linshi/录屏素材/2023-02-17(袁忠 曹属馨)/20230217-160724910曹属馨(电脑录屏).mp4"
    # video = "/opt/share/deepfake/linshi/录屏素材/2023-06-20(邓叮玲 许馨文)/20230620-155807793邓叮玲(电脑录屏).mp4"
    # video = "/opt/share/deepfake/linshi/录屏素材/2023-08-28(李小姐 邱小姐)/20230828-122302李小姐(电脑录屏).mp4"
    # video = "/opt/share/deepfake/linshi/录屏素材/2025-07-03(蒋女士 林嘉惠 王钰 符媛媛 石星儿)/王钰(电脑录屏).mp4"

    # id = "/home/liaohaixun/swap/IDAssets/wuyanzu.png"
    id = "/home/liaohaixun/swap/IDAssets/周杰伦.png"
    # id = "/home/liaohaixun/swap/IDAssets/陈冠希.png"
    # id = "/home/liaohaixun/swap/faceset/ljx/arcface_pts/6000_0.png"
    # id = "/home/liaohaixun/swap/IDAssets/安妮·海瑟薇.png"
    # id = "/home/liaohaixun/swap/IDAssets/2025-06-15 18_19_50小树🌿人间体验卡限时掉落✨ _3.jpg"
    # id = "w700d1q75cms.jpg"

    swapper = Swap("train_log/256_BLENDFACE_FFHQ/ckpt/260000.pth", PROVIDER.MS1MV3_ARCFACE_R100_FP16)

    swapper.swap_video(video, id)
