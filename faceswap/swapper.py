import argparse
import multiprocessing as mp
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as F
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from torchvision.io import ImageReadMode, decode_image

from misc.face_alignment import align_faces, make_alignment_grid_theta, make_ffhq_to_arcface_112_grid, restore_faces_to_original, transform_sampling_grid
from misc.models import ImageInputRange
from misc.models.face_mask import FaceMasker
from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from misc.models.retinaface import RetinaFace, extract_landmarks

from .contracts import ONNX_CONTRACT
from .inference import get_ffhq_alignment_template, load_generator

random.seed(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False


class FaceSwapper:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("仅支持 cpu 或 cuda 设备")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请指定 device='cpu'")
        self.bf16 = False
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                # Turing 等设备的软件 BF16 模拟比 FP32 更慢，仅启用原生 BF16。
                self.bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        self.is_onnx = str(model_path).lower().endswith(".onnx")

        if self.is_onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            providers: list[str | tuple[str, dict[str, object]]] = ["CPUExecutionProvider"]
            self.ort_device_id = 0
            if self.device.type == "cuda":
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("ONNX Runtime 未提供 CUDAExecutionProvider，不能以 CUDA 模式加载 ONNX")
                self.ort_device_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
                # CUDA 模式不允许算子静默回退 CPU；否则性能回归很难被测试发现。
                sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                providers = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": self.ort_device_id,
                            "use_tf32": 0,
                            "user_compute_stream": str(torch.cuda.current_stream(self.device).cuda_stream),
                        },
                    )
                ]
            self.ort_session = ort.InferenceSession(str(model_path), providers=providers, sess_options=sess_options)
            metadata = self.ort_session.get_modelmeta().custom_metadata_map
            for key, expected in (ONNX_CONTRACT | {"faceswap.kind": "generator"}).items():
                if metadata.get(key) != expected:
                    raise ValueError(f"ONNX 推理约定不匹配：{key}={metadata.get(key)!r}，请用当前 export.py 重新导出")
            try:
                self.id_encoder_provider = IDEncoderProvider[metadata["faceswap.provider"]]
                self.training_iteration = int(metadata["faceswap.iter"])
            except (KeyError, ValueError) as error:
                raise ValueError("ONNX 缺少有效的身份编码器或迭代信息，请重新导出") from error
            self.ort_layout = metadata.get("faceswap.layout")
            if self.ort_layout not in {"NHWC", "NCHW"}:
                raise ValueError("ONNX 缺少有效的图像布局，请重新导出")
            inputs = {item.name: item for item in self.ort_session.get_inputs()}
            outputs = self.ort_session.get_outputs()
            if set(inputs) != {"faces", "identity"} or len(outputs) != 1 or outputs[0].name != "swapped_faces":
                raise ValueError("ONNX 输入/输出不符合当前导出接口")
            shape = inputs["faces"].shape
            if len(shape) != 4 or any(not isinstance(value, int) or value <= 0 for value in shape):
                raise ValueError(f"ONNX 图像输入必须为固定的四维正整数形状，实际为 {shape}")
            if self.ort_layout == "NHWC":
                self.ort_batch_size, height, width, channels = shape
            else:
                self.ort_batch_size, channels, height, width = shape
            if channels != 3 or height != width or inputs["identity"].shape != [self.ort_batch_size, 512]:
                raise ValueError("ONNX 要求方形 RGB 图像和同 batch 的 512 维身份特征")
            if outputs[0].shape != shape or any(item.type != "tensor(float)" for item in [*inputs.values(), *outputs]):
                raise ValueError("ONNX 要求 FP32 输入输出，且输出图像形状与输入一致")
            self.img_resolution = height
        else:
            net_g, self.id_encoder_provider, self.training_iteration = load_generator(model_path)
            self.img_resolution = net_g.network_cfg["img_resolution"]
            self.net_g = net_g.to(self.device).eval()

        print(f"Loaded iter={self.training_iteration}, provider={self.id_encoder_provider.name}, alignment=FFHQ")
        self.face_detector = RetinaFace(input_range=ImageInputRange.ZERO_TO_255).to(self.device).eval()
        self.id_encoder = IDEncoder(self.id_encoder_provider).to(self.device).eval()
        self.face_masker = FaceMasker(input_range=ImageInputRange.MINUS_ONE_TO_ONE, mode="occlusion").to(self.device)  # occ 模型在 eval 模式下无法正常推理
        self.alignment_template = get_ffhq_alignment_template(self.img_resolution, self.device)
        self.identity_encoder_grid = make_ffhq_to_arcface_112_grid(self.img_resolution, 1, self.device)

    @torch.inference_mode()
    def extract_identity_embedding(self, image_path: str | Path) -> Tensor:
        """使用训练端相同的 FFHQ -> ArcFace 112 映射提取 [1,512] 身份特征。"""
        image = decode_image(str(image_path), mode=ImageReadMode.RGB).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        normalized = image / 127.5 - 1.0
        detections, segment_lengths = self.face_detector.detect(image)
        face_count = segment_lengths[0]
        if face_count == 0:
            raise ValueError(f"未从 {image_path} 检测到任何面部")
        if face_count > 1:
            print("Warning: 检测到多张面部，仅使用第一张")

        source_landmarks = extract_landmarks(detections)[:1]
        raw_to_ffhq_theta = make_alignment_grid_theta(
            source_landmarks,
            self.alignment_template,
            tuple(image.shape[-2:]),
            self.img_resolution,
        )
        # 将 raw->FFHQ 与 FFHQ->ArcFace 合成一个采样网格，只对源图做一次 bilinear 重采样。
        identity_grid = transform_sampling_grid(self.identity_encoder_grid, raw_to_ffhq_theta)
        identity_faces = NF.grid_sample(normalized, identity_grid, mode="bilinear", padding_mode="border", align_corners=False)
        return self.id_encoder(identity_faces)

    @torch.inference_mode()
    def swap_faces(self, faces: Tensor, identity_embedding: Tensor) -> Tensor:
        """输入 FFHQ 对齐的 RGB NCHW 人脸（[-1,1]），返回同布局的换脸结果。"""
        if faces.ndim != 4 or tuple(faces.shape[1:]) != (3, self.img_resolution, self.img_resolution):
            raise ValueError(f"faces 必须为 [N,3,{self.img_resolution},{self.img_resolution}]")
        if tuple(identity_embedding.shape) != (1, 512):
            raise ValueError("identity_embedding 必须为 [1,512]")
        faces = faces.to(device=self.device, dtype=torch.float32)
        identity_embedding = identity_embedding.to(device=self.device, dtype=torch.float32)
        if faces.size(0) == 0:
            return faces
        if not self.is_onnx:
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                return self.net_g(faces, identity_embedding.expand(faces.size(0), -1)).float()

        if self.device.type == "cuda":
            return self._swap_faces_onnx_cuda(faces, identity_embedding)

        images = faces.permute(0, 2, 3, 1).contiguous() if self.ort_layout == "NHWC" else faces.contiguous()
        images_np = images.numpy()
        identity_np = np.repeat(identity_embedding.numpy(), self.ort_batch_size, axis=0)
        results = []
        for start in range(0, len(images_np), self.ort_batch_size):
            batch = images_np[start : start + self.ort_batch_size]
            count = len(batch)
            if count < self.ort_batch_size:
                batch = np.concatenate((batch, np.repeat(batch[-1:], self.ort_batch_size - count, axis=0)))
            result = self.ort_session.run(["swapped_faces"], {"faces": np.ascontiguousarray(batch), "identity": identity_np})[0]
            results.append(result[:count])
        swapped = torch.from_numpy(np.concatenate(results))
        return swapped.permute(0, 3, 1, 2).contiguous() if self.ort_layout == "NHWC" else swapped

    def _swap_faces_onnx_cuda(self, faces: Tensor, identity_embedding: Tensor) -> Tensor:
        """使用 IOBinding 让 ORT 直接读写 CUDA Tensor，避免热路径经过 CPU/NumPy。"""
        images = faces.permute(0, 2, 3, 1).contiguous() if self.ort_layout == "NHWC" else faces.contiguous()
        identity_batch = identity_embedding.expand(self.ort_batch_size, -1).contiguous()
        swapped = torch.empty_like(images)

        for start in range(0, images.shape[0], self.ort_batch_size):
            stop = min(start + self.ort_batch_size, images.shape[0])
            count = stop - start
            input_batch = images[start:stop]
            output_batch = swapped[start:stop]

            if count < self.ort_batch_size:
                padded_input = torch.empty((self.ort_batch_size, *images.shape[1:]), device=self.device, dtype=torch.float32)
                padded_input[:count].copy_(input_batch)
                padded_input[count:].copy_(input_batch[-1:].expand(self.ort_batch_size - count, *input_batch.shape[1:]))
                bound_input = padded_input
                bound_output = torch.empty_like(padded_input)
            else:
                bound_input = input_batch
                bound_output = output_batch

            io_binding = self.ort_session.io_binding()
            io_binding.bind_input(
                "faces",
                "cuda",
                self.ort_device_id,
                np.float32,
                tuple(bound_input.shape),
                bound_input.data_ptr(),
            )
            io_binding.bind_input(
                "identity",
                "cuda",
                self.ort_device_id,
                np.float32,
                tuple(identity_batch.shape),
                identity_batch.data_ptr(),
            )
            io_binding.bind_output(
                "swapped_faces",
                "cuda",
                self.ort_device_id,
                np.float32,
                tuple(bound_output.shape),
                bound_output.data_ptr(),
            )
            io_binding.synchronize_inputs()
            self.ort_session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            if count < self.ort_batch_size:
                output_batch.copy_(bound_output[:count])

        return swapped.permute(0, 3, 1, 2).contiguous() if self.ort_layout == "NHWC" else swapped

    @staticmethod
    def _display_worker(display_queue: mp.Queue, fps: int = 25) -> None:
        while True:
            frames = display_queue.get()
            if frames is None:
                break
            for frame in frames:
                cv2.imshow("swapped", frame)
                cv2.waitKey(max(1, int(1000 / fps)))
        cv2.destroyAllWindows()

    @staticmethod
    def _make_preview(original_frames: Tensor, swapped_faces: Tensor, grid_theta: Tensor, segment_lengths: list[int]) -> Tensor:
        """保留全分辨率还原，先缩小左右两侧再拼接，避免复制/拼接全尺寸视频。"""
        if original_frames.shape[-1] % 4:
            # 原实现的左右分界可能落在 bilinear 采样点上，奇数预览宽度保留原采样方式。
            restored = original_frames.clone()
            if swapped_faces.size(0):
                restored = restore_faces_to_original(restored, swapped_faces, grid_theta, segment_lengths)
            return NF.interpolate(torch.cat((original_frames, restored), dim=3), scale_factor=0.25, mode="bilinear", align_corners=False)
        original_preview = NF.interpolate(original_frames, scale_factor=0.25, mode="bilinear", align_corners=False)
        if not swapped_faces.size(0):
            return torch.cat((original_preview, original_preview), dim=3)
        restored = restore_faces_to_original(original_frames, swapped_faces, grid_theta, segment_lengths)
        swapped_preview = NF.interpolate(restored, scale_factor=0.25, mode="bilinear", align_corners=False)
        return torch.cat((original_preview, swapped_preview), dim=3)

    @torch.inference_mode()
    def swap_video(self, video_path: str | Path, identity_image_path: str | Path, batch_size: int = 8, full_resolution_detection: bool = False) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        identity_embedding = self.extract_identity_embedding(identity_image_path)
        decoder = VideoDecoder(str(video_path), device=self.device)
        context = mp.get_context("spawn")
        display_queue = context.Queue(maxsize=15)
        display_process = context.Process(target=self._display_worker, args=(display_queue,), daemon=True)
        display_process.start()
        try:
            for start in range(0, len(decoder), batch_size):
                original_frames = decoder[start : start + batch_size].data.to(device=self.device, dtype=torch.float32)
                # 大帧复用检测器原生 resize/padding 路径；检测结果已还原到原图坐标。
                # 小帧直接检测，避免上采样或 padding 反而增加工作量。
                height, width = original_frames.shape[-2:]
                resize_detection = not full_resolution_detection and height * width > self.face_detector.config["image_size"] ** 2
                detection_input = list(original_frames) if resize_detection else original_frames
                detections, segment_lengths = self.face_detector.detect(detection_input, confidence_threshold=0.99, iou_threshold=0.2, min_face_size=(256, 256))
                faces, grid_theta = align_faces(original_frames, extract_landmarks(detections), segment_lengths, self.alignment_template, self.img_resolution)
                original_frames.div_(127.5).sub_(1.0)
                faces.div_(127.5).sub_(1.0)
                swapped_faces = faces
                if faces.size(0):
                    swapped_faces = self.swap_faces(faces, identity_embedding)
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                        face_mask = self.face_masker(faces)
                    face_mask = F.gaussian_blur(face_mask.float(), [7, 7], [13.0, 13.0])
                    swapped_faces = faces * (1.0 - face_mask) + swapped_faces * face_mask
                frames = self._make_preview(original_frames, swapped_faces, grid_theta, segment_lengths)
                frames = frames.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)[:, [2, 1, 0]]
                display_queue.put(frames.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy())
            display_queue.put(None)
            display_process.join()
        finally:
            if display_process.is_alive():
                display_process.terminate()
                display_process.join()
            display_queue.cancel_join_thread()
            display_queue.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="使用当前 checkpoint 或配套 ONNX 预览视频换脸")
    parser.add_argument("--model", required=True, help="当前训练 checkpoint，或当前 export.py 导出的 ONNX")
    parser.add_argument("--video", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8, help="每批解码和检测的视频帧数")
    parser.add_argument("--full-resolution-detection", action="store_true", help="按原视频分辨率检测，保留原有检测精度；默认大帧使用检测器原生缩放")
    args = parser.parse_args()
    FaceSwapper(args.model, device=args.device).swap_video(args.video, args.identity, batch_size=args.batch_size, full_resolution_detection=args.full_resolution_detection)


if __name__ == "__main__":
    main()
