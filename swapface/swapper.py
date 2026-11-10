import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as F
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from torchvision.io import ImageReadMode, decode_image

from misc.face_alignment import align_faces, invert_grid_theta, make_alignment_grid_theta, make_ffhq_to_arcface_112_grid, transform_sampling_grid
from misc.models import ImageInputRange
from misc.models.face_mask import FaceMasker
from misc.models.id_encoder import IDEncoder, IDEncoderProvider
from misc.models.retinaface import RetinaFace, extract_landmarks

from .contracts import ONNX_CONTRACT
from .inference import get_arcface_alignment_template, get_ffhq_alignment_template, load_generator


def _configure_swapper_runtime() -> None:
    random.seed(42)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


class SwapFace:
    def __init__(self, model_path: str, device: str = "cuda", lookup_coarse: bool = False) -> None:
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
        self.lookup_coarse = lookup_coarse
        if self.is_onnx and self.lookup_coarse:
            raise ValueError("lookup_coarse 仅支持 checkpoint；当前 ONNX 只导出完整 Generator 输出")

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
            self.face_alignment = metadata.get("swapface.face_alignment")
            if self.face_alignment not in ("ffhq", "arcface"):
                raise ValueError(f"ONNX 对齐类型无效：{self.face_alignment!r}")
            for key, expected in (ONNX_CONTRACT | {"swapface.face_alignment": self.face_alignment, "swapface.kind": "generator"}).items():
                if metadata.get(key) != expected:
                    raise ValueError(f"ONNX 推理约定不匹配：{key}={metadata.get(key)!r}，请用当前 export.py 重新导出")
            try:
                self.id_encoder_provider = IDEncoderProvider[metadata["swapface.provider"]]
                self.training_step = int(metadata["swapface.step"])
            except (KeyError, ValueError) as error:
                raise ValueError("ONNX 缺少有效的身份编码器或 step 信息，请重新导出") from error
            self.ort_layout = metadata.get("swapface.layout")
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
            net_g, self.id_encoder_provider, self.training_step = load_generator(model_path)
            self.img_resolution = net_g.network_cfg["img_resolution"]
            self.net_g = net_g.to(self.device).eval()
            self.face_alignment = getattr(net_g, "face_alignment", "ffhq")

        print(f"Loaded step={self.training_step}, provider={self.id_encoder_provider.name}, alignment={self.face_alignment.upper()}")
        self.face_detector = RetinaFace(input_range=ImageInputRange.ZERO_TO_255).to(self.device).eval()
        self.id_encoder = IDEncoder(self.id_encoder_provider).to(self.device).eval()
        self.face_masker = FaceMasker(input_range=ImageInputRange.MINUS_ONE_TO_ONE, mode="occlusion").to(self.device)  # occ 模型在 eval 模式下无法正常推理
        self.alignment_template = (
            get_ffhq_alignment_template(self.img_resolution, self.device)
            if self.face_alignment == "ffhq"
            else get_arcface_alignment_template(self.img_resolution, self.device)
        )
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
        raw_to_alignment_theta = make_alignment_grid_theta(
            source_landmarks,
            self.alignment_template,
            tuple(image.shape[-2:]),
            self.img_resolution,
        )
        # FFHQ 模式复合一次原图->FFHQ->ArcFace 的采样；ArcFace 模式直接 112 对齐。
        if self.face_alignment == "ffhq":
            identity_grid = transform_sampling_grid(self.identity_encoder_grid, raw_to_alignment_theta)
        else:
            identity_grid = NF.affine_grid(raw_to_alignment_theta, (1, 3, 112, 112), align_corners=False)
        identity_faces = NF.grid_sample(normalized, identity_grid, mode="bilinear", padding_mode="border", align_corners=False)
        return self.id_encoder(identity_faces)

    @torch.inference_mode()
    def swap_faces(self, faces: Tensor, identity_embedding: Tensor) -> Tensor:
        """输入与 checkpoint 对齐模式一致的 RGB NCHW 人脸（[-1,1]）。"""
        if faces.ndim != 4 or tuple(faces.shape[1:]) != (3, self.img_resolution, self.img_resolution):
            raise ValueError(f"faces 必须为 [N,3,{self.img_resolution},{self.img_resolution}]")
        if tuple(identity_embedding.shape) != (1, 512):
            raise ValueError("identity_embedding 必须为 [1,512]")
        faces = faces.to(device=self.device, dtype=torch.float32)
        identity_embedding = identity_embedding.to(device=self.device, dtype=torch.float32)
        if faces.size(0) == 0:
            return faces
        if not self.is_onnx:
            identity_batch = identity_embedding.expand(faces.size(0), -1)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                if self.lookup_coarse:
                    swapped = self.net_g.coarse(faces, identity_batch)
                    if swapped.shape[-2:] != faces.shape[-2:]:
                        swapped = NF.interpolate(swapped, size=faces.shape[-2:], mode="bilinear", align_corners=False)
                    return swapped.float()
                return self.net_g(faces, identity_batch).float()

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
    def _make_preview(original_frames: Tensor, swapped_faces: Tensor, grid_theta: Tensor, segment_lengths: list[int]) -> Tensor:
        """返回原尺寸 RGB uint8 左右对比图，仅在受影响的人脸 ROI 内进行浮点还原。"""
        height, width = original_frames.shape[-2:]
        preview = torch.cat((original_frames, original_frames), dim=3)
        if not swapped_faces.size(0):
            return preview

        # 映射对齐裁剪的四个角到源视频，外扩 2 个采样像素以容纳 bicubic 插值边缘。
        pad = 1.0 + 4.0 / swapped_faces.shape[-1]
        corners = grid_theta.new_tensor([[-pad, -pad], [-pad, pad], [pad, -pad], [pad, pad]])
        coords = torch.matmul(corners, grid_theta[:, :, :2].transpose(1, 2)) + grid_theta[:, None, :, 2]
        coords = (coords + 1.0) * coords.new_tensor([width / 2, height / 2]) - 0.5
        lo = coords.amin(dim=1).floor().to(torch.int32)
        hi = (coords.amax(dim=1) + 2).ceil().to(torch.int32)
        # 高级索引会产生副本，必须直接夹取各列才能更新边界。
        lo[:, 0].clamp_(0, width)
        lo[:, 1].clamp_(0, height)
        hi[:, 0].clamp_(0, width)
        hi[:, 1].clamp_(0, height)
        boxes = torch.cat((lo, hi), dim=1)
        boxes_list = boxes.cpu().tolist()
        inverses = invert_grid_theta(grid_theta)

        offset = 0
        for frame_index, count in enumerate(segment_lengths):
            regions = [(offset + i, *boxes_list[offset + i]) for i in range(count)]
            offset += count
            regions = [(i, x0, y0, x1, y1) for i, x0, y0, x1, y1 in regions if x1 > x0 and y1 > y0]
            if not regions:
                continue
            # 以所有人脸 ROI 的并集为画布，保留原版多脸重叠时的加权合成语义。
            ux0, uy0 = min(v[1] for v in regions), min(v[2] for v in regions)
            ux1, uy1 = max(v[3] for v in regions), max(v[4] for v in regions)
            total = swapped_faces.new_zeros((3, uy1 - uy0, ux1 - ux0))
            weights = swapped_faces.new_zeros((1, uy1 - uy0, ux1 - ux0))
            for face_index, x0, y0, x1, y1 in regions:
                roi_w, roi_h = x1 - x0, y1 - y0
                inverse = inverses[face_index]
                # 将 ROI 局部坐标映射回源图规范化坐标，再复合原图->人脸变换。
                sx, sy = roi_w / width, roi_h / height
                tx, ty = (2 * x0 + roi_w) / width - 1, (2 * y0 + roi_h) / height - 1
                roi_theta = torch.stack(
                    (inverse[:, 0] * sx, inverse[:, 1] * sy, inverse[:, 0] * tx + inverse[:, 1] * ty + inverse[:, 2]),
                    dim=1,
                ).unsqueeze(0)
                grid = NF.affine_grid(roi_theta, (1, 3, roi_h, roi_w), align_corners=False)
                face = swapped_faces[face_index:face_index + 1]
                sampled = NF.grid_sample(face, grid, mode="bicubic", padding_mode="zeros", align_corners=False)
                mask = NF.grid_sample(torch.ones_like(face[:, :1]), grid, mode="bicubic", padding_mode="zeros", align_corners=False)
                ys, xs = slice(y0 - uy0, y1 - uy0), slice(x0 - ux0, x1 - ux0)
                total[:, ys, xs].add_(sampled[0])
                weights[:, ys, xs].add_(mask[0])

            roi = preview[frame_index, :, uy0:uy1, width + ux0:width + ux1]
            original = roi.float().div_(127.5).sub_(1.0)
            mixed = torch.where(weights > 0.5, total / weights.clamp(min=1e-6), original)
            roi.copy_(mixed.add_(1.0).mul_(127.5).clamp_(0, 255).to(torch.uint8))
        return preview

    @torch.inference_mode()
    def _process_batch(self, frames: Tensor, identity_embedding: Tensor, full_resolution_detection: bool, profile: dict[str, float] | None = None) -> np.ndarray:
        def mark(stage: str, started: float) -> float:
            if profile is None:
                return started
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            profile[stage] = profile.get(stage, 0.0) + now - started
            return now

        started = time.perf_counter()
        height, width = frames.shape[-2:]
        # 检测与采样继续使用原始 0..255 FP32，确保模型输入及插值规则不变。
        float_frames = frames.to(dtype=torch.float32)
        resize_detection = not full_resolution_detection and height * width > self.face_detector.config["image_size"] ** 2
        detection_input = list(float_frames) if resize_detection else float_frames
        detections, lengths = self.face_detector.detect(detection_input, confidence_threshold=0.99, iou_threshold=0.2, min_face_size=(256, 256))
        started = mark("detect", started)
        faces, theta = align_faces(float_frames, extract_landmarks(detections), lengths, self.alignment_template, self.img_resolution)
        del detection_input, float_frames
        faces.div_(127.5).sub_(1.0)
        started = mark("align", started)
        if faces.size(0):
            swapped = self.swap_faces(faces, identity_embedding)
        else:
            swapped = faces
        started = mark("generator", started)
        if faces.size(0):
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                face_mask = self.face_masker(faces)
            face_mask = F.gaussian_blur(face_mask.float(), [7, 7], [13.0, 13.0])
            swapped = faces * (1.0 - face_mask) + swapped * face_mask
        started = mark("mask", started)
        preview = self._make_preview(frames, swapped, theta, lengths)
        started = mark("composite", started)
        result = preview[:, [2, 1, 0]].permute(0, 2, 3, 1).contiguous().cpu().numpy()
        mark("transfer", started)
        return result

    @torch.inference_mode()
    def swap_video(
        self,
        video_path: str | Path,
        identity_image_path: str | Path,
        batch_size: int = 1,
        full_resolution_detection: bool = False,
        start: float = 0.0,
        frames: int | None = None,
        profile: bool = False,
    ) -> None:
        if batch_size <= 0 or not np.isfinite(start) or start < 0 or (frames is not None and frames <= 0):
            raise ValueError("batch_size 和 frames 必须为正数，start 必须为非负有限秒数")
        identity_embedding = self.extract_identity_embedding(identity_image_path)
        decoder = VideoDecoder(str(video_path), device=self.device)
        source_fps = decoder.metadata.average_fps
        if source_fps is None or not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"视频缺少有效帧率：average_fps={source_fps!r}")
        source_fps = float(source_fps)
        start_index = round(start * source_fps)  # 简单按平均帧率换算；VFR 视频可能不是精确时间点。
        stop_index = min(len(decoder), start_index + frames if frames is not None else len(decoder))
        if start_index >= len(decoder):
            raise ValueError(f"起始时间超出视频范围：start={start}s")
        print(f"Preview FPS={source_fps:.3f}, frames={stop_index - start_index}, start_index={start_index} (press Q/Esc to quit)")

        stats: dict[str, float] | None = {} if profile else None
        done = 0
        compute_seconds = 0.0
        started = time.perf_counter()
        try:
            # 解码用连续小块摊薄 MP4 随机定位开销；推理仍保持独立的低内存 batch。
            decode_chunk = max(8, batch_size)
            for index in range(start_index, stop_index, decode_chunk):
                decode_started = time.perf_counter()
                decoded = decoder.get_frames_in_range(index, min(index + decode_chunk, stop_index)).data.to(self.device)
                if stats is not None and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                decode_seconds = time.perf_counter() - decode_started
                compute_seconds += decode_seconds
                if stats is not None:
                    stats["decode"] = stats.get("decode", 0.0) + decode_seconds
                for offset in range(0, len(decoded), batch_size):
                    batch_started = time.perf_counter()
                    batch = decoded[offset:offset + batch_size]
                    output = self._process_batch(batch, identity_embedding, full_resolution_detection, stats)
                    batch_seconds = time.perf_counter() - batch_started
                    compute_seconds += batch_seconds
                    wait_ms = max(1, round(1000 * (1 / source_fps - batch_seconds / len(output))))
                    for frame in output:
                        cv2.imshow("swapped", frame)
                        done += 1
                        if (cv2.waitKey(wait_ms) & 0xff) in (ord("q"), 27):
                            return
        finally:
            cv2.destroyAllWindows()
            elapsed = time.perf_counter() - started
            if done:
                print(f"Processed {done} frames, compute FPS={done / compute_seconds:.2f}, wall FPS={done / elapsed:.2f}")
            if stats is not None and done:
                print("Stage ms/frame: " + ", ".join(f"{key}={seconds * 1000 / done:.2f}" for key, seconds in stats.items()))


def main() -> None:
    _configure_swapper_runtime()
    parser = argparse.ArgumentParser(description="使用当前 checkpoint 或配套 ONNX 预览视频换脸")
    parser.add_argument("--model", required=True, help="当前训练 checkpoint，或当前 export.py 导出的 ONNX")
    parser.add_argument("--video", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1, help="每批推理帧数（默认 1；解码独立按小块读取）")
    parser.add_argument("--start", type=float, default=0.0, help="从指定秒数开始预览（按平均 FPS 近似定位）")
    parser.add_argument("--frames", type=int, help="最多预览多少帧，默认直到视频结束")
    parser.add_argument("--profile", action="store_true", help="输出各阶段耗时；GPU 同步可能降低吞吐")
    parser.add_argument("--full-resolution-detection", action="store_true", help="按原视频分辨率检测，保留原有检测精度；默认大帧使用检测器原生缩放")
    parser.add_argument("--lookup-coarse", action="store_true", help="仅使用 Coarse 输出进行换脸；默认使用完整 Generator 输出")
    args = parser.parse_args()
    SwapFace(args.model, device=args.device, lookup_coarse=args.lookup_coarse).swap_video(
        args.video,
        args.identity,
        batch_size=args.batch_size,
        full_resolution_detection=args.full_resolution_detection,
        start=args.start,
        frames=args.frames,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
