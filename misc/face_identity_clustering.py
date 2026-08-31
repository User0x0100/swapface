import argparse
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .face_alignment import VideoFaceExtractor, center_crop_and_resize
from .models.id_encoder import IDEncoder, IDEncoderProvider


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("video_path", type=str)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--use-mobilenet", action="store_true")

    parser.add_argument("--identity-similarity-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", type=float, default=0.99)
    parser.add_argument("--iou-threshold", type=float, default=0.3)

    parser.add_argument("--min-identity-samples", type=int, default=50)
    parser.add_argument("--min-face-size", type=int, nargs=2, default=(256, 256))

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


@dataclass(slots=True)
class IdentityCluster:
    centroid: Tensor
    folder: Path
    count: int = 1

    def update_centroid(self, embedding: Tensor) -> None:
        self.count += 1
        self.centroid = self.centroid * (self.count - 1) / self.count + embedding / self.count
        self.centroid = F.normalize(self.centroid, dim=-1)


def save_image(img: Tensor, path: Path) -> None:
    image = img[[2, 1, 0]].permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Failed to write image: {path}")


@torch.no_grad()
def cluster_video_faces_by_identity(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    batch_size: int = 16,
    output_size: int = 512,
    use_mobilenet: bool = False,
    identity_similarity_threshold: float = 0.3,
    confidence_threshold: float = 0.99,
    iou_threshold: float = 0.3,
    min_identity_samples: int = 50,
    min_face_size: tuple[int, int] = (256, 256),
    device: str | torch.device = "cuda",
) -> None:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    video_name = video_path.stem
    output_path = (video_path.parent if output_dir is None else Path(output_dir)) / f"{video_name}_class_result"
    output_path.mkdir(parents=True, exist_ok=True)

    identities: list[IdentityCluster] = []
    identity_centroids: Tensor | None = None
    device_obj = torch.device(device)

    identity_encoder = IDEncoder(provider=IDEncoderProvider.MS1MV2_TRANSFACE_L).to(device=device_obj).eval()

    extractor = VideoFaceExtractor(
        video_path,
        batch_size,
        output_size,
        device=device_obj,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        min_face_size=min_face_size,
        use_mobilenet=use_mobilenet,
    )

    futures: list[Future[None]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        pbar = tqdm(extractor, desc="Processing")
        for _, faces, _, _, frame_count in pbar:
            identity_embeddings = identity_encoder(center_crop_and_resize((faces / 127.5) - 1.0, 0.102))
            face_count = identity_embeddings.size(0)

            for i in range(face_count):
                identity_embedding = identity_embeddings[i : i + 1].clone()
                matched = False

                if identities:
                    if identity_centroids is None:
                        identity_centroids = torch.cat([identity.centroid for identity in identities], dim=0)

                    similarities = F.cosine_similarity(identity_centroids, identity_embedding, dim=1)
                    max_similarity, max_index_tensor = similarities.max(dim=0)
                    max_index = int(max_index_tensor.item())

                    if float(max_similarity.item()) > identity_similarity_threshold:
                        identity = identities[max_index]
                        matched = True
                        identity.update_centroid(identity_embedding)
                        identity_centroids[max_index] = identity.centroid
                        save_path = identity.folder / f"{identity.count - 1:05d}.png"
                        futures.append(executor.submit(save_image, faces[i].cpu(), save_path))

                if not matched:
                    cluster_folder = output_path / f"id{len(identities):05d}"
                    cluster_folder.mkdir(parents=True, exist_ok=True)

                    if identity_centroids is None:
                        identity_centroids = identity_embedding.clone()
                    else:
                        identity_centroids = torch.cat([identity_centroids, identity_embedding], dim=0)

                    identities.append(IdentityCluster(centroid=identity_embedding, folder=cluster_folder))
                    futures.append(executor.submit(save_image, faces[i].cpu(), cluster_folder / "00000.png"))

            pbar.set_postfix({"Total ids": len(identities), "faces": f"{face_count:02d}", "frames": frame_count})

    for future in futures:
        future.result()

    # 清理小样本identity
    for identity in identities:
        if identity.count < min_identity_samples:
            shutil.rmtree(identity.folder)


if __name__ == "__main__":
    args = parse_args()

    cluster_video_faces_by_identity(
        video_path=args.video_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        output_size=args.output_size,
        use_mobilenet=args.use_mobilenet,
        identity_similarity_threshold=args.identity_similarity_threshold,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        min_identity_samples=args.min_identity_samples,
        min_face_size=tuple(args.min_face_size),
        device=args.device,
    )
