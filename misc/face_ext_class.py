from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

from .facealign import AlignFaceExtractor
from .models.idencoder import IDEncoder, PROVIDER


class Identity:
    feat: Tensor
    folder: Path
    count: int

    def __init__(self, feat: Tensor, folder: Path, count: int = 0):
        self.feat = feat
        self.folder = folder
        self.count = count


def save_image(img: Tensor, fp: Path):
    img = img[[2, 1, 0], :, :]  # RGB -> BGR
    img = img.permute(1, 2, 0)  # CHW -> HWC
    img_cpu = img.to(device="cpu", dtype=torch.uint8).numpy()
    cv2.imwrite(fp, img_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def exp(
    vfp: str | Path,
    output_dir: str | None = None,
    batch_size: int = 16,
    align_size: int = 512,
    id_feat_similarity_thres: float = 0.4,
    conf_thresh: float = 0.98,
    iou_thresh: float = 0.5,
    mini_id_nb: int = 50,
    min_box_size: tuple[int, int] = (256, 256),
    device: str = "cuda",
):
    vfn = Path(vfp).stem
    output_dir = Path(vfp).parent / f"{vfn}_class_result" if output_dir is None else Path(output_dir)

    exis_id_feats: list[Identity] = []
    exis_feats_cache: Tensor | None = None
    device = torch.device(device)

    total_frames = int(cv2.VideoCapture(vfp).get(cv2.CAP_PROP_FRAME_COUNT))

    ID_Encoder = IDEncoder(provider=PROVIDER.MS1MV3_ARCFACE_R50_FP16).to(device=device).eval()

    # I/O线程池
    executor = ThreadPoolExecutor(max_workers=4)
    pbar = tqdm(total=total_frames, desc="Processing")
    for faces, _, _ in AlignFaceExtractor(vfp, batch_size, align_size, device, conf_thresh=conf_thresh, iou_thresh=iou_thresh, min_box_size=min_box_size):
        with torch.inference_mode():
            id_feats = ID_Encoder((faces / 127.5) - 1.0)  # (B, C)

        B = id_feats.size(0)

        for i in range(B):
            id_feat = id_feats[i : i + 1].clone()  # (1, C)
            matched = False

            if len(exis_id_feats) > 0:
                # (N, C)
                if exis_feats_cache is None:
                    exis_feats_cache = torch.cat([id.feat for id in exis_id_feats], dim=0)

                # (N,)
                similarity = F.cosine_similarity(exis_feats_cache, id_feat, dim=1)

                max_sim, max_idx = similarity.max(dim=0)

                if max_sim > id_feat_similarity_thres:
                    max_idx = max_idx.item()
                    exis_id = exis_id_feats[max_idx]
                    matched = True

                    # 更新计数
                    exis_id.count += 1

                    # 特征滑动平均（提高稳定性）
                    exis_id.feat = 0.9 * exis_id.feat + 0.1 * id_feat
                    exis_feats_cache[max_idx] = exis_id.feat

                    save_path = exis_id.folder / f"{exis_id.count:05d}.png"
                    executor.submit(save_image, faces[i].cpu(), save_path)

            if not matched:
                # 创建新身份

                new_id_folder = output_dir / f"id{len(exis_id_feats):05d}"
                if not new_id_folder.exists():
                    new_id_folder.mkdir(parents=True)

                new_id = Identity(feat=id_feat, folder=new_id_folder, count=0)
                exis_id_feats.append(new_id)
                exis_feats_cache = None
                save_path = new_id_folder / "00000.png"
                save_image(faces[i], save_path)
                executor.submit(save_image, faces[i].cpu(), save_path)

        pbar.update(batch_size)

        pbar.set_postfix({"ids": len(exis_id_feats), "faces": B})

    executor.shutdown(wait=True)

    # 清理小样本identity
    for i in exis_id_feats:
        if i.count < mini_id_nb:
            shutil.rmtree(i.folder)


if __name__ == "__main__":
    exp("dataset/r01qao0_Gng.webm")
