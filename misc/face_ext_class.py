import argparse
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

from .facealign import AlignFaceExtractor, zoom_in
from .models.idencoder import IDEncoder, PROVIDER


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("vfp", type=str)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--align_size", type=int, default=512)
    parser.add_argument("--use_mobile_net_backbone", action="store_true")

    parser.add_argument("--id_feat_similarity_thres", type=float, default=0.3)
    parser.add_argument("--conf_thresh", type=float, default=0.99)
    parser.add_argument("--iou_thresh", type=float, default=0.3)

    parser.add_argument("--mini_id_nb", type=int, default=50)
    parser.add_argument("--min_box_size", type=int, nargs=2, default=(256, 256))

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


class Identity:
    feat: Tensor
    folder: Path
    count: int

    def __init__(self, feat: Tensor, folder: Path, count: int = 0):
        self.feat = feat
        self.folder = folder
        self.count = count

    def update_feat(self, new_feat: Tensor):
        self.count += 1
        self.feat = self.feat * (self.count - 1) / self.count + new_feat / self.count
        self.feat = F.normalize(self.feat, dim=-1)


def save_image(img: Tensor, fp: Path):
    img = img[[2, 1, 0], :, :]  # RGB -> BGR
    img = img.permute(1, 2, 0)  # CHW -> HWC
    img_cpu = img.to(device="cpu", dtype=torch.uint8).numpy()
    cv2.imwrite(fp, img_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


@torch.no_grad()
def exp(
    vfp: str | Path,
    output_dir: str | None = None,
    batch_size: int = 16,
    align_size: int = 512,
    use_mobile_net_backbone: bool = False,
    id_feat_similarity_thres: float = 0.3,
    conf_thresh: float = 0.99,
    iou_thresh: float = 0.3,
    mini_id_nb: int = 50,
    min_box_size: tuple[int, int] = (256, 256),
    device: str = "cuda",
):
    vfp = Path(vfp)
    if not vfp.exists():
        raise FileNotFoundError(vfp)
    vfn = vfp.stem
    output_dir = Path(vfp).parent if output_dir is None else Path(output_dir)

    output_dir = output_dir / f"{vfn}_class_result"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    exis_id_feats: list[Identity] = []
    exis_feats_cache: Tensor | None = None
    device = torch.device(device)

    ID_Encoder = IDEncoder(provider=PROVIDER.MS1MV2_TRANSFACE_L).to(device=device).eval()
    extractor = AlignFaceExtractor(
        vfp,
        batch_size,
        align_size,
        use_mobile_net_backbone=use_mobile_net_backbone,
        device=device,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        min_box_size=min_box_size,
    )
    total_frames = len(extractor)

    # I/O线程池
    executor = ThreadPoolExecutor(max_workers=4)
    pbar = tqdm(total=total_frames, desc="Processing")
    for faces, _, _ in extractor:
        with torch.inference_mode():
            id_feats = ID_Encoder(zoom_in((faces / 127.5) - 1.0, 0.3))  # (B, C)

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
                max_idx = max_idx.item()

                if max_sim > id_feat_similarity_thres:
                    exis_id = exis_id_feats[max_idx]
                    matched = True

                    exis_id.update_feat(id_feat)
                    exis_feats_cache[max_idx] = exis_id.feat

                    save_path = exis_id.folder / f"{exis_id.count:05d}.png"
                    executor.submit(save_image, faces[i].cpu(), save_path)

            if not matched:
                # 创建新身份

                new_id_folder = output_dir / f"id{len(exis_id_feats):05d}"
                if not new_id_folder.exists():
                    new_id_folder.mkdir(parents=True)

                if exis_feats_cache is None:
                    exis_feats_cache = id_feat.clone()
                else:
                    exis_feats_cache = torch.cat([exis_feats_cache, id_feat], dim=0)

                new_id = Identity(feat=id_feat, folder=new_id_folder, count=0)
                exis_id_feats.append(new_id)

                save_path = new_id_folder / "00000.png"
                executor.submit(save_image, faces[i].cpu(), save_path)

        pbar.update(batch_size)

        pbar.set_postfix({"Total ids": len(exis_id_feats), "faces": f"{B:02d}"})

    executor.shutdown(wait=True)

    # 清理小样本identity
    for i in exis_id_feats:
        if i.count < mini_id_nb:
            shutil.rmtree(i.folder)


if __name__ == "__main__":
    args = parse_args()

    exp(
        vfp=args.vfp,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        align_size=args.align_size,
        use_mobile_net_backbone=args.use_mobile_net_backbone,
        id_feat_similarity_thres=args.id_feat_similarity_thres,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        mini_id_nb=args.mini_id_nb,
        min_box_size=tuple(args.min_box_size),
        device=args.device,
    )
