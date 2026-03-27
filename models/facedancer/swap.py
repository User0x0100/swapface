from pathlib import Path
from typing import Any
import multiprocessing as mp

import torch
from torch import Tensor
import torchvision.transforms.functional as F
from misc.facealign import face_align_batch, AlignFaceExtractor
from misc.models.retinaface import get_pts, RetinaFace
from misc.models.idencoder import IDEncoder, get_align_landmarks, PROVIDER
from misc.models.face_parsing import FaceParsing
from .networks import Generator
from torchvision.io import decode_image
from tqdm import tqdm

import cv2


class Swap:
    def __init__(self, ckpt: str, idencoder_provider: PROVIDER, device: str = "cuda") -> None:
        super().__init__()

        if not Path(ckpt).exists():
            raise FileNotFoundError(f"ckpt file: {ckpt} Not found")

        self.device = torch.device(device)

        print(f"Loading ckpt from {ckpt}")
        ckpt: dict[str, Any] = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

        self.iter = ckpt["iter"]

        print(f"ckpt Info:\n  {'iter':25}: {self.iter}")
        print("net_g:")
        for k, v in ckpt["net_g"]["network_cfg"].items():
            print(f"  {k:25}: {v}")
        print("net_d:")
        for k, v in ckpt["net_d"]["network_cfg"].items():
            print(f"  {k:25}: {v}")

        self.img_resolution = ckpt["net_g"]["network_cfg"]["img_resolution"]

        net_g = Generator(**ckpt["net_g"]["network_cfg"])
        net_g.load_state_dict(ckpt["net_g"]["state_dict"])
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

        id_emb = self.idencoder(align_face)

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

    @torch.inference_mode()
    def swap_video(self, vfp: str, id_fp: str):

        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=15)
        p = mp.Process(target=self._display_worker, args=(q,), daemon=True)
        p.start()

        batch_size = 8

        id_emb = self.extract_id_feats_from_image(id_fp)

        for faces, norm_theta, offset in AlignFaceExtractor(vfp, batch_size=batch_size, align_size=self.img_resolution, device=self.device):
            N = faces.size(0)
            if N == 0:
                continue
            faces.div_(127.5).sub_(1.0)

            with torch.autocast(device_type="cuda"):
                swap_face: Tensor = self.net_g(faces, id_emb)
                mask = self.face_mask(faces)
                mask = F.gaussian_blur(mask, 7, 13)
                swap_face = faces * (1.0 - mask) + swap_face * mask
                mask = mask * 2.0 - 1.0

            swap_face = torch.cat((faces, swap_face, mask.expand(N, 3, -1, -1)), dim=3)
            swap_face = swap_face.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)[:, [2, 1, 0], :, :]  # RGB -> BGR
            swap_face = swap_face.permute(0, 2, 3, 1)  # NCHW -> NHWC
            frames = swap_face.to(device="cpu", dtype=torch.uint8).numpy()

            q.put(frames)

        q.put(None)
        p.join()


if __name__ == "__main__":
    video = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    id = "/home/liaohaixun/swap/IDAssets/wuyanzu.png"
    # id = "/home/liaohaixun/swap/IDAssets/周杰伦.png"
    # id = "/home/liaohaixun/swap/IDAssets/陈冠希.png"

    # video = "/opt/share/deepfake/dataset_1/oneman/1.mp4"
    # id = "/home/liaohaixun/swap/faceset/ljx/arcface_pts/6000_0.png"
    # id = "/home/liaohaixun/swap/IDAssets/安妮·海瑟薇.png"
    # id = "/home/liaohaixun/swap/IDAssets/2025-06-15 18_19_50小树🌿人间体验卡限时掉落✨ _3.jpg"

    swapper = Swap("train_log/256_BLENDFACE_AlphaDise_New_ID_8_Injection_2/ckpt/590000.pth", PROVIDER.BLENDFACE)

    swapper.swap_video(video, id)
