from pathlib import Path
from typing import Any

import torch
from torch import nn, Tensor
from misc.facealign import face_align_batch
from misc.models.retinaface import get_pts, batch_resize_and_pad_varsize, RetinaFace
from misc.models.idencoder import IDEncoder, get_align_landmarks
from .networks.generator import Generator
from torchcodec.decoders import VideoDecoder
from torchvision.io import decode_image
from tqdm import tqdm
import cv2


class Swap:
    def __init__(self, ckpt: str, device: str = "cuda") -> None:
        super().__init__()

        if Path(ckpt).exists() == False:
            raise FileNotFoundError(f"ckpt file: {ckpt} Not found")

        self.device = torch.device(device)

        print(f"Loading ckpt from {ckpt}")
        ckpt: dict[str, Any] = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

        self.iter = ckpt["iter"]

        print(f"ckpt Info:\n" f"  {'iter':25}: {self.iter}")
        print("net_g:")
        for k, v in ckpt["net_g"]["network_cfg"].items():
            print(f"  {k:25}: {v}")
        print("net_d:")
        for k, v in ckpt["net_d"]["network_cfg"].items():
            print(f"  {k:25}: {v}")

        self.size = ckpt["net_g"]["network_cfg"]["input_res"]

        self.net_g = Generator(**ckpt["net_g"]["network_cfg"]).to(device=self.device)
        self.net_g.load_state_dict(ckpt["net_g"]["state_dict"])

        self.facedetch = RetinaFace(from_normalized=True).to(device=self.device)
        self.idencoder = IDEncoder().to(device=self.device)

        self.dst_pts = torch.tensor(get_align_landmarks(self.size), device=self.device)

    def get_face_id(self, fp: str) -> Tensor:

        image = decode_image(fp).to(device=self.device, dtype=torch.float)
        image.div_(127.5).sub_(1.0).unsqueeze_(0)  # [-1 ~ 1]

        detected = self.facedetch.detector(image)
        src_pts = get_pts(detected)
        dst_pts = get_align_landmarks(self.size)
        dst_pts = torch.tensor(dst_pts, device=self.device)
        align_face, _ = face_align_batch(image, src_pts, dst_pts, (self.size, self.size))

        align_face = torch.cat(align_face, dim=0)

        id_emb = self.idencoder(align_face)

        return id_emb[:1]

    def swap_video(self, fp: str, id: str):

        batch_size = 8

        id_emb = self.get_face_id(id)

        vr = VideoDecoder(fp, device="cuda")
        num_frames = vr.metadata.num_frames_from_content
        pbar = tqdm(range(vr.metadata.num_frames_from_content), unit="frame")
        slices = [(i, min(i + batch_size, num_frames)) for i in range(0, num_frames, batch_size)]
        dst_pts = get_align_landmarks(self.size)
        dst_pts = torch.tensor(dst_pts, device=self.device)

        for slic in slices:
            pbar.n = slic[1]
            pbar.refresh()

            chunk = vr.get_frames_in_range(*slic).data.to(device=self.device, dtype=torch.float).div_(127.5).sub_(1.0)

            detected = self.facedetch.detector(chunk)
            src_pts = get_pts(detected)
            align_face, _ = face_align_batch(chunk, src_pts, dst_pts, (self.size, self.size))

            align_face = torch.cat(align_face, dim=0)

            swap_face: Tensor = self.net_g(align_face, id_emb.expand(align_face.shape[0], -1))

            swap_face = swap_face.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)[:, [2, 1, 0], :, :]  # RGB -> BGR
            swap_face = swap_face.permute(0, 2, 3, 1)  # NCHW -> NHWC
            frames = swap_face.to(device="cpu", dtype=torch.uint8).numpy()

            for frame in frames:
                cv2.imshow("swaped", frame)
                cv2.waitKey(1)


if __name__ == "__main__":

    video = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    id = "/home/liaohaixun/swap/IDAssets/wuyanzu.png"
    swapper = Swap("train_log/256_blendface/ckpt/1180000.pth")

    swapper.swap_video(video, id)
