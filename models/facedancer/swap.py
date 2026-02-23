from pathlib import Path
from typing import Any

import torch
from torch import nn, Tensor
from misc.facealign import face_align_batch
from misc.models.retinaface import get_pts, batch_resize_and_pad_varsize, RetinaFace
from misc.models.idencoder import IDEncoder
from .networks.generator import Generator


class Swap(nn.Module):
    def __init__(self, ckpt: str) -> None:
        super().__init__()

        if Path(ckpt).exists() == False:
            raise FileNotFoundError(f"ckpt file: {ckpt} Not found")

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

        self.net_g = Generator(**ckpt["net_g"]["network_cfg"])

        self.facedetch = RetinaFace(from_normalized=True)

        self.idencoder = IDEncoder()

    def swap_video(self,fp:str):
        pass



if __name__ == "__main__":

    ckpt = ""
