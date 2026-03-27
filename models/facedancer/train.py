import copy
import random
import itertools
from enum import Enum
from typing import Any
from pathlib import Path

import torch
from torch.amp import autocast
import torch.nn.functional as F
from torch import Tensor, optim, nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2
from tqdm import tqdm
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from losses import (
    IDLoss,
    l1_loss_fn,
    VGGPerceptualLoss,
    DLoss,
    GANLoss,
    StyleLossLabChroma,
    r1_reg_loss,
    WFMLoss,
    IFSRLoss,
)

from .dataloader import datasetloader
from .networks import (
    Generator,
    Discriminator,
    Stylegan2DiscriminatorLite,
    AlphaFaceDiscriminator,
)


EPS = 1e-8

assert torch.cuda.is_available(), "仅支持使用NVIDIA显卡训练"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class DISCRIMINATOR_TYPT(Enum):
    ORIGIN = Discriminator
    ALPHAFACE = AlphaFaceDiscriminator
    STYLEGAN2 = Stylegan2DiscriminatorLite


def print_dict(d: dict, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_dict(v, indent + 2)
        else:
            print(" " * indent + f"{k:25}: {v}")


class Trainer:
    def __init__(
        self,
        src: list[tuple[str, float]],
        dst: list[tuple[str, float]],
        identity_root: list[str] | None = None,
        batch_size: int = 10,
        lr: float = 1e-4,
        lr_scheduler_t_max: int = 0,
        discriminator_typt: DISCRIMINATOR_TYPT = DISCRIMINATOR_TYPT.ORIGIN,
        d_train_setp: int = 1,
        r1_reg_step: int = 1,
        bf16: bool = True,
        device: str = "cuda:0",
        compile_module: bool = True,
        ckpt: str | None = None,
        log_path: str = "train_log/exper_0",
        log_interval: int = 10,
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
        same_image_prob: float = 0.2,
        # 模型配置
        net_g_cfg: dict[str, int] | None = None,
        net_d_cfg: dict[str, int] | None = None,
        id_encode_provider: IDLoss.Provider = IDLoss.Provider.BLENDFACE,
        id_loss_weight: float = 10.0,
        rec_loss: float = 5.0,
        perceptual_loss_weight: dict[str, float] = {
            # vgg16
            "relu1_2": 0.25,
            "relu2_2": 0.25,
            "relu3_3": 0.25,
            "relu4_2": 0.25,
            # "pool1": 0.2,
            # "pool2": 0.2,
            # "pool3": 0.2,
            # "pool4": 0.2,
            # "pool5": 0.2,
        },
        enable_wfm_loss: bool = False,
        wfm_loss_weight: dict[int, float] = {
            1: 1.0,
            2: 1.0,
        },
        enable_ifsr_loss: bool = False,
        ifsr_scale: float = 1.2,
        ifsr_weight: dict[str, tuple[float, float]] = {
            "layer3.5": (0.121357, 1.0),
            "layer3.4": (0.128827, 1.0),
            "layer3.3": (0.117972, 1.0),
            "layer3.2": (0.109391, 1.0),
            "layer3.1": (0.097296, 1.0),
            "layer3.0": (0.089046, 1.0),
            "layer2.3": (0.044928, 1.0),
            "layer2.2": (0.048719, 1.0),
            "layer2.1": (0.047487, 1.0),
            "layer2.0": (0.047970, 1.0),
            "layer1.2": (0.035144, 1.0),
        },
        enable_color_loss: bool = False,
        color_loss_weight: float = 0.5,
    ):

        args = locals().copy()
        for k in ["src", "dst", "self"]:
            args.pop(k)

        print("Train Config:")
        print_dict(args, 2)

        self.rng = random.Random()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.d_train_setp = d_train_setp
        self.r1_reg_step = r1_reg_step
        self.enable_wfm_loss = enable_wfm_loss
        self.enable_ifsr_loss = enable_ifsr_loss
        self.enable_color_loss = enable_color_loss

        self.bf16 = bool(bf16 and torch.cuda.is_bf16_supported())
        if bf16 and not self.bf16:
            print("Warning: 当前显卡不支持 BF16")

        self.sample_save_every = sample_save_every
        self.weight_save_every = weight_save_every
        self.log_interval = log_interval
        self.enable_lr_scheduler = lr_scheduler_t_max > 0

        # ========================= Init Model =========================

        if ckpt is not None:
            if not Path(ckpt).exists():
                raise FileNotFoundError(f"ckpt file: {ckpt} Not found")

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
            net_d = discriminator_typt.value(**ckpt["net_d"]["network_cfg"])
            net_g.load_state_dict(ckpt["net_g"]["state_dict"])
            net_d.load_state_dict(ckpt["net_d"]["state_dict"])

        else:
            self.iter, self.img_resolution = 0, net_g_cfg["img_resolution"]
            net_g = Generator(**net_g_cfg)
            net_d = discriminator_typt.value(**net_d_cfg)

        self.net_g = net_g.to(self.device).train()
        self.net_d = net_d.to(self.device).train()

        self.net_g_ema = copy.deepcopy(self.net_g)
        self.net_g_ema.eval().requires_grad_(False)

        # ========================= Optim =========================
        self.optim_g = optim.AdamW(self.net_g.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)
        self.optim_d = optim.AdamW(self.net_d.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)

        if self.enable_lr_scheduler:
            self.lr_scheduler_g = CosineAnnealingLR(self.optim_g, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)
            self.lr_scheduler_d = CosineAnnealingLR(self.optim_d, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)

        # ========================= LOSS =========================

        self.d_loss = DLoss(weight=1.0, reduction="mean").to(self.device)
        self.gan_loss = GANLoss(weight=1.0, reduction="mean").to(self.device)

        self.id_loss = IDLoss(weight=id_loss_weight, provider=id_encode_provider).to(self.device)
        self.rec_loss = l1_loss_fn(weight=rec_loss, reduction="none")

        self.perceptual_loss = VGGPerceptualLoss(layer_weights=perceptual_loss_weight, reduction="mean").to(self.device)

        if self.enable_ifsr_loss:
            self.ifsr_loss = IFSRLoss(ifsr_scale=ifsr_scale, ifsr_weight=ifsr_weight).to(self.device)

        if self.enable_wfm_loss:
            self.wfm_loss = WFMLoss(layer_weights=wfm_loss_weight, criterion="l1").to(self.device)
        if self.enable_color_loss:
            self.color_loss = StyleLossLabChroma(weight=color_loss_weight, range_norm=True).to(self.device)

        # ========================= LOG =========================
        base_log_path = Path(log_path)
        self.ckpt_dir = base_log_path.joinpath("ckpt")
        self.sample_dir = base_log_path.joinpath("sample")
        self.tensorboard_dir = base_log_path.joinpath("tensorboard")

        for p in [self.ckpt_dir, self.sample_dir, self.tensorboard_dir]:
            p.mkdir(exist_ok=True, parents=True)

        self.log_writer = SummaryWriter(self.tensorboard_dir)

        # ========================= Sample =========================
        pipe = datasetloader(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=5,
            py_num_workers=3,
            py_start_method="spawn",
            device_id=self.device.index,
            resize=self.img_resolution,
            src=src,
            dst=dst,
            identity_root=identity_root,
            same_image_prob=same_image_prob,
        )

        self.dataset = DALIGenericIterator(pipelines=pipe, output_map=["src", "dst", "is_same"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        # ========================= compile_module =========================
        self.train_module: dict[str, nn.Module] = {}

        for net_name in ["net_g", "net_d"]:
            net: nn.Module = getattr(self, net_name)
            self.train_module[net_name] = torch.compile(net, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True}) if compile_module else net

    @torch.no_grad()
    def log(self, k: str, v: Tensor, right_now: bool = False) -> None:
        if self.iter % self.log_interval == 0 or right_now:
            self.log_writer.add_scalar(f"Loss/{k}", v.detach().mean().item(), self.iter)

    @torch.no_grad()
    def fetch_sample(self) -> tuple[Tensor, Tensor, Tensor]:
        data: dict[str, Tensor] = self.dataset.next()[0]
        src, dst, is_same = (data[k] for k in ("src", "dst", "is_same"))
        is_same = is_same.squeeze_(-1)

        return src, dst, is_same

    @torch.no_grad()
    def update_ema(self, decay=0.999):

        decay = min(decay, 1 - 1 / (self.iter + 1))
        alpha = 1.0 - decay

        for p_ema, p_train in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
            p_ema.lerp_(p_train, alpha)

    @torch.no_grad()
    def save_ckpt(self):

        net_g = {
            "network_cfg": self.net_g_ema.network_cfg,
            "state_dict": self.net_g_ema.state_dict(),
        }

        net_d = {
            "network_cfg": self.net_d.network_cfg,
            "state_dict": self.net_d.state_dict(),
        }

        state_dict = {
            "iter": self.iter,
            "net_g": net_g,
            "net_d": net_d,
        }
        try:
            ckpt_file = self.ckpt_dir / f"{self.iter}.pth"
            torch.save(state_dict, ckpt_file)
        except Exception as e:
            print(f"Failed to save ckpt: {e}")

    def train(self):

        net_d, net_g = (self.train_module[module_name] for module_name in ("net_d", "net_g"))

        for self.iter in tqdm(itertools.count(start=self.iter), initial=self.iter, mininterval=1.0, bar_format="{n_fmt:7} | speed {rate_fmt:3} | train time {elapsed}"):
            src, dst, is_same = self.fetch_sample()

            torch.compiler.cudagraph_mark_step_begin()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                with torch.inference_mode():
                    src_id_feats = self.id_loss.get_id_feats(src)
                    if self.enable_ifsr_loss:
                        dst_ifsr_feats = self.ifsr_loss.get_ifsr_feats(dst)

                fake: Tensor = net_g(dst, src_id_feats)

                # ========================= train d =========================
                if self.iter % self.d_train_setp == 0:
                    self.optim_d.zero_grad()
                    d_loss: Tensor = torch.tensor(0.0, device=self.device)

                    d_step = self.iter // self.d_train_setp
                    is_r1_reg_step = d_step % self.r1_reg_step == 0

                    real_img = dst.detach().requires_grad_(is_r1_reg_step)

                    fake_global_score = net_d(fake.detach())
                    real_global_score = net_d(real_img)

                    global_d_loss = self.d_loss(fake_global_score, real_global_score)
                    self.log("global_d_loss", global_d_loss, True)
                    d_loss += global_d_loss

                    if is_r1_reg_step:
                        r1_loss = r1_reg_loss(real_global_score, real_img)
                        self.log("r1_loss", r1_loss)
                        d_loss += r1_loss

                    d_loss.backward()
                    self.optim_d.step()

                # ========================= train g =========================
                self.optim_g.zero_grad()
                loss: Tensor = torch.tensor(0.0, device=self.device)

                # gan_loss
                if self.enable_wfm_loss:
                    fake_global_score, fake_feats = net_d(fake, True)
                else:
                    fake_global_score = net_d(fake)
                global_gan_loss = self.gan_loss(fake_global_score)
                self.log("global_gan_loss", global_gan_loss)
                loss += global_gan_loss

                # wfm_loss
                if self.enable_wfm_loss:
                    with torch.inference_mode():
                        _, real_feats = net_d(dst, True)
                    wfm_loss = self.wfm_loss(fake_feats, real_feats)
                    self.log("wfm_loss", wfm_loss)
                    loss += wfm_loss

                # loss_id
                fake_id_feats = self.id_loss.get_id_feats(fake)
                id_loss = self.id_loss(fake_id_feats, src_id_feats)
                self.log("id_loss", id_loss)
                loss += id_loss

                # ifsr_loss
                if self.enable_ifsr_loss:
                    fake_ifsr_feats = self.ifsr_loss.get_ifsr_feats(fake)
                    ifsr_loss = self.ifsr_loss(fake_ifsr_feats, dst_ifsr_feats)
                    self.log("ifsr_loss", ifsr_loss)
                    loss += ifsr_loss

                # perceptual_loss
                perceptual_loss = self.perceptual_loss(fake, dst)
                # perceptual_loss = (perceptual_loss * is_same).sum() / is_same.sum().clamp_min(1.0)
                self.log("perceptual_loss", perceptual_loss)
                loss += perceptual_loss

                # rec_loss
                rec_loss = self.rec_loss(fake, dst)  # BCHW
                rec_loss = rec_loss.mean(dim=[1, 2, 3])
                rec_loss = (rec_loss * is_same).sum() / is_same.sum().clamp_min(1.0)
                self.log("rec_loss", rec_loss)
                loss += rec_loss

                # color_loss
                if self.enable_color_loss:
                    color_loss = self.color_loss(fake, dst)
                    self.log("color_loss", color_loss)
                    loss += color_loss

            loss.backward()
            self.optim_g.step()

            if self.enable_lr_scheduler:
                self.lr_scheduler_g.step()
                self.lr_scheduler_d.step()

            self.update_ema()

            if self.iter % self.weight_save_every == 0:
                self.save_ckpt()

            if self.iter % self.sample_save_every == 0:
                with torch.inference_mode():
                    src_id_feats = self.id_loss.get_id_feats(src)
                    fake: Tensor = self.net_g_ema(dst, src_id_feats)
                    attn_map: list[Tensor] = self.net_g_ema.get_attention_maps()
                    grid = [src, dst, fake]

                    if len(attn_map) > 0:
                        for m in attn_map:
                            m = m.mean(dim=1, keepdim=True)
                            m = F.interpolate(m, size=self.img_resolution, mode="bilinear", align_corners=False)
                            m = m.expand(-1, 3, -1, -1)

                            amin = m.amin(dim=(2, 3), keepdim=True)
                            amax = m.amax(dim=(2, 3), keepdim=True)
                            m = 2.0 * (m - amin) / (amax - amin + 1e-6) - 1.0

                            grid.append(m)

                    grid = torch.cat(grid, dim=0)
                    grid.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
                    grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB -> BGR
                    grid = grid.permute(1, 2, 0)  # CHW -> HWC
                grid_cpu = grid.to(device="cpu", dtype=torch.uint8).numpy()
                cv2.imwrite(
                    self.sample_dir / f"{self.iter}.png",
                    grid_cpu,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )


if __name__ == "__main__":
    src = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        # ("/opt/share/deepfake/dataset_1/vggface2_hq512", 0.0),
    ]

    dst = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        # ("/opt/share/deepfake/dataset_1/vggface2_hq512", 0.0),
        # ("/opt/share/deepfake/dataset_1/RealOcc/image/realign_arcface_dst", 1.0),
        # ("/opt/share/deepfake/dataset_1/youtube/What_s_considered_tall_in_South_Korea_Street_Interview_align_results", 0.0),
        # ("/opt/share/deepfake/dataset_1/youtube/4k_Face_Close_Up_HDR_Video_Vivid_Colors_Ambient_Sound_-_Relaxing_align_results", 0.0),
        # ("/opt/share/deepfake/dataset_1/oneman/1_align_results/", 0.0),
    ]
    identity_root = ["dataset"]

    def_config = {"src": src, "dst": dst}

    # def_config.update(
    #     {
    #         "ckpt": "train_log/256_BLENDFACE_ADAIN_WFM_Same0.0/ckpt/1296007.pth",
    #         "net_g_cfg": {
    #             "img_resolution": 256,
    #             "img_channels": 3,
    #             "num_encoder": 5,
    #             "base_ch": 64,
    #             "max_ch": 512,
    #             "id_dim": 512,
    #             "w_dim": 256,
    #             "mapping_num": 4,
    #             "skip_index": 2,
    #         },
    #         "net_d_cfg": {
    #             "img_resolution": 256,
    #             "img_channels": 3,
    #             "num_encoder": 5,
    #             "base_ch": 64,
    #             "max_ch": 512,
    #         },
    #         "log_path": "train_log/256_BLENDFACE_ADAIN_WFM_Same0.0",
    #         "enable_wfm_loss": True,
    #         "wfm_loss_weight": {
    #             0: 2.0,
    #             1: 1.0,
    #         },
    #         "same_image_prob": 0.3,
    #         "enable_color_loss": True,
    #     }
    # )

    def_config.update(
        {
            # "ckpt": "train_log/256_BLENDFACE_AlphaDise_New_ID_8/ckpt/630000.pth",
            "net_g_cfg": {
                "img_resolution": 256,
                "img_channels": 3,
                "num_encoder": 5,
                "base_ch": 64,
                "max_ch": 512,
                "id_dim": 512,
                "w_dim": 256,
                "mapping_num": 4,
                "skip_index": 2,
            },
            "net_d_cfg": {
                "img_resolution": 256,
                "img_channels": 3,
                "base_ch": 64,
                "max_ch": 512,
                "group_size": 4,
            },
            "log_path": "train_log/256_BLENDFACE_AlphaDise_New_ID_8_Injection_2",
            "enable_wfm_loss": True,
            "wfm_loss_weight": {
                0: 2.0,
                1: 1.0,
            },
            "same_image_prob": 0.0,
            "discriminator_typt": DISCRIMINATOR_TYPT.ALPHAFACE,
            "r1_reg_step": 16,
            "perceptual_loss_weight": {
                # vgg16
                "relu1_2": 1.0,
                "relu2_2": 1.0,
                "relu3_3": 0.5,
                "relu4_2": 0.5,
                # "pool1": 0.2,
                # "pool2": 0.2,
                # "pool3": 0.2,
                # "pool4": 0.2,
                # "pool5": 0.2,
            },
            "enable_color_loss": True,
            "batch_size": 12,
            "d_train_setp": 2,
            "id_loss_weight": 8,
        }
    )

    trainer = Trainer(**def_config)

    try:
        trainer.train()
    finally:
        trainer.save_ckpt()
