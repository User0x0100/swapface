import random
import itertools
from pathlib import Path
from typing import Any, Mapping

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import cv2
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from losses import IDLoss, l1_loss_fn, PerceptualLoss, DLoss, GANLoss, StyleLossLabChroma, DINOv2PerceptualLoss, r1_reg_loss, WFMLoss, IFSRLoss

from .dataloader import datasetloader
from .networks import Generator, Discriminator


EPS = 1e-8

assert torch.cuda.is_available(), "仅支持使用NVIDIA显卡训练"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Trainer:
    def __init__(
        self,
        src: list[tuple[str, float]],
        dst: list[tuple[str, float]],
        batch_size: int = 10,
        lr: float = 1e-4,
        lr_scheduler_t_max: int = 0,
        d_train_setp: int = 3,
        r1_reg_step: int = 1,
        bf16: bool = True,
        device: str = "cuda:0",
        compile_module: bool = True,
        ckpt: str | None = None,
        log_path: str = "train_log/exper_0",
        log_interval: int = 10,
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
        same_image_prob: float = 0.1,
        # 模型配置
        net_g_cfg={
            "input_res": 256,
            "num_encoder": 5,
            "base_ch": 64,
            "max_ch": 512,
            "mapping_size": 512,
            "z_dim": 512,
            "skip_conn_start_with": 2,
        },
        net_d_cfg: Mapping[str, int] = {
            "input_res": 256,
            "num_encoder": 5,
            "base_ch": 64,
            "max_ch": 512,
        },
        id_encode_provider: IDLoss.Provider = IDLoss.Provider.BLENDFACE,
        id_loss_weight: float = 5.0,
        rec_loss: float = 2.5,
        use_dinov2: bool = False,
        perceptual_loss_weight: dict[str, float] = {
            # vgg19
            "relu1_2": 0.25,
            "relu2_2": 0.25,
            "relu3_3": 0.25,
            "relu4_2": 0.25,
            # DINOv2
            # 2: 0.5,  # 低层纹理/色彩一致性
            # 5: 1.0,  # 皮肤结构主力层
            # 8: 0.5,  # 面部组件对齐（眼鼻口位置）
        },
        wfm_loss_weight: dict[int, float] = {
            1: 0.5,
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
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.d_train_setp = d_train_setp
        self.r1_reg_step = r1_reg_step
        self.enable_ifsr_loss = enable_ifsr_loss

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

            print(f"ckpt Info:\n" f"  {'iter':25}: {self.iter}")
            print("net_g:")
            for k, v in ckpt["net_g"]["network_cfg"].items():
                print(f"  {k:25}: {v}")
            print("net_d:")
            for k, v in ckpt["net_d"]["network_cfg"].items():
                print(f"  {k:25}: {v}")

            self.size = ckpt["net_g"]["network_cfg"]["input_res"]

            self.net_g = Generator(**ckpt["net_g"]["network_cfg"])
            self.net_d = Discriminator(**ckpt["net_d"]["network_cfg"])
            self.net_g.load_state_dict(ckpt["net_g"]["state_dict"])
            self.net_d.load_state_dict(ckpt["net_d"]["state_dict"])

        else:

            self.iter, self.size = 0, net_g_cfg["input_res"]
            self.net_g = Generator(**net_g_cfg)
            self.net_d = Discriminator(**net_d_cfg)

        self.net_g.to(self.device)
        self.net_d.to(self.device)
        self.net_g.train()
        self.net_d.train()

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

        perceptual_loss = DINOv2PerceptualLoss if use_dinov2 else PerceptualLoss
        self.perceptual_loss = perceptual_loss(layer_weights=perceptual_loss_weight).to(self.device)

        if self.enable_ifsr_loss:
            self.ifsr_loss = IFSRLoss(ifsr_scale=ifsr_scale, ifsr_dict=ifsr_weight).to(self.device)

        self.wfm_loss = WFMLoss(layer_weights=wfm_loss_weight, criterion="l1").to(self.device)

        self.color_loss = StyleLossLabChroma(weight=1.0, range_norm=True).to(self.device)

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
            num_threads=8,
            prefetch_queue_depth=10,
            py_num_workers=1,
            py_start_method="spawn",
            device_id=self.device.index,
            resize=self.size,
            src=src,
            dst=dst,
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
    def save_ckpt(self):

        net_g = {
            "network_cfg": self.net_g.network_cfg,
            "state_dict": self.net_g.state_dict(),
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

                id_feats_mix = False
                if id_feats_mix and random.random() < 0.5:

                    w1_all = self.net_g.mapping_z_source(src_id_feats)
                    w2_all = self.net_g.mapping_z_source(torch.randn_like(src_id_feats))

                    cutoff = torch.randint(1, len(w1_all), (1,)).item()

                    w_all = []
                    for i in range(len(w1_all)):
                        if i < cutoff:
                            w_all.append(w1_all[i])
                        else:
                            w_all.append(w2_all[i])
                else:
                    w_all = None

                fake = net_g(dst, src_id_feats, w_all)

                # ========================= train d =========================
                if self.iter % self.d_train_setp == 0:
                    self.optim_d.zero_grad()
                    d_loss: Tensor = torch.tensor(0.0, device=self.device)

                    real_img = dst.detach()

                    d_step = self.iter // self.d_train_setp
                    is_r1_reg_step = d_step % self.r1_reg_step == 0

                    real_img.requires_grad_(is_r1_reg_step)

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
                loss: Tensor

                # gan_loss
                fake_global_score, fake_feats = net_d(fake, True)
                global_gan_loss = self.gan_loss(fake_global_score)
                self.log("global_gan_loss", global_gan_loss)
                loss = global_gan_loss

                # wfm_loss
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
                self.log("perceptual_loss", perceptual_loss)
                loss += perceptual_loss

                # rec_loss
                rec_loss = self.rec_loss(fake, dst)  # BCHW
                rec_loss = rec_loss.mean(dim=[1, 2, 3])
                rec_loss = (rec_loss * is_same).sum() / is_same.sum().clamp_min(1.0)
                self.log("rec_loss", rec_loss)
                loss += rec_loss

                # color_loss
                color_loss = self.color_loss(fake, dst)
                self.log("color_loss", color_loss)
                loss += color_loss

            loss.backward()
            self.optim_g.step()

            if self.enable_lr_scheduler:
                self.lr_scheduler_g.step()
                self.lr_scheduler_d.step()

            if self.iter % self.weight_save_every == 0:
                self.save_ckpt()

            if self.iter % self.sample_save_every == 0:
                with torch.inference_mode():
                    attn_map: list[Tensor] = self.net_g.get_attention_maps()

                    if len(attn_map) == 0:
                        attn_norm = torch.full_like(src, -1.0)
                    else:
                        maps = []
                        for m in attn_map:
                            m = m.mean(dim=1, keepdim=True)
                            m = F.interpolate(m, size=self.size, mode="bilinear", align_corners=False)
                            maps.append(m)
                        attn = torch.mean(torch.stack(maps, dim=0), dim=0).expand(-1, 3, -1, -1)
                        amin = attn.amin(dim=(2, 3), keepdim=True)
                        amax = attn.amax(dim=(2, 3), keepdim=True)
                        attn_norm = 2.0 * (attn - amin) / (amax - amin + 1e-6) - 1.0

                grid = torch.cat((src, dst, fake, attn_norm), dim=0)
                grid.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
                grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB -> BGR
                grid = grid.permute(1, 2, 0)  # CHW -> HWC
                grid_cpu = grid.to(device="cpu", dtype=torch.uint8).numpy()
                cv2.imwrite(self.sample_dir / f"{self.iter}.png", grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


if __name__ == "__main__":
    src = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
    ]

    dst = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        # ("/opt/share/deepfake/dataset_1/RealOcc/image/realign_arcface_dst", 1.0),
        # ("/opt/share/deepfake/dataset_1/youtube/What_s_considered_tall_in_South_Korea_Street_Interview_align_results", 0.0),
        # ("/opt/share/deepfake/dataset_1/youtube/4k_Face_Close_Up_HDR_Video_Vivid_Colors_Ambient_Sound_-_Relaxing_align_results", 0.0),
        # ("/opt/share/deepfake/dataset_1/oneman/1_align_results/", 0.0),
    ]

    def_config = {
        "src": src,
        "dst": dst,
    }

    def_config.update(
        {
            "log_path": "train_log/256_BLENDFACE_ModCONV_use_refinement",
            "id_loss_weight": 7.5,
        }
    )

    trainer = Trainer(**def_config)

    try:
        trainer.train()
    finally:
        trainer.save_ckpt()
