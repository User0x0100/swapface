import copy
import random
import itertools
from typing import Any
from pathlib import Path

import torch
from torch.amp import autocast
import torch.nn.functional as NF
from torch import Tensor, optim, nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2
from tqdm import tqdm
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from losses import IDLoss, l1_loss_fn, VGGPerceptualLoss, DLoss, GANLoss, StyleLossLabChroma, r1_reg_loss, WFMLoss, IFSRLoss, DSSIMLoss

from .dataloader import datasetloader
from .networks import Generator, AlphaFaceDiscriminator
from misc.facealign import zoom_in
from misc.models.face_parsing import FaceParsing
from misc.models.idencoder import PROVIDER
import torchvision.transforms.functional as TF


assert torch.cuda.is_available(), "仅支持使用NVIDIA显卡训练"

EPS = 1e-8

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")

torch.manual_seed(42)
random.seed(42)


def print_dict(title: str, d: dict, indent: int = 2):
    print(f"{title}:")
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_dict("", v, indent + 2)
        else:
            print(" " * indent + f"{k:25}: {v}")


class Trainer:
    def __init__(
        self,
        src: list[tuple[str, float]],
        dst: list[tuple[str, float]],
        masked_train: bool = True,
        occ_mask: bool = False,
        batch_size: int = 10,
        lr: float = 1e-4,
        lr_scheduler_t_max: int = 0,
        r1_reg_step: int = 16,
        r1_gamma: float = 10.0,
        bf16: bool = True,
        device: str = "cuda",
        compile_module: bool = True,
        ckpt: str | None = None,
        log_path: str = "train_log/exper_0",
        log_interval: int = 10,
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
        # 模型配置
        net_g_cfg: dict | None = None,
        net_d_cfg: dict | None = None,
        # 身份损失
        id_encode_provider: IDLoss.Provider = IDLoss.Provider.BLENDFACE,
        id_loss_weight: float = 10.0,
        # 重建损失
        enable_rec_loss: bool = True,
        rec_loss_weight: float = 10.0,
        # VGG特征匹配
        enable_perceptual_loss: bool = True,
        perceptual_loss_weight: dict[str, float] = {
            # vgg19
            "conv1_2": 2.5,
            "conv2_2": 2.5,
            "conv3_3": 2.5,
            "conv4_3": 2.5,
            # "pool1": 1.0,
            # "pool2": 1.0,
            # "pool3": 1.0,
            # "pool4": 1.0,
            # "pool5": 1.0,
        },
        # 判别器中间特征得弱特征匹配
        enable_wfm_loss: bool = False,
        wfm_loss_weight: dict[int, float] = {
            # 0: 10.0,
            # 1: 10.0,
            # 2: 10.0,
            # 3: 10.0,
            # 0: 1.0,
            # 1: 1.0,
            # 2: 1.0,
            # 3: 1.0,
        },
        # arcfaceid编码器前几层特征的带边界特征匹配
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
        # 色彩一致损失
        enable_color_loss: bool = False,
        color_loss_weight: float = 0.1,
        # 结构损失
        enable_dssim_loss: bool = False,
        dssim_loss_weight: float = 10.0,
    ):

        args = locals().copy()
        for k in ["src", "dst", "self"]:
            args.pop(k)

        print_dict("Train_Info", args)

        self.rng = random.Random()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.masked_train = masked_train
        self.r1_reg_step = r1_reg_step
        self.r1_gamma = r1_gamma
        self.enable_rec_loss = enable_rec_loss
        self.enable_perceptual_loss = enable_perceptual_loss
        self.enable_wfm_loss = enable_wfm_loss
        self.enable_ifsr_loss = enable_ifsr_loss
        self.enable_color_loss = enable_color_loss
        self.enable_dssim_loss = enable_dssim_loss

        self.bf16 = bool(bf16 and torch.cuda.is_bf16_supported())
        if bf16 and not self.bf16:
            print("Warning: 当前显卡不支持 BF16")

        self.sample_save_every = sample_save_every
        self.weight_save_every = weight_save_every
        self.log_interval = log_interval
        self.use_cosine_lr = lr_scheduler_t_max > 0

        # ========================= Init Model =========================

        if ckpt is not None:
            if not Path(ckpt).exists():
                raise FileNotFoundError(f"ckpt file: {ckpt} Not found")

            print(f"Loading ckpt from {ckpt}")
            ckpt: dict[str, Any] = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

            self.iter = ckpt["iter"]

            print(f"ckpt info:\n  {'iter':25}: {self.iter}")
            print_dict("net_g", ckpt["net_g"]["network_cfg"])
            print_dict("net_d", ckpt["net_d"]["network_cfg"])

            self.img_resolution = ckpt["net_g"]["network_cfg"]["img_resolution"]

            net_g = Generator(**ckpt["net_g"]["network_cfg"])
            net_d = AlphaFaceDiscriminator(**ckpt["net_d"]["network_cfg"])
            net_g.load_state_dict(ckpt["net_g"]["state_dict"])
            net_d.load_state_dict(ckpt["net_d"]["state_dict"])

        else:
            self.iter, self.img_resolution = 0, net_g_cfg["img_resolution"]
            net_g = Generator(**net_g_cfg)
            net_d = AlphaFaceDiscriminator(**net_d_cfg)

        # net_d = Discriminator(**net_d_cfg)
        self.net_g = net_g.to(self.device).train()
        self.net_d = net_d.to(self.device).train()

        self.net_g_ema = copy.deepcopy(self.net_g)
        self.net_g_ema.eval().requires_grad_(False)

        # ========================= Optim =========================
        self.optim_g = optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)
        self.optim_d = optim.Adam(self.net_d.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)

        if self.use_cosine_lr:
            self.lr_scheduler_g = CosineAnnealingLR(self.optim_g, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)
            self.lr_scheduler_d = CosineAnnealingLR(self.optim_d, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)

        # ========================= LOSS =========================

        self.d_loss = DLoss(weight=1.0, reduction="mean").to(self.device)
        self.gan_loss = GANLoss(weight=1.0, reduction="mean").to(self.device)

        self.id_loss = IDLoss(weight=id_loss_weight, provider=id_encode_provider).to(self.device)

        if self.enable_rec_loss:
            self.rec_loss = l1_loss_fn(weight=rec_loss_weight, reduction="none" if self.masked_train else "mean")

        if self.enable_perceptual_loss:
            self.perceptual_loss = VGGPerceptualLoss(layer_weights=perceptual_loss_weight, reduction="mean").to(self.device)

        if self.enable_ifsr_loss:
            self.ifsr_loss = IFSRLoss(ifsr_scale=ifsr_scale, ifsr_weight=ifsr_weight, idencoder_provider=PROVIDER.MS1MV3_ARCFACE_R100_FP16).to(self.device)

        if self.enable_wfm_loss:
            self.wfm_loss = WFMLoss(layer_weights=wfm_loss_weight, criterion="l1").to(self.device)

        if self.enable_color_loss:
            self.color_loss = StyleLossLabChroma(weight=color_loss_weight, range_norm=True).to(self.device)

        if self.enable_dssim_loss:
            self.dssim_loss = DSSIMLoss(weight=dssim_loss_weight, reduction="mean").to(self.device)

        # ========================= LOG =========================
        base_log_path = Path(log_path)
        self.ckpt_dir = base_log_path.joinpath("ckpt")
        self.sample_dir = base_log_path.joinpath("sample")
        self.tensorboard_dir = base_log_path.joinpath("tensorboard")

        for p in [self.ckpt_dir, self.sample_dir, self.tensorboard_dir]:
            p.mkdir(exist_ok=True, parents=True)

        self.log_writer = SummaryWriter(self.tensorboard_dir)

        # ========================= Sample =========================

        major, _minor = torch.cuda.get_device_capability(device)
        hw_decoder = major >= 8
        pipe = datasetloader(
            batch_size=self.batch_size,
            num_threads=16,
            prefetch_queue_depth=20,
            py_num_workers=8,
            py_start_method="spawn",
            device_id=self.device.index,
            img_resolution=self.img_resolution,
            src=src,
            dst=dst,
            hw_decoder=hw_decoder,
            rotation_range=(-10.0, 10.0),
            scale_range=(-0.3, 0.25),
            tx_range=(-0.15, 0.15),
            ty_range=(-0.15, 0.15),
        )

        print(f"hw_decoder={hw_decoder}")

        self.sample_output_map = ["src", "dst", "theta_restore"]
        self.dataset = DALIGenericIterator(pipelines=pipe, output_map=self.sample_output_map, auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        # ========================= compile_module =========================
        self.train_module: dict[str, nn.Module] = {}

        for net_name in ["net_g", "net_d"]:
            net = getattr(self, net_name)
            if compile_module:
                self.train_module[net_name] = torch.compile(net, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})
            else:
                self.train_module[net_name] = net

        face_parser = FaceParsing(range_norm=True, occ=occ_mask).to(device=self.device)
        if not occ_mask:
            face_parser = face_parser.eval()
        face_parser.requires_grad_(False)
        self.face_parser = torch.compile(face_parser, fullgraph=True, dynamic=False, options={"max_autotune": True, "epilogue_fusion": True})

        mask_blur_kernel_size = max(3, self.img_resolution // 32)
        if mask_blur_kernel_size % 2 == 0:
            mask_blur_kernel_size += 1
        self.mask_blur_kernel_size = mask_blur_kernel_size
        self.mask_blur_sigma = self.img_resolution / 128

    @torch.no_grad()
    def log(self, k: str, v: Tensor, right_now: bool = False) -> None:
        if self.iter % self.log_interval == 0 or right_now:
            self.log_writer.add_scalar(f"Loss/{k}", v.detach().mean().item(), self.iter)

    @torch.no_grad()
    def fetch_sample(self) -> tuple[Tensor, ...]:
        data: dict[str, Tensor] = self.dataset.next()[0]
        src, dst, theta_restore = (data[k] for k in self.sample_output_map)
        return src, dst, theta_restore

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

    def loss_grad_map(self, loss: Tensor, x: Tensor) -> Tensor:
        (grad,) = torch.autograd.grad(outputs=loss.sum(), inputs=x, retain_graph=False, create_graph=False)

        h = grad.detach().float().abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

        h = torch.log1p(h)
        h = h / h.amax(dim=(2, 3), keepdim=True).clamp_min(EPS)

        h = h.mul(2.0).sub(1.0)  # [0,1] -> [-1,1]
        h = h.expand(-1, 3, -1, -1).contiguous()

        return h

    @torch.no_grad()
    def apply_gaussian_blur_use_mask(self, x: Tensor) -> tuple[Tensor, Tensor]:

        x_mask = self.face_parser(x)
        x_mask = TF.gaussian_blur(x_mask, self.mask_blur_kernel_size, self.mask_blur_sigma).clamp_(0.0, 1.0)
        x_inv_mask = 1.0 - x_mask

        low = NF.interpolate(x, size=16, mode="bilinear", align_corners=False)
        blur = NF.interpolate(low, size=self.img_resolution, mode="bilinear", align_corners=False)
        x_blur = TF.gaussian_blur(blur, kernel_size=51, sigma=15)

        return (x * x_mask + x_inv_mask * x_blur, x_mask)

    def train(self):

        net_d, net_g = (self.train_module[module_name] for module_name in ("net_d", "net_g"))
        sample_src, sample_dst, sample_theta_restore = self.fetch_sample()

        for self.iter in tqdm(itertools.count(start=self.iter), initial=self.iter, mininterval=1.0, bar_format="{n_fmt:7} | speed {rate_fmt:3} | train time {elapsed}"):
            src, dst, theta_restore = self.fetch_sample()

            torch.compiler.cudagraph_mark_step_begin()

            # ========================= forward g =========================
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                with torch.no_grad():
                    src_id_feats = self.id_loss.get_id_feats(zoom_in(src))
                fake: Tensor = net_g(dst, src_id_feats)

            dst_org = dst
            if self.masked_train:
                dst, dst_mask = self.apply_gaussian_blur_use_mask(dst)

            # ========================= train d =========================
            net_d.requires_grad_(True)
            self.optim_d.zero_grad()
            is_r1_reg_step = self.iter % self.r1_reg_step == 0

            if is_r1_reg_step:
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                    fake_img = fake.detach().float()
                    real_img = dst.detach().float().requires_grad_(True)

                    # R1: use raw discriminator in FP32
                    fake_score = self.net_d(fake_img)
                    real_score = self.net_d(real_img)

                    d_loss = self.d_loss(fake_score, real_score)
                    self.log("d_loss", d_loss, True)

                    r1_loss_raw = r1_reg_loss(real_score, real_img, gamma=self.r1_gamma)
                    self.log("r1_loss_raw", r1_loss_raw)
                    r1_loss = r1_loss_raw * self.r1_reg_step
                    self.log("r1_loss", r1_loss)
                    d_loss += r1_loss
            else:
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                    fake_score = net_d(fake.detach())
                    real_score = net_d(dst.detach())

                    d_loss = self.d_loss(fake_score, real_score)
                    self.log("d_loss", d_loss, True)

            d_loss.backward()
            self.optim_d.step()

            if self.use_cosine_lr:
                self.lr_scheduler_d.step()

            # ========================= train g =========================
            net_d.requires_grad_(False)
            self.optim_g.zero_grad()

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                g_loss: Tensor = torch.tensor(0.0, device=self.device)

                # gan_loss
                if self.enable_wfm_loss:
                    fake_score, fake_feats = net_d(fake, True)
                else:
                    fake_score = net_d(fake)

                gan_loss = self.gan_loss(fake_score)
                self.log("gan_loss", gan_loss)
                g_loss += gan_loss

                # wfm_loss
                if self.enable_wfm_loss:
                    with torch.no_grad():
                        _, real_feats = net_d(dst, True)
                    wfm_loss = self.wfm_loss(fake_feats, real_feats)
                    self.log("wfm_loss", wfm_loss)
                    g_loss += wfm_loss

                # id_loss
                grid = NF.affine_grid(theta_restore, size=fake.shape, align_corners=False)
                fake_restored = NF.grid_sample(fake, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
                fake_id_feats = self.id_loss.get_id_feats(zoom_in(fake_restored))
                id_loss = self.id_loss(fake_id_feats, src_id_feats)
                self.log("id_loss", id_loss)
                g_loss += id_loss

                # ifsr_loss
                if self.enable_ifsr_loss:
                    fake_ifsr_feats = self.ifsr_loss.get_ifsr_feats(fake)
                    with torch.no_grad():
                        dst_ifsr_feats = self.ifsr_loss.get_ifsr_feats(dst)
                    ifsr_loss = self.ifsr_loss(fake_ifsr_feats, dst_ifsr_feats)
                    self.log("ifsr_loss", ifsr_loss)
                    g_loss += ifsr_loss

                # perceptual_loss
                if self.enable_perceptual_loss:
                    perceptual_loss = self.perceptual_loss(fake, dst)
                    self.log("perceptual_loss", perceptual_loss)
                    g_loss += perceptual_loss

                # rec_loss
                if self.enable_rec_loss:
                    rec_loss = self.rec_loss(fake, dst)  # B C H W
                    if self.masked_train:
                        rec_loss *= dst_mask
                        rec_loss = rec_loss.sum(dim=(1, 2, 3)) / (dst_mask.sum(dim=(1, 2, 3)) * rec_loss.size(1) + EPS)
                        rec_loss = rec_loss.mean()
                    self.log("rec_loss", rec_loss)
                    g_loss += rec_loss

                # color_loss
                if self.enable_color_loss:
                    if self.masked_train:
                        fake_color_in = fake * dst_mask + dst.detach() * (1.0 - dst_mask)
                    else:
                        fake_color_in = fake
                    color_loss = self.color_loss(fake_color_in, dst)
                    self.log("color_loss", color_loss)
                    g_loss += color_loss

                # dssim_loss
                if self.enable_dssim_loss:
                    dssim_loss = self.dssim_loss(fake, dst)
                    self.log("dssim_loss", dssim_loss)
                    g_loss += dssim_loss

            g_loss.backward()
            self.optim_g.step()

            if self.use_cosine_lr:
                self.lr_scheduler_g.step()

            self.update_ema()

            if self.iter % self.weight_save_every == 0:
                self.save_ckpt()

            if self.iter % self.sample_save_every == 0:
                with torch.no_grad():
                    half = self.batch_size // 2
                    src_vis = torch.cat((sample_src[:half], src[: self.batch_size - half]), dim=0)
                    dst_vis = torch.cat((sample_dst[:half], dst_org[: self.batch_size - half]), dim=0)

                    theta_restore_vis = torch.cat((sample_theta_restore[:half], theta_restore[: self.batch_size - half]), dim=0)
                    grid_vis = NF.affine_grid(theta_restore_vis, size=fake.shape, align_corners=False)
                    dst_restored_vis = NF.grid_sample(dst_vis, grid_vis, mode="bilinear", padding_mode="reflection", align_corners=False)

                    src_id_feats_vis = self.id_loss.get_id_feats(zoom_in(src_vis))
                    fake_vis: Tensor = self.net_g_ema(dst_vis, src_id_feats_vis)

                    grid = [src_vis, dst_vis, fake_vis, dst_restored_vis]

                    # ========================= GAN loss grad map =========================
                    fake_for_gan_grad = fake_vis.detach().requires_grad_(True)
                    with torch.enable_grad():
                        fake_score_vis = net_d(fake_for_gan_grad)
                        gan_loss_vis = self.gan_loss(fake_score_vis)
                        gan_grad_map = self.loss_grad_map(gan_loss_vis, fake_for_gan_grad)
                    grid.append(gan_grad_map)

                    # ========================= ID loss grad map =========================
                    with torch.enable_grad():
                        fake_for_id_grad = fake_vis.detach().requires_grad_(True)
                        grid_id_vis = NF.affine_grid(theta_restore_vis, size=fake_for_id_grad.shape, align_corners=False)
                        fake_for_id_grad_restored = NF.grid_sample(fake_for_id_grad, grid_id_vis, mode="bilinear", padding_mode="reflection", align_corners=False)

                        fake_id_feats_vis = self.id_loss.get_id_feats(zoom_in(fake_for_id_grad_restored))
                        id_loss_vis = self.id_loss(fake_id_feats_vis, src_id_feats_vis.detach())
                        id_grad_map = self.loss_grad_map(id_loss_vis, fake_for_id_grad)

                    grid.append(id_grad_map)

                    attn_map: list[Tensor] = []

                    get_attention_maps = getattr(self.net_g_ema, "get_attention_maps", None)
                    if callable(get_attention_maps):
                        attn_map = get_attention_maps() or []

                    if len(attn_map) > 0:
                        for m in attn_map:
                            m = m.mean(dim=1, keepdim=True)
                            m = NF.interpolate(m, size=self.img_resolution, mode="bilinear", align_corners=False)
                            m = m.expand(-1, 3, -1, -1)

                            amin = m.amin(dim=(2, 3), keepdim=True)
                            amax = m.amax(dim=(2, 3), keepdim=True)
                            m = 2.0 * (m - amin) / (amax - amin + EPS) - 1.0

                            grid.append(m)

                    grid = torch.cat(grid, dim=0)
                    grid.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
                    grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB -> BGR
                    grid = grid.permute(1, 2, 0)  # CHW -> HWC
                    grid_cpu = grid.to(device="cpu", dtype=torch.uint8).numpy()
                cv2.imwrite(self.sample_dir / f"{self.iter}.png", grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


if __name__ == "__main__":
    src = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        # ("/opt/share/deepfake/dataset_1/vggface2_hq512/align_result", 0.0),
    ]

    dst = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/vggface2_hq512/align_result", 0.0),
        ("/opt/share/deepfake/dataset_1/RealOcc/image/realign_arcface_dst", 1.0),
        # ("/opt/share/deepfake/dataset_1/oneman/1_align_results/", 0.0),
    ]

    def_config = {"src": src, "dst": dst}

    def_config.update(
        {
            "ckpt": "train_log/512-MS1MV3_ARCFACE_R50_FP16/ckpt/328696.pth",
            "masked_train": False,
            "occ_mask": False,
            "log_path": "train_log/512-MS1MV3_ARCFACE_R50_FP16",
            "batch_size": 16,
            "id_encode_provider": IDLoss.Provider.MS1MV3_ARCFACE_R50_FP16,
            "net_g_cfg": {
                # "img_resolution": 512,
                # "img_channels": 3,
                # "num_depth": 4,
                # "num_latent": 6,
                # "base_ch": 64,
                # "max_ch": 2048,
                # "id_dim": 512,
                # "skip": True,
                "img_resolution": 512,
                "img_channels": 3,
                "num_depth": 5,
                "num_latent": 6,
                "base_ch": 16,
                "max_ch": 2048,
                "id_dim": 512,
                "skip": True,
            },
            "net_d_cfg": {
                "img_resolution": 512,
                "img_channels": 3,
                "base_ch": 64,
                "max_ch": 512,
                # "group_size": 4,
            },
        }
    )

    trainer = Trainer(**def_config)

    try:
        trainer.train()
    finally:
        trainer.save_ckpt()
