from . import *

import torch
from fvcore.nn import FlopCountAnalysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
img_size = 256

# model = Discriminator(img_size).to(device)
model = Stylegan2DiscriminatorLite(img_size, group_size=1).to(device)
# model = UNetDualDiscriminator().to(device)
# model = UNetDiscriminatorSN().to(device)
# model = StyleGAN2Discriminator(img_size).to(device)

model.eval()

x_target = torch.randn(batch_size, 3, img_size, img_size, device=device)

model(x_target)

flops = FlopCountAnalysis(model, x_target)

total_flops = flops.total()
total_params = sum(p.numel() for p in model.parameters())

print("\n=== Model Profile ===")
labels = [
    ("Batch size", batch_size),
    ("Network Info", model.network_cfg),
    ("Params", f"{total_params/1e6:.3f} M"),
    ("FLOPs (total)", f"{total_flops/1e9:.3f} GFLOPs"),
]
max_label_len = max(len(label) for label, _ in labels)
for label, value in labels:
    print(f"{label:<{max_label_len}} : {value}")
print()
