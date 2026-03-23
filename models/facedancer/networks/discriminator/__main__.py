from . import Discriminator, AlphaFaceDiscriminator, Stylegan2DiscriminatorLite

import torch
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
img_resolution = 256

# model = Discriminator(img_resolution).to(device)
# model = Stylegan2DiscriminatorLite(img_resolution, group_size=1).to(device)
model = AlphaFaceDiscriminator(img_resolution).to(device)
# model = UNetDiscriminatorSN().to(device)
# model = StyleGAN2Discriminator(img_size).to(device)

model.eval()

x = torch.randn(batch_size, 3, img_resolution, img_resolution, device=device)
model(x)

with torch.inference_mode():
    y, feats = model(x, return_feats=True)
print("各层输出特征形状:")
for i, feat in enumerate(feats):
    print(f"feat[{i}]: {feat.shape}")
print("最终输出形状:", y.shape)

print("\n模型结构信息（torchinfo）:")
summary(model, input_data=(x), depth=2, col_names=("input_size", "output_size", "mult_adds"), row_settings=("var_names",))


flops = FlopCountAnalysis(model, x)
print(f"\n模型总FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
