from . import Discriminator, AlphaFaceDiscriminator, Stylegan2DiscriminatorLite

import torch
from torchinfo import summary

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
    y, feats = model(x, True)
print([feat.shape for feat in feats])
print(y.shape)


summary(
    model,
    input_data=(x),
    depth=2,
    col_names=(
        "input_size",
        "output_size",
        "num_params",
        "kernel_size",
        "mult_adds",
    ),
    row_settings=("var_names",),
)
