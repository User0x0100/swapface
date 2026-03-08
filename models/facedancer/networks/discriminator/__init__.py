from .alphaface import AlphaFaceDiscriminator
from .original import Discriminator
from .stylegan2 import Stylegan2DiscriminatorLite
from .unet_sn import UNetDiscriminatorSN


__all__ = [
    "AlphaFaceDiscriminator",
    "Discriminator",
    "Stylegan2DiscriminatorLite",
    "UNetDiscriminatorSN",
]
