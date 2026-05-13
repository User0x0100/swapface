# from .generator import Generator
from .inswap import Generator
# from .generator_new import Generator
from .discriminator import AlphaFaceDiscriminator, Discriminator, Stylegan2DiscriminatorLite, UNetDiscriminatorSN


__all__ = [
    "Generator",
    "AlphaFaceDiscriminator",
    "Discriminator",
    "Stylegan2DiscriminatorLite",
    "UNetDiscriminatorSN",
]
