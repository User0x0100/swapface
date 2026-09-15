from .adversarial import DiscriminatorAdversarialLoss, GeneratorAdversarialLoss
from .color import LabStyleLoss
from .facs import FACSConsistencyLoss
from .functional import (
    make_bce_loss,
    make_bce_with_logits_loss,
    make_charbonnier_loss,
    make_l1_loss,
    make_mse_loss,
    make_orthogonal_loss,
    r1_reg_loss,
)
from .gaze import GazeLoss
from .hrffa import HRFFAFacialGeometryLoss
from .identity import IdentityLoss, IFSRLoss
from .perceptual import (
    DINOv2PerceptualLoss,
    VGGPerceptualLoss,
    WeightedFeatureMatchingLoss,
)
from .structural import DSSIMLoss
from .vgg import get_supported_vgg_types, get_vgg_layer_names

__all__ = [
    "DINOv2PerceptualLoss",
    "DSSIMLoss",
    "DiscriminatorAdversarialLoss",
    "FACSConsistencyLoss",
    "GazeLoss",
    "GeneratorAdversarialLoss",
    "HRFFAFacialGeometryLoss",
    "IFSRLoss",
    "IdentityLoss",
    "LabStyleLoss",
    "VGGPerceptualLoss",
    "WeightedFeatureMatchingLoss",
    "get_supported_vgg_types",
    "get_vgg_layer_names",
    "make_bce_loss",
    "make_bce_with_logits_loss",
    "make_charbonnier_loss",
    "make_l1_loss",
    "make_mse_loss",
    "make_orthogonal_loss",
    "r1_reg_loss",
]
