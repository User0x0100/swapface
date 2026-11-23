"""ROCm R1 convolution path: unfold + matmul avoids slow conv gradgrad kernels."""

import copy
from types import MethodType

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _r1_matmul_conv2d(self: nn.Conv2d, x: Tensor) -> Tensor:
    columns = F.unfold(
        x, kernel_size=self.kernel_size, dilation=self.dilation,
        padding=self.padding, stride=self.stride,
    )
    result = torch.matmul(self.weight.flatten(1), columns)
    if self.bias is not None:
        result = result + self.bias.to(result.dtype).view(1, -1, 1)
    height = (x.shape[-2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
    width = (x.shape[-1] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
    return result.reshape(x.shape[0], self.out_channels, height, width)


def make_r1_matmul_discriminator(discriminator: nn.Module) -> nn.Module:
    """Keep a separate forward implementation while sharing every trainable parameter."""
    r1_model = copy.deepcopy(discriminator)
    for name, parameter in discriminator.named_parameters():
        module_path, _, parameter_name = name.rpartition(".")
        r1_model.get_submodule(module_path)._parameters[parameter_name] = parameter
    for module in r1_model.modules():
        if isinstance(module, nn.Conv2d):
            module.forward = MethodType(_r1_matmul_conv2d, module)
    return r1_model
