from torch import Tensor, nn


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return x + identity


class QCFaceIResNet(nn.Module):
    """IResNet backbone used by the official QCFace checkpoints."""

    def __init__(self, layers: tuple[int, int, int, int], num_features: int = 512) -> None:
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=2e-5, momentum=0.9)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9),
            )

        layers: list[nn.Module] = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(IBasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        return self.features(x)


def qcface_iresnet100() -> QCFaceIResNet:
    return QCFaceIResNet((3, 13, 30, 3))
