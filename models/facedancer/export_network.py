import torch
from torch import nn, Tensor
import onnx


def exp(model: nn.Module, sample_input: tuple[Tensor], opset_version: int | None, onnx_path: str):

    model.eval()

    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        opset_version=opset_version,
        export_params=True,
        dynamo=True,
        fallback=False,
        optimize=True,
        verify=True,
        external_data=False,
        keep_initializers_as_inputs=False,
        verbose=True,
        profile=True,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    print(f"模型成功导出到: {onnx_path}")


if __name__ == "__main__":
    from .networks import Generator

    bs = 1
    size = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Generator(size).to(device)

    sample_input = (torch.randn([bs, 3, size, size], dtype=torch.float, device=device), torch.randn([bs, 512], dtype=torch.float, device=device))

    exp(net, sample_input, None, "1.onnx")
