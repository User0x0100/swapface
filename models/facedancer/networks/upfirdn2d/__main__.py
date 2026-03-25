import torch
from torch.autograd import gradcheck
from . import UpFIRDn2d, DownFIRDn2d, upfirdn2d_fn
from torch.library import opcheck

if __name__ == "__main__":
    device = torch.device("cuda")

    x = torch.randn([1, 3, 16, 16], device=device, dtype=torch.double, requires_grad=True)
    y = torch.randn([1, 3, 16, 16], device=device, dtype=torch.double, requires_grad=True)

    examples = (
        x,
        torch.randn(4, 4, device=device, dtype=torch.float32),
        1,
        1,
        2,
        2,
        0,
        0,
        0,
        0,
        False,
        1.0,
    )

    opcheck_result = opcheck(upfirdn2d_fn, examples)
    print(f"Opcheck result: {opcheck_result}\n")

    up_op = UpFIRDn2d().to(device)
    down_op = DownFIRDn2d().to(device)

    x_up = up_op(x)
    y_up = up_op(y)
    torch.nn.functional.l1_loss(x_up, y_up).backward()

    x_down = down_op(x)
    y_down = down_op(y)
    torch.nn.functional.l1_loss(x_down, y_down).backward()

    def test_grad(op, input_tensor):
        print(f"Running gradcheck for {op.__class__.__name__}...")
        result = gradcheck(op, (input_tensor,), eps=1e-6, atol=1e-4)
        print(f"Gradcheck result: {result}\n")

    test_grad(up_op, x)
    test_grad(down_op, x)
