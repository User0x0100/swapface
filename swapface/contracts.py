"""训练检查点与部署模型共享的轻量格式约定。"""

CHECKPOINT_VERSION = 4

ONNX_CONTRACT = {
    "swapface.format": "2",
    "swapface.face_alignment": "ffhq",
    "swapface.identity_alignment": "arcface112",
    "swapface.color": "RGB",
    "swapface.range": "[-1,1]",
}
