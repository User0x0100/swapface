"""训练检查点与部署模型共享的轻量格式约定。"""

CHECKPOINT_VERSION = 2

ONNX_CONTRACT = {
    "faceswap.format": "1",
    "faceswap.face_alignment": "ffhq",
    "faceswap.identity_alignment": "arcface112",
    "faceswap.color": "RGB",
    "faceswap.range": "[-1,1]",
}
