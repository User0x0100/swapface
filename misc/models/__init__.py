from enum import Enum


class ImageInputRange(Enum):
    """图像张量的输入值域。"""

    ZERO_TO_255 = "zero_to_255"
    ZERO_TO_ONE = "zero_to_one"
    MINUS_ONE_TO_ONE = "minus_one_to_one"


MODEL_REPOSITORY_ID = "leehx/model_hub"
