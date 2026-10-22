"""离线把 NatOcc 的遮挡图/Alpha mask 配对成训练用 RGBA 素材。"""

import argparse
from pathlib import Path

from PIL import Image


def prepare_occluders(images: Path, masks: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    converted = 0
    for image_path in sorted(images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        mask_path = masks / (image_path.stem + ".png")
        if not mask_path.is_file():
            raise FileNotFoundError(f"缺少遮挡物掩码：{mask_path}")
        with Image.open(image_path) as image, Image.open(mask_path) as alpha:
            rgb = image.convert("RGB")
            alpha = alpha.convert("L")
            if alpha.size != rgb.size:
                alpha = alpha.resize(rgb.size, Image.Resampling.BILINEAR)
            rgba = rgb.copy()
            rgba.putalpha(alpha)
            bounds = rgba.getchannel("A").getbbox()
            if bounds is None:
                raise ValueError(f"遮挡素材完全透明：{image_path}")
            rgba.crop(bounds).save(output / (image_path.stem + ".png"))
            converted += 1
    if not converted:
        raise FileNotFoundError(f"遮挡素材目录为空：{images}")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="将 NatOcc 前景图像及 PNG 掩码转换为 RGBA 素材")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"已生成 {prepare_occluders(args.images, args.masks, args.output)} 个 RGBA 遮挡素材：{args.output}")


if __name__ == "__main__":
    main()
