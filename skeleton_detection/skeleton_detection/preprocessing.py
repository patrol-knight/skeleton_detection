from typing import Tuple

import PIL.Image


WHITE = (255, 255, 255)


def preprocess_image(
    image_source,
    target_size: Tuple[int, int],
    background_color=WHITE,
):
    """Normalize an image to target_size without distortion."""
    target_width, target_height = target_size

    if isinstance(image_source, PIL.Image.Image):
        image = image_source.convert("RGB")
    else:
        image = PIL.Image.open(image_source).convert("RGB")

    source_width, source_height = image.size
    scale = min(target_width / source_width, target_height / source_height, 1.0)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    if (resized_width, resized_height) != image.size:
        image = image.resize((resized_width, resized_height), PIL.Image.Resampling.LANCZOS)

    canvas = PIL.Image.new("RGB", target_size, background_color)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(image, (offset_x, offset_y))
    return canvas
