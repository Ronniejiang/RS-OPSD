"""Shared in-memory image sizing for Student rollout and Teacher/probe inputs."""
from io import BytesIO

from PIL import Image


def load_capped_image(content: dict) -> Image.Image:
    value = content.get("image", content.get("path"))
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif isinstance(value, str):
        with Image.open(value) as source:
            image = source.convert("RGB")
    elif content.get("bytes") is not None:
        with Image.open(BytesIO(content["bytes"])) as source:
            image = source.convert("RGB")
    else:
        raise TypeError("Expected a local image path, PIL image or image bytes")
    limit = content.get("pa_opd_max_side")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("pa_opd_max_side must be a positive integer")
        if max(image.size) > limit:
            scale = limit / max(image.size)
            size = tuple(max(1, round(side * scale)) for side in image.size)
            image = image.resize(size, Image.Resampling.LANCZOS)
    return image
