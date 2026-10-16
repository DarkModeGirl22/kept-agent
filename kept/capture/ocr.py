"""Opening phone photos. The reading itself is done by NVIDIA Nemotron 3.5 Super VL
(see vision.py); this module only fixes rotation and size before the photo is stored."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

try:  # iPhone photos are HEIC; this optional package lets Pillow open them.
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
except Exception:  # pragma: no cover - optional
    pass


def normalize_image(src: Path, dest: Path, max_side: int = 1600) -> Path:
    """Open any phone photo (JPEG, PNG, HEIC), fix its rotation, shrink it, save as JPEG."""
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    img.save(dest, "JPEG", quality=88)
    return dest


def available() -> bool:
    from . import llm
    return llm.vision_available()
