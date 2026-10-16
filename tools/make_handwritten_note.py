"""Draw the fictional handwritten note from Keisha's corner-store manager
(samples/proof_corner_note_oct8.jpg): blue pen on lined paper, photographed at an angle."""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
random.seed(7)
W, H = 1100, 1400
FONT = "/usr/share/fonts/truetype/google-fonts/Lora-Italic-Variable.ttf"

paper = Image.new("RGB", (W, H), (250, 247, 236))
d = ImageDraw.Draw(paper)
for y in range(170, H - 60, 70):
    d.line([(40, y), (W - 40, y)], fill=(170, 200, 230), width=2)
d.line([(120, 0), (120, H)], fill=(230, 150, 150), width=3)

ink = (28, 48, 140)
lines = [
    "Oct 8, 2026",
    "",
    "To whom it may concern,",
    "Keisha Moore worked at",
    "Corner Market on Thursday",
    "Oct 8 from 12 to 3 pm.",
    "3 hours. Paid $36 cash.",
    "",
    "Ray Alvarez, manager",
    "(406) 555-0142",
]
y = 115
for text in lines:
    x = 150 + random.randint(-8, 8)
    for word in text.split(" "):
        size = random.randint(50, 56)
        f = ImageFont.truetype(FONT, size)
        try:
            f.set_variation_by_axes([random.randint(450, 600)])
        except Exception:
            pass
        bbox = d.textbbox((0, 0), word, font=f)
        w, h = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 40
        tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((10, 5), word, font=f, fill=ink + (235,))
        tile = tile.rotate(random.uniform(-4, 4), resample=Image.BICUBIC, expand=True)
        paper.paste(tile, (x, y + random.randint(-5, 5)), tile)
        x += w - 6 + random.randint(0, 10)
    y += 70

ImageDraw.Draw(paper).text((W - 330, 40), "SAMPLE - FICTIONAL", fill=(190, 40, 40),
                            font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26))

# photographed on a table: tilt, perspective, uneven light, a little blur
bg = Image.new("RGB", (W + 300, H + 300), (92, 74, 58))
paper = paper.rotate(-4, resample=Image.BICUBIC, expand=True, fillcolor=(92, 74, 58))
bg.paste(paper, (120, 110))
coeffs = (1.02, 0.03, -30, 0.01, 1.0, -10, 0.00004, 0.00002)
bg = bg.transform(bg.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=(92, 74, 58))
shade = Image.linear_gradient("L").resize(bg.size).rotate(35)
bg = Image.composite(bg, Image.eval(bg, lambda v: int(v * 0.8)), shade)
bg = bg.filter(ImageFilter.GaussianBlur(0.8)).resize(((W + 300) * 3 // 4, (H + 300) * 3 // 4))
bg.save(ROOT / "samples" / "proof_corner_note_oct8.jpg", quality=85)
print("ok")
