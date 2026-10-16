"""Make the fictional screen recording used in the demo (samples/sample_dashly_history.mp4):
Keisha scrolls through her Dashly earnings history on her phone. Needs ffmpeg."""
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H = 720, 1280
TEAL = (17, 120, 110)

# newest first, like the app; these match Keisha's texts (Oct 16 includes $18 of tips she forgot)
DAYS = [
    ("Fri, Oct 16", "5:00 PM - 8:30 PM", "3h 30m", 4, "64.00"),
    ("Tue, Oct 13", "10:00 AM - 2:00 PM", "4h 00m", 5, "50.00"),
    ("Fri, Oct 9", "6:00 PM - 9:30 PM", "3h 30m", 4, "44.00"),
    ("Tue, Oct 6", "11:00 AM - 3:00 PM", "4h 00m", 5, "52.00"),
    ("Fri, Oct 2", "5:00 PM - 8:00 PM", "3h 00m", 3, "38.00"),
    ("Tue, Sep 29", "11:00 AM - 2:00 PM", "3h 00m", 4, "41.00"),
    ("Fri, Sep 25", "5:30 PM - 9:00 PM", "3h 30m", 4, "47.00"),
    ("Tue, Sep 22", "10:00 AM - 1:30 PM", "3h 30m", 4, "45.00"),
]


def page():
    row_h = 210
    img = Image.new("RGB", (W, 260 + row_h * len(DAYS) + 200), (246, 247, 248))
    d = ImageDraw.Draw(img)
    f = lambda s, b=False: ImageFont.truetype(FB if b else F, s)
    d.rectangle([0, 0, W, 170], fill=TEAL)
    d.text((40, 60), "Dashly", font=f(54, True), fill="white")
    d.text((W - 300, 30), "SAMPLE - FICTIONAL", font=f(22, True), fill=(255, 210, 210))
    d.text((40, 195), "Earnings history", font=f(36, True), fill=(20, 20, 20))
    y = 260
    for day, shift, active, n, pay in DAYS:
        d.rounded_rectangle([24, y, W - 24, y + row_h - 20], radius=18, fill="white", outline=(225, 228, 230))
        d.text((50, y + 22), day, font=f(34, True), fill=(20, 20, 20))
        d.text((W - 60, y + 22), f"${pay}", font=f(34, True), fill=(20, 20, 20), anchor="ra")
        d.text((50, y + 78), f"Shift {shift}", font=f(26), fill=(80, 80, 80))
        d.text((50, y + 122), f"Active time {active}  ·  {n} deliveries", font=f(26), fill=(80, 80, 80))
        y += row_h
    return img


def main():
    ff = shutil.which("ffmpeg")
    full = page()
    tmp = Path(tempfile.mkdtemp())
    frames, fps = [], 30
    max_scroll = full.height - H
    # hold, scroll down slowly, hold
    positions = [0] * 30 + [int(max_scroll * (i / 210) ** 1) for i in range(211)] + [max_scroll] * 30
    for i, top in enumerate(positions):
        phone = Image.new("RGB", (W, H), "black")
        phone.paste(full.crop((0, top, W, top + H - 60)), (0, 60))
        ImageDraw.Draw(phone).text((30, 15), "9:41", font=ImageFont.truetype(FB, 26), fill="white")
        phone.save(tmp / f"{i:04d}.png")
    out = ROOT / "samples" / "sample_dashly_history.mp4"
    subprocess.run([ff, "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(out)], check=True)
    shutil.rmtree(tmp)
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
