"""Draw the fictional sample documents used in the demo. Every one is marked SAMPLE and
uses invented names; none imitates a real company or agency.

Run once: python tools/make_samples.py  (the PNGs are already in samples/)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "samples"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def stamp(img):
    d = ImageDraw.Draw(img)
    d.text((img.width - 330, 18), "SAMPLE - FICTIONAL", fill=(200, 40, 40), font=font(26, True))


def gig_statement():
    img = Image.new("RGB", (900, 1500), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 900, 140], fill=(16, 120, 110))
    d.text((40, 45), "Dashly", fill="white", font=font(52, True))
    stamp(img)
    y = 190
    for txt, f in [("Weekly earnings summary", font(40, True)), ("Oct 5 - Oct 11, 2026", font(34)),
                   ("Driver: Keisha Moore", font(30))]:
        d.text((40, y), txt, fill=(20, 20, 20), font=f); y += 70
    y += 30
    rows = [("Deliveries", "14"), ("Active time", "7h 30m"), ("Delivery pay", "$71.40"),
            ("Customer tips", "$24.60"), ("Total earnings", "$96.00")]
    for k, v in rows:
        d.text((40, y), k, fill=(60, 60, 60), font=font(34))
        d.text((600, y), v, fill=(20, 20, 20), font=font(34, True))
        y += 80
    d.line([40, y, 860, y], fill=(200, 200, 200), width=3)
    y += 40
    d.text((40, y), "Mon Oct 6     4h 00m     $52.00", fill=(60, 60, 60), font=font(28)); y += 55
    d.text((40, y), "Thu Oct 9     3h 30m     $44.00", fill=(60, 60, 60), font=font(28))
    img.save(OUT / "sample_gig_statement.png")


def pay_stub():
    img = Image.new("RGB", (1240, 900), "white")
    d = ImageDraw.Draw(img)
    stamp(img)
    d.text((50, 50), "CORNER MARKET", fill=(20, 20, 20), font=font(48, True))
    d.text((50, 115), "Earnings statement", fill=(60, 60, 60), font=font(32))
    d.text((50, 190), "Employee: Keisha Moore", fill=(20, 20, 20), font=font(30))
    d.text((50, 240), "Pay period: 10/19/2026 - 10/31/2026", fill=(20, 20, 20), font=font(30))
    d.text((50, 290), "Pay date: 11/03/2026", fill=(20, 20, 20), font=font(30))
    y = 380
    d.line([50, y, 1190, y], fill=(0, 0, 0), width=2); y += 25
    for k, v in [("Total hours", "4.00"), ("Rate", "$12.00"), ("Gross pay", "$48.00"), ("Net pay", "$44.33")]:
        d.text((50, y), k, fill=(20, 20, 20), font=font(32))
        d.text((700, y), v, fill=(20, 20, 20), font=font(32, True))
        y += 70
    img.save(OUT / "sample_pay_stub.png")


def volunteer_sheet():
    img = Image.new("RGB", (1240, 1000), "white")
    d = ImageDraw.Draw(img)
    stamp(img)
    d.text((50, 60), "Riverside Community Pantry", fill=(20, 20, 20), font=font(44, True))
    d.text((50, 125), "Volunteer service hours", fill=(60, 60, 60), font=font(32))
    d.text((50, 200), "Volunteer: Keisha Moore", fill=(20, 20, 20), font=font(30))
    d.text((50, 250), "Period: Oct 1, 2026 - Oct 31, 2026", fill=(20, 20, 20), font=font(30))
    y = 330
    for day in ["Sat Oct 3", "Sat Oct 10", "Sat Oct 17", "Sat Oct 24", "Sat Oct 31"]:
        d.text((50, y), f"{day}      4.0 hours", fill=(20, 20, 20), font=font(30)); y += 55
    d.text((50, y + 20), "Hours served: 20.0", fill=(20, 20, 20), font=font(34, True))
    d.text((50, y + 110), "Coordinator signature: D. Brooks", fill=(20, 20, 20), font=font(30))
    img.save(OUT / "sample_volunteer_sheet.png")


def state_letter():
    img = Image.new("RGB", (1240, 1600), "white")
    d = ImageDraw.Draw(img)
    stamp(img)
    d.text((60, 70), "STATE MEDICAID OFFICE (SAMPLE)", fill=(20, 20, 20), font=font(40, True))
    d.text((60, 125), "Community Engagement Unit", fill=(60, 60, 60), font=font(30))
    lines = [
        "November 23, 2026", "", "Keisha Moore", "Case number: 000-SAMPLE-000", "",
        "REQUEST FOR VERIFICATION", "",
        "Dear Keisha Moore,", "",
        "We are reviewing your Medicaid coverage. We were unable to verify",
        "that you completed 80 hours of qualifying activities, or earned",
        "$580, in the following months:", "",
        "    October 2026", "    November 2026", "",
        "Please send proof of your work, community service or school hours",
        "for these months. Please respond by December 23, 2026.", "",
        "If we do not receive proof, your coverage may end.", "",
        "This is a fictional sample for a demo.",
    ]
    y = 210
    for ln in lines:
        d.text((60, y), ln, fill=(20, 20, 20), font=font(30, ln.isupper() and len(ln) > 5)); y += 50
    img.save(OUT / "sample_state_letter.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    gig_statement(); pay_stub(); volunteer_sheet(); state_letter()
    print("samples written to", OUT)
