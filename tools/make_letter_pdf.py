"""Draw the fictional renewal letter used in the demo video (samples/sample_renewal_letter.pdf)."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "samples" / "sample_renewal_letter.pdf"


def main():
    c = canvas.Canvas(str(OUT), pagesize=letter)
    c.setTitle("Sample renewal letter (fictional)")
    W, H = letter
    x = inch
    c.setFillColor(colors.HexColor("#b3261e"))
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(W - inch, H - 0.6 * inch, "SAMPLE - FICTIONAL")
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(x, H - 1.0 * inch, "STATE MEDICAID OFFICE (SAMPLE)")
    c.setFont("Helvetica", 10.5)
    c.drawString(x, H - 1.22 * inch, "Community Engagement Unit  -  PO Box 0000, Sample City")
    c.setLineWidth(0.8)
    c.line(x, H - 1.35 * inch, W - inch, H - 1.35 * inch)

    y = H - 1.75 * inch
    for line in ["October 19, 2026", "", "Keisha Moore", "1200 Sample Street, Apt 4", "Sample City",
                 "", "Case number: 000-SAMPLE-000"]:
        c.drawString(x, y, line)
        y -= 15
    y -= 12
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "YOUR MEDICAID RENEWAL")
    y -= 26
    c.setFont("Helvetica", 11)
    body = [
        "Dear Keisha Moore,",
        "",
        "Your Medicaid coverage is due for renewal on January 1, 2027.",
        "",
        "To keep your coverage, you must show 80 hours each month of work, community",
        "service, job training or school, or at least $580 of income, for these months:",
        "",
        "        October 2026",
        "        November 2026",
        "        December 2026",
        "",
        "Our records cannot confirm your hours from employer data. Please send proof of",
        "your hours or income for each month: pay stubs, earnings statements from apps,",
        "or a signed page from the organization where you volunteer.",
        "",
        "Please respond by December 31, 2026.",
        "",
        "You can send it online, by mail, or at your local office. If we do not receive",
        "proof, your coverage may stop. If you think you should not have to show hours,",
        "for example because of a health condition or caring for a child, tell us.",
        "",
        "Sincerely,",
        "Community Engagement Unit",
    ]
    for line in body:
        if line.startswith("Please respond"):
            c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, line)
        c.setFont("Helvetica", 11)
        y -= 16
    c.setFont("Helvetica-Oblique", 8.5)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(x, 0.7 * inch, "This is a fictional sample for a demo. It is not from any real government agency.")
    c.save()


if __name__ == "__main__":
    main()
