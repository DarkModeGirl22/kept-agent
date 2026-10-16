"""Printable outputs. Kept fills them in; she checks them; a real person signs where a
signature is needed; she submits them herself. None of these are official state forms."""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import db
from .rules import engine as E

styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=4, spaceBefore=10)
BODY = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10, leading=13)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))

GRID = TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
    ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECEA")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
])


def _doc(buf, title):
    return SimpleDocTemplate(buf, pagesize=letter, title=title, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)


def _footer(rules):
    return Paragraph(f"Prepared with Kept on {date.today().strftime('%B %d, %Y')} · rules version "
                     f"{rules['version']}. Not an official state form. Check every line before you send it.", SMALL)


def reporting_log(conn, ctx, months: list[str]) -> bytes:
    """A log with the same columns as Montana's reporting log (Appendix A): individual,
    activity, calendar month, hours completed, verification provided / organization."""
    p, rules = ctx.person, ctx.rules
    buf = BytesIO()
    story = [Paragraph("Community engagement reporting log", H1),
             Paragraph(f"Name: <b>{p['name']}</b> &nbsp;&nbsp; State: {rules['state_name']} &nbsp;&nbsp; "
                       f"Months: {', '.join(E.month_label(m) for m in months)}", BODY),
             Paragraph("Use it to fill in or attach to your state's reporting form. "
                       "Same columns as Montana's reporting log (form HCS 494, Appendix A).", SMALL),
             Spacer(1, 8)]
    rows = [["Individual", "Activity", "Calendar month", "Hours completed", "Verification provided / org name"]]
    for m in months:
        s = ctx.summary(m)
        for (aid, name, typ), hrs in sorted(s.hours_by_activity.items(), key=lambda kv: kv[0][1]):
            proof = _proof_line(conn, ctx, aid, m, typ)
            rows.append([p["name"], f"{name} ({E.ACTIVITY_LABELS[typ].lower()})", E.month_label(m), f"{hrs:g}",
                         Paragraph(proof, SMALL)])
        rows.append(["", Paragraph("<b>Total</b>", BODY), E.month_label(m), f"{s.hours_total:g}",
                     Paragraph(f"Earned income: ${s.income_total:,.2f}. Status: {E.STATUS_TEXT[s.status]}"
                               + (f" (met by {', '.join(s.met_by)})" if s.met_by else ""), SMALL)])
    t = Table(rows, colWidths=[1.0 * inch, 1.9 * inch, 1.0 * inch, 0.8 * inch, 2.2 * inch], repeatRows=1)
    t.setStyle(GRID)
    story += [t, Spacer(1, 12), _footer(rules)]
    _doc(buf, "Reporting log").build(story)
    return buf.getvalue()


def _proof_line(conn, ctx, activity_id, month, typ) -> str:
    docs = conn.execute(
        "SELECT DISTINCT d.kind, d.original_name FROM entry e JOIN document d ON d.id=e.document_id "
        "WHERE e.person_id=? AND e.activity_id=? AND e.superseded_by IS NULL AND d.status='confirmed' "
        "AND substr(e.start_date,1,7)<=? AND substr(e.end_date,1,7)>=?",
        (ctx.person["id"], activity_id, month, month)).fetchall()
    so = conn.execute("SELECT status FROM signoff WHERE person_id=? AND activity_id=? AND month=?",
                      (ctx.person["id"], activity_id, month)).fetchone()
    act = ctx.activity(activity_id)
    parts = []
    if docs:
        parts.append("Attached: " + ", ".join({_kind_name(d['kind']) for d in docs}))
    if so:
        parts.append("Signed verification page" if so["status"] == "signed" else "Verification page prepared, not signed yet")
    if typ == "volunteer" and act and act.get("org_tax_id"):
        parts.append(f"{act['name']}, tax ID {act['org_tax_id']}")
    if not parts:
        parts.append("Kept daily log (no document yet)")
    return "; ".join(parts)


def _kind_name(k):
    return {"paystub": "pay stub", "gig": "app earnings statement", "volunteer": "signed volunteer sheet",
            "school": "school schedule"}.get(k, "document")


def volunteer_page(conn, ctx, act: dict, month: str, hours: float) -> bytes:
    """Pre-filled with what Kept knows. The organization's official checks and signs it."""
    p, rules = ctx.person, ctx.rules
    buf = BytesIO()
    rows = [
        ["Organization name", act["name"]],
        ["501(c)(3) or tax ID", act.get("org_tax_id") or ""],
        ["Address", act.get("org_address") or ""],
        ["Phone", act.get("org_phone") or ""],
        ["Volunteer", p["name"]],
        ["Calendar month", E.month_label(month)],
        ["Hours this month", f"{hours:g}"],
    ]
    t = Table(rows, colWidths=[2.0 * inch, 4.6 * inch], rowHeights=0.42 * inch)
    t.setStyle(TableStyle([("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10), ("FONT", (1, 0), (1, -1), "Helvetica", 11),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    sig = Table([["Authorizing official, printed name", ""], ["Signature", ""], ["Date", ""]],
                colWidths=[2.6 * inch, 4.0 * inch], rowHeights=0.55 * inch)
    sig.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 10),
                             ("LINEBELOW", (1, 0), (1, -1), 0.8, colors.black),
                             ("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
    entries = conn.execute(
        "SELECT start_date, hours FROM entry WHERE person_id=? AND activity_id=? AND superseded_by IS NULL "
        "AND substr(start_date,1,7)=? ORDER BY start_date", (p["id"], act["id"], month)).fetchall()
    days = ", ".join(f"{date.fromisoformat(e['start_date']).strftime('%b %d')} ({e['hours']:g}h)" for e in entries if e["hours"])
    story = [Paragraph("Community service hours: verification", H1),
             Paragraph(f"For {p['name']}'s Medicaid community engagement record ({rules['state_name']}).", BODY),
             Spacer(1, 10), t, Spacer(1, 6),
             Paragraph(f"Days logged by the volunteer: {days or 'none'}.", SMALL),
             Spacer(1, 18),
             Paragraph("To be completed by the organization. Please check the hours above. If they are right, "
                       "sign below. If not, correct them before signing.", BODY),
             Spacer(1, 8), sig, Spacer(1, 18), _footer(rules)]
    _doc(buf, "Volunteer verification").build(story)
    return buf.getvalue()


def proof_packet(conn, ctx, months: list[str], title: str = "Proof packet") -> bytes:
    """A cover summary, the reporting log, then every confirmed document for those months."""
    p = ctx.person
    buf = BytesIO()
    story = [Paragraph(title, H1),
             Paragraph(f"{p['name']}, {ctx.rules['state_name']}: {', '.join(E.month_label(m) for m in months)}", BODY),
             Spacer(1, 8)]
    rows = [["Month", "Hours", "Earned", "Status", "Proof"]]
    for m in months:
        s = ctx.summary(m)
        rows.append([E.month_label(m), f"{s.hours_total:g}", f"${s.income_total:,.2f}", E.STATUS_TEXT[s.status],
                     "complete" if s.proof_complete else f"{s.unproven_entries} item(s) without a document"])
    t = Table(rows, colWidths=[1.3 * inch, 0.8 * inch, 1.0 * inch, 1.0 * inch, 2.7 * inch])
    t.setStyle(GRID)
    story += [t, Spacer(1, 10), _footer(ctx.rules)]
    docs = conn.execute(
        "SELECT DISTINCT d.* FROM document d JOIN entry e ON e.document_id=d.id WHERE d.person_id=? AND d.status='confirmed' "
        "AND e.superseded_by IS NULL ORDER BY e.start_date", (p["id"],)).fetchall()
    for d in docs:
        ents = conn.execute("SELECT start_date,end_date FROM entry WHERE document_id=?", (d["id"],)).fetchall()
        if not any(m >= e["start_date"][:7] and m <= e["end_date"][:7] for e in ents for m in months):
            continue
        path = db.UPLOAD_DIR / d["filename"]
        if not path.exists() or path.suffix.lower() == ".pdf":
            continue
        story += [PageBreak(), Paragraph(f"{_kind_name(d['kind']).capitalize()}: {d['original_name']}", H2)]
        img = RLImage(str(path))
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(6.6 * inch / iw, 8.6 * inch / ih)
        img.drawWidth, img.drawHeight = iw * scale, ih * scale
        story.append(img)
    _doc(buf, title).build(story)
    return buf.getvalue()
