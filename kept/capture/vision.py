"""Reading photos and app-history videos with NVIDIA Nemotron 3.5 Super VL.

The vision model looks at the picture the way a person would: crumpled pay stubs, app
screenshots, handwritten notes from a boss, sign-in sheets, letters. It returns two things:
every line of text it sees, and the values it thinks matter. Kept's rules then re-read
those lines on their own and compare. A value is trusted only when it is written in what
the model read; when the model and the rules disagree, Kept asks the person.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

from . import llm

PHOTO_PROMPT = (
    "You are reading a photo that a low-income worker in the United States took as proof of work "
    "for Medicaid. It may be a pay stub, a gig-app earnings screen, a handwritten note from an "
    "employer, a volunteer sign-in sheet, a school schedule, or a letter from the state.\n"
    "1. Transcribe every line of text exactly as written, top to bottom, including handwriting. "
    "Keep each label and its value on the same line.\n"
    "2. Say what kind of document it is: paystub, gig, cash_note, volunteer, school, letter or other.\n"
    "3. Extract only values that are written: period_start and period_end as YYYY-MM-DD (the dates "
    "the work covers; one day means both are the same), hours (total hours worked or served; "
    "convert 3h 30m to 3.5), income (gross or total pay before deductions, tips included), "
    "source_name (employer, app or organization), person_name.\n"
    "Use null for anything that is not written. Never estimate. Today is {today}.\n"
    'Reply with JSON only: {{"lines": [..], "kind": "..", "period_start": .., "period_end": .., '
    '"hours": .., "income": .., "source_name": .., "person_name": ..}}'
)

KIND_MAP = {"cash_note": "other", "paystub": "paystub", "gig": "gig", "volunteer": "volunteer",
            "school": "school", "letter": "letter", "other": "other"}


def read_photo(path: Path, today: date) -> dict | None:
    """One call to Super VL. Returns {"text", "kind", "ai": {fields}} or None."""
    out = llm.vision_json(PHOTO_PROMPT.format(today=today.isoformat()), [path])
    if not out:
        return None
    lines = out.get("lines")
    if not isinstance(lines, list):
        return None
    text = "\n".join(str(x) for x in lines if str(x).strip())
    if len(text.strip()) < 3:
        return None
    ai = {k: out.get(k) for k in ("period_start", "period_end", "hours", "income", "source_name", "person_name")}
    kind = KIND_MAP.get(str(out.get("kind") or "other").lower(), "other")
    return {"text": text, "kind": kind, "ai": ai, "handwritten_note": str(out.get("kind")) == "cash_note"}


# ------------------------------------------------------------------ app-history videos

HISTORY_PROMPT = (
    "These are frames from a screen recording. A delivery or rideshare worker scrolled through the "
    "earnings history in her app. List every day (or every week, if only weeks are shown) that is "
    "visible in any frame, once each, even if it appears in several frames.\n"
    "For each, give: date (YYYY-MM-DD; for a week, its first day), end_date (same as date for one "
    "day), hours (time worked or active/online time, as a number; 2h 15m = 2.25; null if not shown), "
    "pay (total earnings for that row, tips included; null if not shown), and quote: the exact text "
    "of that row as shown on screen.\n"
    "Also give app_name, the app's name as shown. Only include rows you can actually read. Today is "
    "{today}; the year is {year} unless the screen says otherwise.\n"
    'Reply with JSON only: {{"app_name": .., "rows": [{{"date": .., "end_date": .., "hours": .., '
    '"pay": .., "quote": ".."}}]}}'
)


def _num(v):
    try:
        f = float(str(v).replace("$", "").replace(",", ""))
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _in_quote(value: float, quote: str, hours: bool) -> bool:
    """Is this number written in the row the model quoted? 3.5 hours may be written 3h 30m."""
    q = quote.replace(",", "")
    cands = {f"{value:.2f}", f"{value:g}"}
    if hours:
        h, m = int(value), round((value - int(value)) * 60)
        cands |= {f"{h}h {m}m", f"{h}h {m:02d}m", f"{h} hr {m} min", f"{h}:{m:02d}", f"{h}h{m:02d}"}
        if m == 0:
            cands |= {f"{h}h", f"{h} hr", f"{h} hours"}
    return any(c in q for c in cands)


MAX_IMAGES = 8   # Token Factory accepts up to 8 images per request


def read_history(frames: list, today: date) -> dict | None:
    """Super VL reads the frames, 8 per call; Kept's rules then check every row."""
    raw_rows, app = [], ""
    for i in range(0, len(frames), MAX_IMAGES):
        out = llm.vision_json(HISTORY_PROMPT.format(today=today.isoformat(), year=today.year),
                              frames[i:i + MAX_IMAGES], max_tokens=3000, timeout=90)
        if out and isinstance(out.get("rows"), list):
            raw_rows += out["rows"]
            app = app or str(out.get("app_name") or "")
    if not raw_rows:
        return None
    rows, seen = [], set()
    for r in raw_rows:
        if not isinstance(r, dict):
            continue
        try:
            d = date.fromisoformat(str(r.get("date")))
            e = date.fromisoformat(str(r.get("end_date") or r.get("date")))
        except ValueError:
            continue
        quote = str(r.get("quote") or "")
        hours, pay = _num(r.get("hours")), _num(r.get("pay"))
        problems = []
        # rules: dates in the past and recent, one row per period, plausible hours and pay
        if e < d:
            d, e = e, d
        if e > today:
            problems.append("date in the future")
        if d < today - timedelta(days=120):
            problems.append("more than 4 months ago")
        if (e - d).days > 7:
            problems.append("period longer than a week")
        if hours is not None and hours > 16 * ((e - d).days + 1):
            problems.append("too many hours for the dates")
        if pay is not None and pay > 600 * ((e - d).days + 1):
            problems.append("pay looks too high")
        if hours is None and pay is None:
            problems.append("no hours or pay")
        grounded = (hours is None or _in_quote(hours, quote, True)) and (pay is None or _in_quote(pay, quote, False))
        if quote and not grounded:
            problems.append("numbers not in the row it quoted")
        key = (d, e)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"start": d.isoformat(), "end": e.isoformat(), "hours": hours, "pay": pay, "quote": quote[:120],
                     "ok": not problems, "problems": problems})
    rows.sort(key=lambda x: x["start"])
    return {"app_name": app.strip()[:40], "rows": rows}


def sample_frames(video: Path, out_dir: Path, max_frames: int = 16) -> list:
    """Take still pictures from a screen recording, keeping only frames that changed.
    Uses ffmpeg (installed in the container); returns [] if it isn't available."""
    import shutil
    import subprocess
    from PIL import Image

    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg  # type: ignore
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return []
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([ff, "-v", "error", "-y", "-i", str(video), "-t", "180", "-vf", "fps=2,scale=720:-2",
                        str(out_dir / "f%03d.jpg")], check=True, timeout=120)
    except Exception:
        return []
    files = sorted(out_dir.glob("f*.jpg"))

    def sig(p):
        im = Image.open(p).convert("L").resize((24, 40))
        return list(im.getdata())

    kept, last = [], None
    for p in files:
        s = sig(p)
        if last is None or sum(abs(a - b) for a, b in zip(s, last)) / len(s) > 6:
            kept.append(p)
            last = s
    if len(kept) > max_frames:   # spread evenly over the recording
        step = len(kept) / max_frames
        kept = [kept[int(i * step)] for i in range(max_frames)]
    return kept


def grounded_value(v, text: str) -> bool:
    plain = text.replace(",", "")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    cands = {f"{f:.2f}", f"{f:g}", str(int(f)) if f.is_integer() else f"{f:.1f}"}
    h, m = int(f), round((f - int(f)) * 60)
    cands |= {f"{h}h {m}m", f"{h}h {m:02d}m", f"{h}:{m:02d}"}
    return any(re.search(rf"(?<![\d.]){re.escape(c)}(?![\d])", plain) for c in cands)
