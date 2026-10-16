"""From the text read in a photo to clean fields (dates, hours, pay), each with a
confidence. Rules first; the local AI may fill gaps, but any value it gives must be
found in the document's own text, or it is thrown away."""

from __future__ import annotations

import re
from datetime import date

from . import llm
from .textlog import MONTHS, match_activity

ASK_BELOW = 0.75  # below this, Kept asks her instead of logging the value
LABELED = 0.9
UNLABELED = 0.5
AI_GROUNDED = 0.8

KIND_WORDS = {
    "letter": ["medicaid", "notice", "department", "respond", "verification", "dear", "case number",
               "community engagement", "eligibility"],
    "paystub": ["gross pay", "net pay", "pay period", "earnings statement", "ytd", "deductions", "pay date",
                "pay stub", "paystub"],
    "gig": ["deliveries", "trips", "active time", "online time", "weekly earnings", "tips", "customer tips",
            "earnings summary", "dashes", "orders"],
    "volunteer": ["volunteer", "community service", "sign-in", "sign in", "hours served", "service hours"],
    "school": ["schedule", "credits", "credit hours", "enrollment", "semester", "registrar", "course"],
}

_MON = "|".join(m[:3] + r"[a-z]*" for m in MONTHS)
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"), "mdy"),
    (re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), "Mdy"),
    (re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!,?\s*\d{{4}})", re.I), "Md"),
]


def classify(text: str) -> str:
    t = text.lower()
    scores = {k: sum(1 for w in words if w in t) for k, words in KIND_WORDS.items()}
    kind, best = max(scores.items(), key=lambda kv: kv[1])
    return kind if best > 0 else "other"


def find_dates(text: str, default_year: int) -> list[tuple[date, int]]:
    """All dates in the text with their position. A date without a year (\"Oct 5 – Oct 11,
    2026\") borrows the year of the next dated value."""
    found: list[tuple[date | None, int, tuple]] = []
    taken: list[range] = []
    for rx, kind in DATE_PATTERNS:
        for m in rx.finditer(text):
            if any(m.start() in r for r in taken):
                continue
            taken.append(range(m.start(), m.end()))
            g = m.groups()
            try:
                if kind == "ymd":
                    d = date(int(g[0]), int(g[1]), int(g[2]))
                elif kind == "mdy":
                    y = int(g[2])
                    d = date(y + 2000 if y < 100 else y, int(g[0]), int(g[1]))
                elif kind == "Mdy":
                    d = date(int(g[2]), [x[:3] for x in MONTHS].index(g[0][:3].lower()) + 1, int(g[1]))
                else:
                    found.append((None, m.start(), ([x[:3] for x in MONTHS].index(g[0][:3].lower()) + 1, int(g[1]))))
                    continue
                found.append((d, m.start(), ()))
            except ValueError:
                continue
    found.sort(key=lambda x: x[1])
    out: list[tuple[date, int]] = []
    for i, (d, pos, md) in enumerate(found):
        if d is None:
            year = next((x[0].year for x in found[i + 1:] if x[0] is not None), default_year)
            try:
                d = date(year, md[0], md[1])
            except ValueError:
                continue
        out.append((d, pos))
    return out


def _money(s: str) -> float:
    return float(s.replace(",", "").replace("$", "").strip())


def _amount_after(label_rx: str, text: str):
    m = re.search(label_rx + r"[^\n$\d]{0,25}\$?\s*(\d[\d,]*\.\d{2})", text, re.I)
    return _money(m.group(1)) if m else None


def _hours_after(label_rx: str, text: str):
    m = re.search(label_rx + r"[^\n\d]{0,20}(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours)?\s*(?:(\d{1,2})\s*(?:m|min|mins|minutes))?", text, re.I)
    if not m:
        return None
    h = float(m.group(1))
    if m.group(2):
        h += int(m.group(2)) / 60
    return round(h, 2)


def rule_fields(kind: str, text: str, today: date) -> tuple[dict, dict]:
    fields, conf = {}, {}

    def put(k, v, c):
        if v is not None and k not in fields:
            fields[k], conf[k] = v, c

    dates = find_dates(text, today.year)
    # period: a labeled line first, then the first two dates joined by a dash
    pm = re.search(r"(pay\s*period|period|week of|dates?)[^\n]*", text, re.I)
    if pm:
        in_line = find_dates(pm.group(0), today.year)
        if len(in_line) >= 2:
            put("period_start", in_line[0][0], LABELED)
            put("period_end", in_line[1][0], LABELED)
    if "period_start" not in fields:
        # a written range on one line: "Oct 5 - Oct 11, 2026" or "10/19/2026 - 10/31/2026"
        for line in text.splitlines():
            in_line = find_dates(line, today.year)
            if len(in_line) >= 2 and re.search(r"\d\s*(?:-|–|—|to|through)\s*[A-Za-z]{0,9}\.?\s*\d", line):
                a, b = in_line[0][0], in_line[1][0]
                if a <= b and (b - a).days <= 62:
                    put("period_start", a, 0.85)
                    put("period_end", b, 0.85)
                    break
    if "period_start" not in fields and len(dates) >= 2:
        a, b = dates[0][0], dates[1][0]
        if a <= b and (b - a).days <= 31:
            put("period_start", a, UNLABELED)
            put("period_end", b, UNLABELED)
    if "period_start" not in fields and len(dates) == 1:
        # one date for the whole document; a "Date:" label on its line makes it clear
        line = next((ln for ln in text.splitlines() if find_dates(ln, today.year)), "")
        c = LABELED if re.match(r"\s*(date|day|shift date)\b", line, re.I) or len(text.splitlines()) < 40 and \
            re.search(r"\b(date|summary|record|sign-in)\b", text, re.I) else UNLABELED
        put("period_start", dates[0][0], c)
        put("period_end", dates[0][0], c)

    hour_labels = [r"total\s+hours", r"hours\s+worked", r"active\s+time", r"online\s+time", r"time\s+worked",
                   r"hours\s+served", r"regular\s+hours"]
    for lab in hour_labels:
        v = _hours_after(lab, text)
        if v is not None and 0 < v <= 400:
            put("hours", v, LABELED)
            break
    if "hours" not in fields:
        # written the other way round, as in a note: "3 hours", "4.5 hrs"
        m = re.search(r"(?<![\d.$])(\d+(?:\.\d+)?)\s*(?:hours|hrs|hr)\b", text, re.I)
        if m and 0 < float(m.group(1)) <= 400:
            put("hours", float(m.group(1)), LABELED)
    if "hours" not in fields:
        v = _hours_after(r"\bhours\b", text)
        if v is not None and 0 < v <= 400:
            put("hours", v, UNLABELED)

    money_labels = [r"gross\s+pay", r"gross\s+earnings", r"total\s+earnings", r"total\s+pay", r"earnings",
                    r"\btotal\b"]
    if kind in ("paystub", "gig", "other"):
        for lab in money_labels:
            v = _amount_after(lab, text)
            if v is not None:
                put("income", v, LABELED)
                break
        if "income" not in fields:
            amounts = [_money(x) for x in re.findall(r"\$\s*(\d[\d,]*\.\d{2})", text)]
            if amounts:
                put("income", max(amounts), UNLABELED)

    first_line = next((ln.strip() for ln in text.splitlines()
                       if len(ln.strip()) > 2 and "sample" not in ln.lower()), "")
    put("source_name", first_line[:60] or None, UNLABELED)
    return fields, conf


AI_SYSTEM = (
    "You read the text of a {kind} that a worker photographed. Reply with JSON only: "
    '{{"period_start": "YYYY-MM-DD or null", "period_end": "YYYY-MM-DD or null", '
    '"hours": number or null, "income": number or null, "source_name": "employer, app or organization or null"}}. '
    "income is gross pay or total earnings before deductions. Copy values exactly as written. "
    "Use null for anything not written in the text. Never estimate."
)


def _grounded_number(v: float, text: str) -> bool:
    plain = text.replace(",", "")
    cands = {f"{v:.2f}", f"{v:g}", str(int(v)) if float(v).is_integer() else f"{v:.1f}"}
    return any(re.search(rf"(?<![\d.]){re.escape(c)}(?![\d])", plain) for c in cands)


VERIFIED = 0.97   # the vision model and the rules read the same value


def _merge_vision(ai: dict, fields: dict, conf: dict, wanted: list, text: str, today: date) -> bool:
    """Nemotron Super VL read the photo; Kept's rules re-read its transcription. Agreeing
    values are trusted, values only the model gave must be written in the transcription,
    disagreements are left for her to settle."""
    used = False
    dates_in_text = {d for d, _ in find_dates(text, today.year)}
    for k in wanted:
        v = ai.get(k)
        if v in (None, "", "null"):
            continue
        try:
            if k in ("period_start", "period_end"):
                v = date.fromisoformat(str(v))
                grounded = v in dates_in_text
            else:
                v = float(v)
                grounded = v > 0 and (_grounded_number(v, text) or k == "hours" and _hours_written(v, text))
        except (TypeError, ValueError):
            continue
        if k in fields:
            same = fields[k] == v if isinstance(v, date) else abs(float(fields[k]) - v) < 0.01
            if same:
                conf[k], used = VERIFIED, True
            elif grounded:
                # both are written in the document: the model read the right row, the rules a nearby one
                fields[k], conf[k], used = v, 0.6, True
        elif grounded:
            fields[k], conf[k], used = v, AI_GROUNDED + 0.05, True
    return used


def _hours_written(v: float, text: str) -> bool:
    h, m = int(v), round((v - int(v)) * 60)
    return any(c in text for c in (f"{h}h {m}m", f"{h}h {m:02d}m", f"{h} hr {m} min", f"{h}:{m:02d}"))


def extract(text: str, today: date, activities: list[dict], ai: dict | None = None,
            ai_kind: str | None = None) -> dict:
    kind = classify(text)
    if ai_kind and ai_kind != "other" and (kind == "other" or ai_kind == "letter"):
        kind = ai_kind
    fields, conf = rule_fields(kind, text, today)
    read_by = "rules"

    wanted = ["period_start", "period_end", "hours", "income"]
    if kind in ("volunteer", "school"):
        wanted = ["period_start", "period_end", "hours"]
    if ai:
        if _merge_vision(ai, fields, conf, wanted, text, today):
            read_by = "ai"
    gaps = [k for k in wanted if conf.get(k, 0) < ASK_BELOW]
    if not ai and gaps and kind != "letter" and llm.available():
        out = llm.chat_json(AI_SYSTEM.format(kind=kind), text[:4000])
        if out:
            dates_in_text = {d for d, _ in find_dates(text, today.year)}
            used = False
            for k in gaps:
                v = out.get(k)
                if v in (None, "", "null"):
                    continue
                try:
                    if k in ("period_start", "period_end"):
                        dv = date.fromisoformat(str(v))
                        if dv in dates_in_text:
                            fields[k], conf[k], used = dv, AI_GROUNDED, True
                    else:
                        fv = float(v)
                        if fv > 0 and _grounded_number(fv, text):
                            fields[k], conf[k], used = fv, AI_GROUNDED, True
                except (TypeError, ValueError):
                    continue
            if used:
                read_by = "ai"

    act, ac = match_activity(text, activities)
    if act:
        fields["activity_id"], conf["activity_id"] = act["id"], ac

    # sanity checks: no more than 24h per day in the period, end not before start
    if fields.get("period_start") and fields.get("period_end"):
        if fields["period_end"] < fields["period_start"]:
            fields["period_start"], fields["period_end"] = fields["period_end"], fields["period_start"]
        days = (fields["period_end"] - fields["period_start"]).days + 1
        if fields.get("hours") and fields["hours"] > 24 * days:
            conf["hours"] = 0.3
    if fields.get("period_end") and fields["period_end"] > today:
        conf["period_end"] = min(conf.get("period_end", 0), 0.5)

    ask = [k for k in wanted if conf.get(k, 0) < ASK_BELOW]
    if "activity_id" not in fields or conf.get("activity_id", 0) < ASK_BELOW:
        ask.append("activity_id")
    return {"kind": kind, "fields": fields, "confidence": conf, "read_by": read_by, "ask": ask}
