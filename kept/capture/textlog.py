"""Turn one line like "4h food bank today" or "Dashly 3pm-8pm yesterday, made $85" into a
draft log entry. Rules first; the local AI only fills what the rules could not; she
confirms every draft before it counts."""

from __future__ import annotations

import re
from datetime import date, timedelta

from . import llm

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAYS_ES = ["lunes", "martes", "mi[eé]rcoles", "jueves", "viernes", "s[aá]bado", "domingo"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

TYPE_WORDS = {
    "gig": ["deliver", "delivery", "deliveries", "dash", "rides", "rideshare", "driving", "driver", "gig", "orders",
            "entregas", "reparto", "pedidos"],
    "volunteer": ["volunteer", "volunteered", "pantry", "food bank", "church", "shelter",
                  "voluntari", "banco de comida", "despensa", "iglesia", "albergue"],
    "education": ["class", "school", "college", "course", "studied", "lecture", "clase", "escuela", "curso"],
    "job_training": ["training", "workshop", "apprentice"],
    "cash": ["cash"],
    "work": ["shift", "work", "worked", "store", "job"],
}

WHEN_SAID = re.compile(r"\b(ago|before|week|day|days|night|morning|afternoon|evening|weekend|hace|antes|anteayer|"
                       r"antier|semana|d[ií]as?|noche|ma[nñ]ana|tarde|fin de semana)\b|\d{1,2}(st|nd|rd|th)\b", re.I)

CONFIDENT = 0.9
GUESS = 0.6


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_date(text: str, today: date) -> tuple[date | None, float]:
    t = text.lower()
    if re.search(r"\b(yesterday|last night|ayer|anoche)\b", t):
        return today - timedelta(days=1), CONFIDENT
    if re.search(r"\b(today|tonight|this morning|this afternoon|hoy|esta ma[nñ]ana|esta tarde|esta noche)\b", t):
        return today, CONFIDENT
    for i, (w, es) in enumerate(zip(WEEKDAYS, WEEKDAYS_ES)):
        if re.search(rf"\b{w}\b|\b{es}\b", t):
            back = (today.weekday() - i) % 7
            if re.search(rf"\blast\s+{w}\b|\b{es}\s+pasado\b|\bel\s+pasado\s+{es}\b", t) and back == 0:
                back = 7
            return today - timedelta(days=back), CONFIDENT
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        mo, da = int(m.group(1)), int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else today.year
        yr = yr + 2000 if yr < 100 else yr
        try:
            d = date(yr, mo, da)
            if d > today and not m.group(3):
                d = date(yr - 1, mo, da)
            return d, CONFIDENT
        except ValueError:
            pass
    month_re = "|".join(m[:3] + r"[a-z]*" for m in MONTHS)
    m = re.search(rf"\b({month_re})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t) or \
        re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_re})\b", t)
    if m:
        a, b = m.group(1), m.group(2)
        mon_word, day_s = (a, b) if a[0].isalpha() else (b, a)
        mo = [x[:3] for x in MONTHS].index(mon_word[:3]) + 1
        try:
            d = date(today.year, mo, int(day_s))
            if d > today:
                d = date(today.year - 1, mo, int(day_s))
            return d, CONFIDENT
        except ValueError:
            pass
    return None, 0.0


def _to_24(h: int, mins: int, ampm: str | None) -> float:
    if ampm == "pm" and h != 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return h + mins / 60


def parse_hours(text: str) -> tuple[float | None, float]:
    t = text.lower().replace("–", "-").replace("—", "-")
    # time ranges: 3pm-8pm, 3-8pm, 9:30am to 1pm, 15:00-20:00, from 3 to 8
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:-|to|until|till)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
    if m and (m.group(3) or m.group(6) or m.group(2) or re.search(r"\bfrom\b", t)):
        h1, m1, ap1 = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        h2, m2, ap2 = int(m.group(4)), int(m.group(5) or 0), m.group(6)
        if ap2 and not ap1:
            # "3-8pm" and "12-5pm" share the end's am/pm; "11-3pm" starts in the morning
            ap1 = ap2 if _to_24(h1, m1, ap2) < _to_24(h2, m2, ap2) else ("am" if ap2 == "pm" else "pm")
        start, end = _to_24(h1, m1, ap1), _to_24(h2, m2, ap2)
        if not ap1 and not ap2 and end <= start:
            end += 12
        if end <= start:
            end += 24  # overnight shift
        hrs = round(end - start, 2)
        if 0 < hrs <= 16:
            return hrs, CONFIDENT
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?\s*(?:and\s*)?(\d{1,2})\s*m(?:in(?:utes?)?)?\b", t)
    if m:
        return round(_num(m.group(1)) + int(m.group(2)) / 60, 2), CONFIDENT
    m = re.search(r"\b(\d{1,2})h(\d{2})\b", t)  # 2h30
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 60, 2), CONFIDENT
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", t)
    if m:
        v = _num(m.group(1))
        if 0 < v <= 24:
            return v, CONFIDENT
    m = re.search(r"\b(\d+)\s*(?:min|mins|minutes)\b", t)
    if m:
        return round(int(m.group(1)) / 60, 2), CONFIDENT
    if re.search(r"\bhalf an hour\b", t):
        return 0.5, CONFIDENT
    return None, 0.0


def parse_money(text: str) -> tuple[float | None, float]:
    t = text.lower()
    m = re.search(r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)", t) or re.search(r"\b(\d[\d,]*(?:\.\d{1,2})?)\s*(?:dollars|usd|bucks)\b", t)
    if m:
        return _num(m.group(1)), CONFIDENT
    m = re.search(r"\b(?:made|earned|got paid|paid|tips?)\s*(\d[\d,]*(?:\.\d{1,2})?)\b", t)
    if m:
        return _num(m.group(1)), GUESS
    return None, 0.0


def match_activity(text: str, activities: list[dict]) -> tuple[dict | None, float]:
    t = text.lower()
    best, score, by_name, hits = None, 0, set(), set()
    for ai, a in enumerate(activities):
        names = [a["name"].lower()] + [k.strip().lower() for k in (a.get("keywords") or "").split(",") if k.strip()]
        for i, n in enumerate(names):
            if n and n in t:
                s = len(n) + 10
                hits.add(ai)
                if i == 0:
                    by_name.add(ai)
            else:
                words = [w for w in re.findall(r"[a-z]{4,}", n)]
                s = sum(len(w) for w in words if re.search(rf"\b{re.escape(w)}\b", t))
            if s > score:
                best, score = a, s
    if len(by_name) == 1:
        return activities[by_name.pop()], CONFIDENT
    if len(hits) > 1:
        return None, 0.0   # two of her activities fit ("grocery deliveries"): the AI, then she, decides
    if best:
        return best, (CONFIDENT if score >= 10 else GUESS)
    for typ, words in TYPE_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}", t) for w in words):
            of_type = [a for a in activities if a["type"] == typ]
            if len(of_type) == 1:
                return of_type[0], GUESS
    return None, 0.0


AI_SYSTEM = (
    "You turn one short message from a worker into a JSON log entry. The message may be typed or "
    "a voice note written down by her phone, in English or Spanish, with numbers in words. "
    "Today is {today} ({weekday}). Her activities are: {acts}. Reply with JSON only: "
    '{{"date": "YYYY-MM-DD or null", "hours": number or null, "income": number or null, '
    '"activity_id": number or null}}. "Yesterday"/"ayer" and weekday names mean the most recent such day. '
    "hours is the time worked; a range like 5 to 8 is 3 hours. Use null when the message does not say it. "
    "Never guess hours or money that are not in the message."
)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
    "nueve": 9, "diez": 10, "once": 11, "doce": 12, "veinte": 20, "treinta": 30, "cuarenta": 40,
    "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
}


def _money_said(v: float, text: str) -> bool:
    """Was this amount actually said? As digits ("38"), or as words ("thirty eight", "treinta y ocho")."""
    if re.search(rf"(?<![\d]){int(v)}(?![\d])", text):
        return True
    words = re.findall(r"[a-záéíóúñ]+", text.lower())
    total, cur, best = 0, 0, 0
    for w in words + ["."]:
        if w in NUMBER_WORDS:
            n = NUMBER_WORDS[w]
            cur = cur * n if n == 100 and cur else cur + n
            total = cur
        elif w in ("and", "y") and cur:
            continue
        else:
            best = total if total == int(v) else best
            total = cur = 0
    return best == int(v)


def parse(text: str, today: date, activities: list[dict]) -> dict:
    """Returns a draft: values, a confidence per field, which path read it, and which
    fields still need her answer."""
    d, dc = parse_date(text, today)
    h, hc = parse_hours(text)
    inc, ic = parse_money(text)
    act, ac = match_activity(text, activities)
    read_by = "rules"

    need_ai = d is None or act is None or (h is None and inc is None)
    if need_ai and llm.available():
        acts = "; ".join(f'{a["id"]}: {a["name"]} ({a["type"]})' for a in activities)
        out = llm.chat_json(AI_SYSTEM.format(today=today.isoformat(), weekday=today.strftime("%A"), acts=acts), text)
        if out:
            used = False
            try:
                # a date only if she said when: otherwise it's today and she confirms (never a guessed day)
                if d is None and out.get("date") and WHEN_SAID.search(text):
                    cand = date.fromisoformat(str(out["date"]))
                    if today - timedelta(days=62) <= cand <= today:
                        d, dc, used = cand, GUESS, True
                if h is None and out.get("hours") is not None:
                    v = float(out["hours"])
                    if 0 < v <= 24:
                        h, hc, used = v, GUESS, True
                if inc is None and out.get("income") is not None:
                    v = float(out["income"])
                    if 0 < v <= 5000 and _money_said(v, text):  # must appear in her words
                        inc, ic, used = v, GUESS, True
                if act is None and out.get("activity_id") is not None:
                    cand = next((a for a in activities if a["id"] == int(out["activity_id"])), None)
                    if cand:
                        act, ac, used = cand, GUESS, True
            except (TypeError, ValueError):
                pass
            if used:
                read_by = "ai"

    if d is None:
        d, dc = today, GUESS  # most people log the same day; she confirms
    missing = []
    if act is None:
        missing.append("activity")
    if h is None and inc is None:
        missing.append("hours")
    elif h is None and act and act["type"] not in ("work", "gig", "cash", "self_employment"):
        missing.append("hours")

    return {
        "text": text,
        "date": d.isoformat(),
        "hours": h,
        "income": inc,
        "activity_id": act["id"] if act else None,
        "confidence": {"date": dc, "hours": hc, "income": ic, "activity": ac},
        "read_by": read_by,
        "missing": missing,
    }
