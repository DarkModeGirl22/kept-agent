"""Understand a letter from the state: what kind it is, the deadline, the months it asks
about. The deadline always comes from the letter or the 30-day rule, never from the AI."""

from __future__ import annotations

import re
from datetime import date

from . import llm
from .extract import find_dates
from .textlog import MONTHS

KINDS = {
    "noncompliance": ["not met", "did not meet", "unable to verify", "could not verify", "noncompliance",
                      "non-compliance", "will end", "will be closed", "terminate"],
    "verification_request": ["request for information", "request for verification", "please provide",
                             "send us proof", "verify", "verification"],
    "renewal": ["renewal", "redetermination", "renew your coverage"],
    "approval": ["approved", "you are eligible", "has been approved"],
}

TITLES = {
    "noncompliance": "Notice: the state could not confirm your hours",
    "verification_request": "Request for proof",
    "renewal": "Renewal notice",
    "approval": "Approval notice",
    "other": "Letter from the state",
}

EXPLAIN = {
    "noncompliance": ("The state says it could not confirm that you met the 80 hours (or $580) for the "
                      "months listed. Your coverage continues while you answer. If you send proof before the "
                      "deadline, it can keep going. If you don't answer, it can end the month after."),
    "verification_request": ("The state is asking you to send proof of your hours or income for the months "
                             "listed. Kept has already gathered what you logged for those months."),
    "renewal": ("It's time to renew your coverage. The state will look at the months listed to see if you "
                "met the 80 hours (or $580), or have an exemption."),
    "approval": "Good news: this letter approves your coverage. Keep it in your folder.",
    "other": "Kept couldn't tell exactly what this letter asks. Please read it, and set the deadline if it gives one.",
}

DEADLINE_RX = re.compile(r"(respond|return|reply|send[^.\n]{0,30}?)\s*(?:it\s+)?(?:to us\s+)?(by|before|no later than)\b|"
                         r"\b(due date|deadline)\b[:\s]|\bdue\s*(:|by\b|on\b)", re.I)   # not "due for renewal"


def classify(text: str) -> str:
    t = text.lower()
    # the letter's own title wins over words used in its body
    if re.search(r"request for (verification|information)|please provide", t):
        return "verification_request"
    for k in ("noncompliance", "verification_request", "renewal", "approval"):
        if any(w in t for w in KINDS[k]):
            return k
    return "other"


def months_mentioned(text: str) -> list[str]:
    out = []
    for m in re.finditer(rf"\b({'|'.join(MONTHS)})\s+(\d{{4}})\b", text, re.I):
        k = f"{int(m.group(2)):04d}-{MONTHS.index(m.group(1).lower()) + 1:02d}"
        if k not in out:
            out.append(k)
    return sorted(out)


def read_letter(text: str, today: date) -> dict:
    kind = classify(text)
    dates = find_dates(text, today.year)
    respond_by = None
    for m in DEADLINE_RX.finditer(text):
        after = [d for d, pos in dates if pos >= m.start() and pos - m.end() < 60]
        if after:
            respond_by = after[0]
            break
    letter_date = next((d for d, _ in dates if d <= today and d != respond_by), None)
    explanation = EXPLAIN[kind]
    read_by = "rules"

    explanation_es = None
    if llm.available("deep"):
        out = llm.chat_json(
            "You explain a letter from a US state Medicaid office to a busy, low-income worker. "
            "Write 2 or 3 short, plain sentences at a 6th-grade reading level: what the letter asks, "
            "by when, and what happens if she does nothing. Then the same in simple Spanish. "
            "Do not give legal advice. Only use dates and numbers that are written in the letter. "
            'Reply JSON only: {"explanation_en": "...", "explanation_es": "..."}',
            text[:6000], task="deep", max_tokens=600)
        if out:
            def ok(e):
                # short enough, and no number or date that isn't in the letter
                return isinstance(e, str) and 30 <= len(e.strip()) <= 700 and \
                    not any(n not in text for n in re.findall(r"\d{2,}", e))
            en, es = out.get("explanation_en"), out.get("explanation_es")
            if ok(en):
                explanation, read_by = en.strip(), "ai"
                if ok(es):
                    explanation_es = es.strip()

    return {
        "kind": kind,
        "title": TITLES[kind],
        "letter_date": letter_date,
        "respond_by": respond_by,
        "months": months_mentioned(text),
        "explanation": explanation,
        "explanation_es": explanation_es,
        "read_by": read_by,
    }
