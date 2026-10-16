"""The agent's brain: NVIDIA Nemotron 3 Super decides what Keisha hears today.

Every day, Kept's rules first look at her situation the way a careful caseworker would
(letters, exemptions, the month, the state check, proof, quiet days) and write down what they
found. Then Nemotron takes over as the agent. It can call tools to look deeper — her month,
the work still missing proof, her week ahead — and decides:

  - which of the rules' findings are worth a message today, and which can wait;
  - how to say each one, in plain words, about her own week;
  - at most one idea of its own (for example: "Saturday is your pantry day: ask Diane to sign
    the October page while you're there").

Its decision then goes back through Kept's rules before anything reaches her: urgent
messages (letters, deadlines) are always sent even if the model held them; every number and
date in a message must come from what the tools returned; links must point to real screens;
the weekly message limit she chose still applies. Anything that fails a check falls back to
the rules' own wording, and every step is written to "What Kept checked".
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta

from .. import db
from ..capture import llm
from ..rules import engine as E

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MAX_ROUNDS = 5
ALLOWED_ACTIONS = {"/": "See my month", "/add": "Add proof", "/plan": "Plan the rest", "/folder": "Open my proof folder",
                   "/calendar": "See the days", "/profile": "See details"}

SYSTEM = (
    "You are Kept, an agent that helps one low-income worker in the United States keep her Medicaid. "
    "From 2027 she must show 80 hours a month of work, volunteering or school, or $580 of pay, and keep proof. "
    "Once a day you decide what to tell her. Kept's rules have already checked her situation; call "
    "get_rule_findings first, then use the other tools if you need details. Then call decide.\n"
    "How to decide:\n"
    "- Urgent findings (letters, deadlines) must be sent.\n"
    "- Send few messages: only what helps her this week. She chose a weekly limit; respect it.\n"
    "- Write like a kind friend who knows the rules: plain words, 1 or 2 short sentences, about her real days "
    "and activities. Name the concrete next step.\n"
    "- You may add at most one idea of your own (about='new') if it clearly helps, for example tying missing "
    "proof to a day she will be at that place.\n"
    "- Only use numbers and dates that the tools returned. Never invent hours, pay or deadlines.\n"
    "- Be honest about where she stands: if the month's status is At risk or Short, never say she is on track, set or fine.\n"
    "- Do not mention health conditions or anything private.\n"
    "Today is {today}."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "get_rule_findings",
        "description": "What Kept's rules found today: each finding has a key, whether it is urgent, a title and the rules' own text.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_situation",
        "description": "This month's hours and pay against the goal, days left, the next state check and the months it looks at, open letters, and how many non-urgent messages she can still get this week.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_missing_proof",
        "description": "Work she logged that has no proof yet (date, activity, hours), and volunteer months still needing a signature.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_week",
        "description": "What she logged in the last 7 days, and her usual activities in the next 7 days (date, weekday, activity, usual hours).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "decide",
        "description": "Your final decision for today.",
        "parameters": {"type": "object", "properties": {
            "messages": {"type": "array", "description": "Messages to send today.", "items": {"type": "object", "properties": {
                "about": {"type": "string", "description": "The key of the rule finding this message is about, or 'new' for your own idea."},
                "title": {"type": "string", "description": "Short title, under 60 characters."},
                "body": {"type": "string", "description": "1 or 2 short sentences."},
                "action": {"type": "string", "enum": list(ALLOWED_ACTIONS), "description": "The screen the button opens."},
                "why": {"type": "string", "description": "Why this is worth telling her today, in one sentence."}},
                "required": ["about", "title", "body", "why"]}},
            "hold": {"type": "array", "description": "Findings not worth a message today.", "items": {"type": "object", "properties": {
                "about": {"type": "string"}, "why": {"type": "string"}}, "required": ["about", "why"]}},
            "summary": {"type": "string", "description": "One sentence: how her situation looks today."}},
            "required": ["messages"]}}},
]


# ------------------------------------------------------------------ tools (read-only views of Kept's data)

def _tool_results(run) -> dict:
    """Compute every tool's answer once; the model chooses which to read."""
    ctx, day, pid = run.ctx, run.day, run.person["id"]
    month = E.month_key(day)
    s = ctx.summary(month)
    budget = run.person["max_alerts_week"] or 4
    week_start = day - timedelta(days=day.weekday())
    used = run.conn.execute(
        "SELECT COUNT(*) c FROM alert WHERE person_id=? AND status='sent' AND urgent=0 AND day>=? AND day<?",
        (pid, week_start.isoformat(), day.isoformat())).fetchone()["c"]
    cd = ctx.check_date()
    letters = run.conn.execute("SELECT * FROM letter WHERE person_id=? AND status='open'", (pid,)).fetchall()
    situation = {
        "today": f"{WEEKDAYS[day.weekday()]} {day.isoformat()}",
        "month": E.month_label(month), "status": E.STATUS_TEXT.get(s.status, s.status),
        "hours_logged": s.hours_total, "hours_goal": 80, "hours_left": s.hours_remaining,
        "hours_with_proof": s.hours_proven, "pay_logged": round(s.income_total, 2), "pay_goal": 580,
        "pay_left": round(s.income_remaining, 2), "days_left_in_month": s.days_left,
        "next_state_check": cd.isoformat() if cd else None,
        "days_until_check": (cd - day).days if cd else None,
        "months_checked": [E.month_label(m) for m in ctx.checked_months()],
        "open_letters": [{"id": L["id"], "deadline": L["deadline"],
                          "days_left": (date.fromisoformat(L["deadline"]) - day).days} for L in letters],
        "messages_left_this_week": max(0, budget - used),
    }
    findings = [{"key": c.key, "urgent": c.urgent, "title": c.title, "text": c.body} for c in run.candidates]
    rows = run.conn.execute(
        "SELECT e.start_date, e.hours, e.income, a.name, a.type FROM entry e JOIN activity a ON a.id=e.activity_id "
        "WHERE e.person_id=? AND e.superseded_by IS NULL AND e.document_id IS NULL AND e.start_date<=? "
        "AND e.start_date>=? ORDER BY e.start_date", (pid, day.isoformat(), E.month_start(day).replace(day=1).isoformat())).fetchall()
    missing = [{"date": r["start_date"], "activity": r["name"], "hours": r["hours"]} for r in rows if r["type"] != "volunteer"]
    sign = [{"activity": a["name"], "month": E.month_label(month), "coordinator": a.get("contact_name")}
            for a in ctx.activities if a["type"] == "volunteer" and a.get("active", 1)
            and not run.conn.execute("SELECT 1 FROM signoff WHERE person_id=? AND activity_id=? AND month=? AND status='signed'",
                                     (pid, a["id"], month)).fetchone()]
    last7 = run.conn.execute(
        "SELECT e.start_date, e.hours, a.name FROM entry e JOIN activity a ON a.id=e.activity_id WHERE e.person_id=? "
        "AND e.superseded_by IS NULL AND e.start_date>? AND e.start_date<=? ORDER BY e.start_date",
        (pid, (day - timedelta(days=7)).isoformat(), day.isoformat())).fetchall()
    ahead = []
    for i in range(1, 8):
        d = day + timedelta(days=i)
        for a in ctx.activities:
            if a.get("active", 1) and a.get("usual_day") == d.weekday() and a.get("usual_hours"):
                ahead.append({"date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()], "activity": a["name"],
                              "usual_hours": a["usual_hours"]})
    return {
        "get_rule_findings": {"findings": findings},
        "get_situation": situation,
        "get_missing_proof": {"logged_without_proof": missing, "volunteer_pages_to_sign": sign},
        "get_week": {"last_7_days": [{"date": r["start_date"], "activity": r["name"], "hours": r["hours"]} for r in last7],
                     "next_7_days_usual": ahead},
    }


# ------------------------------------------------------------------ the checks

def _facts_numbers(results: dict) -> set:
    text = json.dumps(results)
    nums = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}
    for iso in re.findall(r"\d{4}-\d{2}-\d{2}", text):   # "Oct 24" may be written for 2026-10-24
        nums.add(float(int(iso[8:])))
    return nums


def _numbers_ok(text: str, allowed: set) -> bool:
    for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")):
        if not any(abs(float(n) - a) < 0.01 for a in allowed):
            return False
    return True


REASSURING = re.compile(r"\b(on track|all set|you'?re set|you'?re fine|you'?re good|covered|nothing to worry|no need to)\b", re.I)
PRIVATE = re.compile(r"\b(pregnan|disab|illness|diagnos|treatment|jail|prison|incarcerat|addict|frail|hospital)", re.I)


def check_decision(decision: dict, candidates: list, results: dict) -> tuple[list, list]:
    """Turn Nemotron's decision into today's messages, keeping only what passes the rules.
    Returns (candidates to deliver, notes for the log)."""
    from .watchdog import Candidate

    by_key = {c.key: c for c in candidates}
    allowed = _facts_numbers(results)
    for c in candidates:   # the rules' own text is a fact too
        allowed |= {float(n) for n in re.findall(r"\d+(?:\.\d+)?", (c.title + " " + c.body).replace(",", ""))}
    out, notes, covered, new_used = [], [], set(), False
    for m in decision.get("messages") or []:
        if not isinstance(m, dict):
            continue
        about, title, body = str(m.get("about") or ""), str(m.get("title") or "").strip(), str(m.get("body") or "").strip()
        why = str(m.get("why") or "").strip()[:200]
        problems = []
        if not title or not body or len(title) > 80 or len(body) > 420:
            problems.append("too long or empty")
        if not _numbers_ok(title + " " + body, allowed):
            problems.append("a number or date that Kept's data doesn't have")
        if PRIVATE.search(title + " " + body):
            problems.append("mentions something private")
        status = str((results.get("get_situation") or {}).get("status") or "")
        if status in ("At risk", "Short") and REASSURING.search(title + " " + body):
            problems.append(f"says she's fine while the month is {status.lower()}")
        action = m.get("action") if m.get("action") in ALLOWED_ACTIONS else None
        if about in by_key:
            base = by_key[about]
            covered.add(about)
            if problems:
                out.append(base)
                notes.append(("rules", f"'{base.title}': Nemotron's wording refused ({', '.join(problems)}); the rules' text is used."))
                continue
            out.append(Candidate(base.key, base.kind, base.priority, title, body, base.action_url if base.urgent or not action else action,
                                 base.action_label if base.urgent or not action else ALLOWED_ACTIONS[action],
                                 urgent=base.urgent, cooldown_days=base.cooldown_days, by="nemotron", why=why))
            notes.append(("send", f"'{title}' ({about}). Why: {why}"))
        elif about == "new":
            if new_used:
                notes.append(("refused", f"'{title}': only one idea of its own per day."))
                continue
            if problems:
                notes.append(("refused", f"'{title}': {', '.join(problems)}."))
                continue
            new_used = True
            out.append(Candidate(f"brain:{hashlib.sha1(title.encode()).hexdigest()[:8]}", "coach", 3, title, body,
                                 action or "/", ALLOWED_ACTIONS[action or "/"], cooldown_days=7, by="nemotron", why=why))
            notes.append(("send", f"Its own idea: '{title}'. Why: {why}"))
        else:
            notes.append(("refused", f"'{title}': refers to '{about}', which isn't a finding today."))
    held = {str(h.get("about")): str(h.get("why") or "")[:200] for h in decision.get("hold") or [] if isinstance(h, dict)}
    for c in candidates:
        if c.key in covered:
            continue
        if c.urgent:
            out.append(c)
            notes.append(("rules", f"'{c.title}' is urgent: sent anyway with the rules' text."
                                   + (f" (Nemotron wanted to hold it: {held[c.key]})" if c.key in held else "")))
        else:
            notes.append(("hold", f"'{c.title}': not today. Why: {held.get(c.key) or 'Nemotron left it out.'}"))
    return out, notes


# ------------------------------------------------------------------ the loop

def _state_key(run) -> str:
    s = run.ctx.summary(E.month_key(run.day))
    raw = json.dumps([run.day.isoformat(), [c.key for c in run.candidates], s.hours_total, s.hours_proven,
                      round(s.income_total, 2)])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def think(run) -> bool:
    """Let Nemotron decide today's messages. Returns False (and changes nothing) if it can't."""
    if not llm.available("plan") or llm.provider() != "nvidia":
        return False
    key = f"brain:{run.person['id']}:{_state_key(run)}"
    cached = db.get_setting(run.conn, key)
    results = _tool_results(run)
    decision, calls = None, []
    if cached:
        decision, calls = json.loads(cached), ["(same situation as earlier today: decision reused)"]
    else:
        messages = [{"role": "system", "content": SYSTEM.format(today=f"{WEEKDAYS[run.day.weekday()]} {run.day.isoformat()}")},
                    {"role": "user", "content": "Do your daily check for her and decide what to tell her today."}]
        for _ in range(MAX_ROUNDS):
            reply = llm.chat_tools(messages, TOOLS, task="plan", max_tokens=1200)
            if not reply:
                break
            tcs = reply.get("tool_calls") or []
            if not tcs:
                parsed = llm._parse_json(reply.get("content") or "")
                if parsed and "messages" in parsed:
                    decision = parsed
                break
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                name = (tc.get("function") or {}).get("name")
                if name == "decide":
                    try:
                        decision = json.loads((tc.get("function") or {}).get("arguments") or "{}")
                    except ValueError:
                        decision = None
                    break
                calls.append(name)
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "name": name,
                                 "content": json.dumps(results.get(name, {"error": "unknown tool"}))})
            if decision is not None:
                break
        if decision is None:
            run.log("brain", "fallback", "Nemotron didn't reach a decision in time; Kept's rules decide today.")
            return False
        db.set_setting(run.conn, key, json.dumps(decision))
    new, notes = check_decision(decision, run.candidates, results)
    run.log("brain", "think", f"{llm.LABELS['plan']} looked at: {', '.join(calls) or 'nothing'}."
            + (f" Its summary: {str(decision.get('summary'))[:200]}" if decision.get("summary") else ""))
    for kind, text in notes:
        run.log("brain", kind, text)
    run.candidates = new
    return True
