"""The rest-of-the-month plan: NVIDIA Nemotron proposes, Kept's rules verify.

Nemotron 3 Nano sees what Keisha logged, her usual routine and the days left, and proposes
which days to work and for how long, in her own words. Before anything is shown, Kept's rules
check every line: a real day left in this month, one of her real activities, a believable
length, no day over 12 hours, and whether the plan actually reaches 80 hours (or $580). Lines
that fail are dropped and the reason is shown. If the model is unavailable or its plan fails,
Kept shows its own rules-only plan instead.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

from .. import db
from ..capture import llm
from ..rules import engine as E

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PAID = {"work", "gig", "cash", "self_employment"}

SYSTEM = (
    "You help a low-income worker keep her Medicaid. To keep it she needs 80 hours this month of "
    "work, volunteering or school, or $580 of pay. You get facts as JSON. Plan only the days left.\n"
    "Rules you must follow:\n"
    "- An activity with fixed_day only happens on that weekday: plan it on those dates only, at its usual hours.\n"
    "- Extra shifts are only possible for activities where extra_shifts is true (apps and jobs she already does), "
    "and only on weekdays she has worked them before (see last_3_weeks), at 3 to 5 hours.\n"
    "- Plan just enough: aim for 80 to 84 hours in total, not more. Keep at least one free day each week.\n"
    "- If income_left is small, mention the $580 path.\n"
    "Then write a message to her: warm, plain, 2 or 3 sentences, 6th-grade level, naming the days. "
    "Only use numbers that are in your plan or in the facts.\n"
    'Reply JSON only: {"items": [{"date": "YYYY-MM-DD", "activity_id": n, "hours": n}], "message": "..."}'
)


def _fixed(a) -> bool:
    """Volunteering and classes happen on their own schedule; she can't add a class on a Wednesday."""
    return a.get("usual_day") is not None and a["type"] not in PAID


def _facts(conn, ctx, today: date) -> dict:
    month = E.month_key(today)
    s = ctx.summary(month)
    end = E.month_end(today)
    acts = []
    for a in ctx.activities:
        if not a.get("active", 1):
            continue
        acts.append({"id": a["id"], "name": a["name"], "type": E.ACTIVITY_LABELS.get(a["type"], a["type"]),
                     "fixed_day": WEEKDAYS[a["usual_day"]] if _fixed(a) else None,
                     "usual_hours": a.get("usual_hours"), "extra_shifts": a["type"] in PAID,
                     "pay_per_hour": a.get("pay_rate")})
    since = (today - timedelta(days=21)).isoformat()
    recent = conn.execute(
        "SELECT e.start_date, e.hours, e.income, a.name FROM entry e JOIN activity a ON a.id=e.activity_id "
        "WHERE e.person_id=? AND e.superseded_by IS NULL AND e.start_date>=? AND e.start_date<=? ORDER BY e.start_date",
        (ctx.person["id"], since, today.isoformat())).fetchall()
    return {
        "today": f"{today.isoformat()} ({WEEKDAYS[today.weekday()]})",
        "month_ends": end.isoformat(),
        "hours_logged": s.hours_total, "hours_needed": 80, "hours_left": s.hours_remaining,
        "income_logged": round(s.income_total, 2), "income_needed": 580, "income_left": round(s.income_remaining, 2),
        "activities": acts,
        "last_3_weeks": [{"date": r["start_date"], "activity": r["name"], "hours": r["hours"], "pay": r["income"]}
                         for r in recent],
    }, s, end


def _verify(raw: dict, ctx, today: date, end: date, s) -> dict:
    acts = {a["id"]: a for a in ctx.activities if a.get("active", 1)}
    items, dropped, per_day = [], [], {}
    for it in raw.get("items") or []:
        if not isinstance(it, dict):
            continue
        why = str(it.get("why") or "")[:80]
        try:
            d = date.fromisoformat(str(it.get("date")))
            aid = int(it.get("activity_id"))
            h = float(it.get("hours"))
        except (TypeError, ValueError):
            dropped.append(("?", "not a complete line"))
            continue
        label = d.strftime("%a %b ") + str(d.day)
        if not (today < d <= end):
            dropped.append((label, "not a day left this month"))
            continue
        if aid not in acts:
            dropped.append((label, "not one of her activities"))
            continue
        if not 0.5 <= h <= 10:
            dropped.append((label, f"{h:g}h is not a believable shift"))
            continue
        a = acts[aid]
        if _fixed(a) and a["usual_day"] != d.weekday():
            dropped.append((label, f"{a['name']} only happens on {WEEKDAYS[a['usual_day']]}s"))
            continue
        if a["type"] not in PAID and a.get("usual_day") is None:
            dropped.append((label, f"{a['name']} has no set day to plan on"))
            continue
        if per_day.get(d, 0) + h > 12:
            dropped.append((label, "more than 12 hours that day"))
            continue
        per_day[d] = per_day.get(d, 0) + h
        usual = a.get("usual_day") == d.weekday()
        why = f"your usual {WEEKDAYS[d.weekday()]}" if usual else "an extra shift"
        pay = round(h * a["pay_rate"], 2) if a["type"] in PAID and a.get("pay_rate") else None
        items.append({"date": d.isoformat(), "label": label, "activity": a["name"], "type": a["type"], "hours": h,
                      "pay": pay, "usual": usual, "why": why})
    items.sort(key=lambda x: x["date"])
    adjusted = []

    # her fixed days come first: if the model forgot one, Kept adds it
    d = today + timedelta(days=1)
    while d <= end:
        for a in acts.values():
            if _fixed(a) and a["usual_day"] == d.weekday() and a.get("usual_hours") and \
                    not any(i["date"] == d.isoformat() and i["activity"] == a["name"] for i in items):
                items.append({"date": d.isoformat(), "label": d.strftime("%a %b ") + str(d.day), "activity": a["name"],
                              "type": a["type"], "hours": a["usual_hours"], "pay": None, "usual": True,
                              "why": f"your usual {WEEKDAYS[d.weekday()]}", "added": True})
                adjusted.append(f"Added {a['name']} on {d.strftime('%a %b ')}{d.day}: it's her usual day.")
        d += timedelta(days=1)
    items.sort(key=lambda x: x["date"])

    def worked_days():
        return sorted({date.fromisoformat(i["date"]) for i in items})

    # at least one free day in any 7: drop the latest extra shift of a long streak
    changed = True
    while changed:
        changed = False
        days = worked_days()
        run = [days[0]] if days else []
        for x in days[1:]:
            run = run + [x] if (x - run[-1]).days == 1 else [x]
            if len(run) >= 7:
                extra = [i for i in items if not i["usual"] and date.fromisoformat(i["date"]) in run]
                if extra:
                    victim = extra[-1]
                    items.remove(victim)
                    adjusted.append(f"Removed the extra shift on {victim['label']}: 7 days in a row is too many.")
                    changed = True
                break

    # still short after the checks: add an extra shift on a free day, with the app or job she already
    # picked for extra shifts, without making 7 days in a row
    paid = [a for a in acts.values() if a["type"] in PAID]
    pick = None
    extras = [i["activity"] for i in items if not i["usual"]]
    if extras:
        pick = next((a for a in paid if a["name"] == max(set(extras), key=extras.count)), None)
    pick = pick or (paid[0] if paid else None)
    d = today + timedelta(days=1)
    while pick and d <= end and s.hours_total + sum(i["hours"] for i in items) < 80:
        taken = {date.fromisoformat(i["date"]) for i in items}
        if d not in taken:
            trial = sorted(taken | {d})
            run, longest = 1, 1
            for a_, b_ in zip(trial, trial[1:]):
                run = run + 1 if (b_ - a_).days == 1 else 1
                longest = max(longest, run)
            if longest < 7:
                need = 80 - s.hours_total - sum(i["hours"] for i in items)
                h = min(5.0, max(3.0, round(need * 2) / 2))
                pay = round(h * pick["pay_rate"], 2) if pick.get("pay_rate") else None
                items.append({"date": d.isoformat(), "label": d.strftime("%a %b ") + str(d.day), "activity": pick["name"],
                              "type": pick["type"], "hours": h, "pay": pay, "usual": False,
                              "why": "an extra shift", "added": True})
                items.sort(key=lambda x: x["date"])
                adjusted.append(f"Added an extra {pick['name']} shift on {d.strftime('%a %b ')}{d.day} to reach 80.")
        d += timedelta(days=1)

    # just enough: drop the latest extra shifts while she still reaches 80 without them
    while True:
        total = s.hours_total + sum(i["hours"] for i in items)
        extra = [i for i in items if not i["usual"]]
        if not extra or total - extra[-1]["hours"] < 80:
            break
        items.remove(extra[-1])
        adjusted.append(f"Removed the extra shift on {extra[-1]['label']}: not needed to reach 80.")

    planned_h = round(sum(i["hours"] for i in items), 2)
    planned_pay = round(sum(i["pay"] or 0 for i in items), 2)
    reaches_h = s.hours_total + planned_h >= 80
    reaches_pay = s.income_total + planned_pay >= 580
    return {"items": items, "dropped": dropped, "adjusted": adjusted, "planned_hours": planned_h,
            "planned_pay": planned_pay, "reaches_hours": reaches_h, "reaches_income": reaches_pay,
            "total_after": round(s.hours_total + planned_h, 2)}


def _message_ok(msg: str, v: dict, s) -> bool:
    """The message may only use numbers that the checked plan or the month's summary contain."""
    if not isinstance(msg, str) or not 20 <= len(msg) <= 500:
        return False
    allowed = {80, 580, round(s.hours_total, 2), round(s.hours_remaining, 2), round(v["planned_hours"], 2),
               round(v["total_after"], 2), round(s.income_total), round(s.income_remaining), round(v["planned_pay"])}
    allowed |= {i["hours"] for i in v["items"]} | {int(i["date"][8:]) for i in v["items"]}
    allowed |= {round(i["pay"]) for i in v["items"] if i["pay"]}
    for n in re.findall(r"\d+(?:\.\d+)?", msg.replace(",", "")):
        f = float(n)
        if not any(abs(f - float(a)) < 0.01 for a in allowed):
            return False
    return True


def rules_plan(ctx, today: date, end: date, s) -> dict:
    """Kept's own plan, without AI: her usual days left this month."""
    items = []
    d = today + timedelta(days=1)
    while d <= end:
        for a in ctx.activities:
            if a.get("active", 1) and a.get("usual_day") == d.weekday() and a.get("usual_hours"):
                pay = round(a["usual_hours"] * a["pay_rate"], 2) if a["type"] in PAID and a.get("pay_rate") else None
                items.append({"date": d.isoformat(), "label": d.strftime("%a %b ") + str(d.day), "activity": a["name"],
                              "type": a["type"], "hours": a["usual_hours"], "pay": pay, "usual": True, "why": "your usual day"})
        d += timedelta(days=1)
    planned_h = round(sum(i["hours"] for i in items), 2)
    planned_pay = round(sum(i["pay"] or 0 for i in items), 2)
    return {"items": items, "dropped": [], "adjusted": [], "planned_hours": planned_h, "planned_pay": planned_pay,
            "reaches_hours": s.hours_total + planned_h >= 80, "reaches_income": s.income_total + planned_pay >= 580,
            "total_after": round(s.hours_total + planned_h, 2)}


def make_plan(conn, ctx, today: date, fresh: bool = False) -> dict:
    key = f"plan:{ctx.person['id']}:{today.isoformat()}"
    if not fresh:
        cached = db.get_setting(conn, key)
        if cached:
            return json.loads(cached)
    facts, s, end = _facts(conn, ctx, today)
    plan = None
    if llm.available("plan") and s.status not in ("met", "exempt"):
        raw = llm.chat_json(SYSTEM, json.dumps(facts), task="plan", max_tokens=1500)
        if raw:
            v = _verify(raw, ctx, today, end, s)
            if v["items"]:
                msg = raw.get("message")
                if v["adjusted"] or not _message_ok(msg, v, s):
                    # the rules changed the plan: Nemotron Nano rewrites the message for the checked plan
                    out = llm.chat_json(
                        "Write a short message to a busy worker about her plan for the rest of the month: warm, "
                        "plain, at most 2 sentences and 45 words, 6th-grade level. Group the days by what she does "
                        "instead of listing hours for each day. Only use numbers that appear in "
                        'the JSON. Reply JSON only: {"message": "..."}',
                        json.dumps({"hours_now": s.hours_total, "hours_after_plan": v["total_after"],
                                    "goal": 80, "plan": [{"day": i["label"], "what": i["activity"], "hours": i["hours"]}
                                                         for i in v["items"]]}),
                        task="fast", max_tokens=300)
                    msg = (out or {}).get("message")
                v["message"] = msg.strip() if _message_ok(msg, v, s) else None
                v["read_by"] = "ai"
                v["model"] = llm.LABELS["plan"]
                plan = v
    if plan is None:
        plan = rules_plan(ctx, today, end, s)
        plan["message"] = None
        plan["read_by"] = "rules"
    plan.update(month=E.month_label(E.month_key(today)), hours_total=s.hours_total, hours_left=s.hours_remaining,
                income_total=round(s.income_total, 2), status=s.status, days_left=s.days_left,
                still_short=round(max(0.0, 80 - s.hours_total - plan["planned_hours"]), 2))
    db.set_setting(conn, key, json.dumps(plan))
    return plan
