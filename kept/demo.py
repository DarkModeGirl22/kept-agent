"""Keisha's demo: a fictional Montana gig worker, October and November 2026.

Her messages are real one-line texts, parsed by the same code as any user's, on the day
she would have sent them. Moving the demo clock forward replays them and runs the agent
for every day that passes, exactly as it would run for a real person."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from . import db
from . import services as S
from .agent import watchdog
from .capture import llm, textlog

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
START = date(2026, 10, 1)
KEY_DAY = date(2026, 10, 20)

ACTIVITIES = [
    dict(name="Dashly", type="gig", keywords="dashly,delivery,deliveries", pay_rate=13),
    dict(name="QuickCart", type="gig", keywords="quickcart,groceries,grocery", pay_rate=13),
    dict(name="Corner Market", type="cash", keywords="corner market,store", pay_rate=12,
         contact_name="Mr. Alvarez (owner)"),
    dict(name="Riverside Community Pantry", type="volunteer", keywords="pantry,food bank", usual_day=5, usual_hours=4,
         org_tax_id="00-0000000 (sample)", org_address="120 River St, Billings, MT (sample)",
         org_phone="(406) 555-0142", contact_name="Diane Brooks, volunteer coordinator"),
    dict(name="Adult Learning Center", type="education", keywords="ged,class,learning center", usual_day=1, usual_hours=3.5),
]

# (date, message she texts)
SCRIPT = [
    ("2026-10-01", "GED class 3.5h today"),
    ("2026-10-02", "Dashly 5-8pm, made $38"),
    ("2026-10-03", "4h pantry today"),
    ("2026-10-06", "Dashly 11am-3pm $52"),
    ("2026-10-06", "GED class 3.5h"),
    ("2026-10-08", "Corner Market 3h, $36 cash"),
    ("2026-10-09", "Dashly 6pm-9:30pm $44"),
    ("2026-10-10", "pantry 4 hours"),
    ("2026-10-11", "QuickCart 2.5h $34"),
    ("2026-10-13", "Dashly 10am-2pm $50"),
    ("2026-10-13", "GED class 3.5h"),
    ("2026-10-16", "Dashly 5pm-8:30pm $46"),
    ("2026-10-17", "4h pantry"),
    ("2026-10-20", "GED class 3.5h"),
    ("2026-10-21", "Dashly 4-8pm $50"),
    ("2026-10-23", "QuickCart 3h $40"),
    ("2026-10-24", "pantry 4h"),
    ("2026-10-25", "Dashly 12-5pm $66"),
    ("2026-10-27", "GED class 3.5h"),
    ("2026-10-28", "Corner Market 4h $48"),
    ("2026-10-30", "Dashly 5-9pm $52"),
    ("2026-10-31", "pantry 4h"),
    ("2026-11-02", "Dashly 4-8pm $52"),
    ("2026-11-03", "GED class 3.5h"),
    ("2026-11-06", "Dashly 5-9pm $54"),
    ("2026-11-07", "pantry 4h"),
    ("2026-11-09", "Dashly 11am-3pm $50"),
    ("2026-11-10", "GED class 3.5h"),
    ("2026-11-12", "QuickCart 3h $40"),
    ("2026-11-13", "Dashly 5-9pm $53"),
    ("2026-11-14", "pantry 4h"),
    ("2026-11-16", "Dashly 4-8pm $51"),
    ("2026-11-17", "GED class 3.5h"),
    ("2026-11-18", "Corner Market 4h $48"),
    ("2026-11-19", "QuickCart 3h $40"),
    ("2026-11-20", "Dashly 5-9pm $52"),
    ("2026-11-21", "pantry 4h"),
    ("2026-11-23", "Dashly 11am-3pm $54"),
    ("2026-11-24", "GED class 3.5h"),
]


def demo_person(conn):
    return conn.execute("SELECT * FROM person WHERE is_demo=1 ORDER BY id DESC LIMIT 1").fetchone()


def reset(conn, until: date = KEY_DAY):
    old = demo_person(conn)
    if old:
        db.delete_person(conn, old["id"])
    cur = conn.execute(
        "INSERT INTO person(name,state,birth_date,status,last_renewal,channel,language,quiet_start,quiet_end,"
        "max_alerts_week,seasonal,answers,is_demo,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("Keisha", "MT", "1992-04-12", "enrolled", "2026-07-01", "text", "en", 21, 8, 4, 0,
         json.dumps({"caregiver_child": False, "pregnant": False}), 1, db.now_iso()))
    pid = cur.lastrowid
    for a in ACTIVITIES:
        cols = ["person_id", "last_confirmed"] + list(a.keys())
        vals = [pid, START.isoformat()] + list(a.values())
        conn.execute(f"INSERT INTO activity({','.join(cols)}) VALUES({','.join('?' * len(vals))})", vals)
    conn.commit()
    db.set_setting(conn, "active_person", pid)
    db.set_setting(conn, "sim_today", START.isoformat())
    db.set_setting(conn, "demo_applied", "")
    advance_to(conn, until)
    return pid


def _apply_events(conn, person, until: date):
    done = set(json.loads(db.get_setting(conn, "demo_applied") or "[]"))
    acts = [dict(a) for a in db.activities(conn, person["id"])]
    for i, (d, msg) in enumerate(SCRIPT):
        day = date.fromisoformat(d)
        if day > until or i in done:
            continue
        r = textlog.parse(msg, day, acts)
        if r["activity_id"] and (r["hours"] or r["income"]):
            S.add_entry(conn, person["id"], r["activity_id"], day, day, r["hours"], r["income"],
                        "demo text", r["read_by"], note=f'Texted: "{msg}"')
        done.add(i)
    db.set_setting(conn, "demo_applied", json.dumps(sorted(done)))


def advance_to(conn, target: date):
    """Move the demo clock forward day by day: replay her texts, run the agent each day."""
    person = demo_person(conn)
    cur = db.today(conn)
    if target < cur:
        reset(conn, target)
        return
    last = conn.execute("SELECT MAX(day) d FROM agent_run WHERE person_id=?", (person["id"],)).fetchone()["d"]
    d = date.fromisoformat(last) + timedelta(days=1) if last else cur
    while d <= target:
        db.set_setting(conn, "sim_today", d.isoformat())
        with llm.paused():
            # her past texts are replayed with Kept's rules alone, so opening the demo is instant
            _apply_events(conn, person, d)
        # the agent's brain (Nemotron) only thinks about the day you land on; past days use the rules
        watchdog.run_day(conn, person, d, think=(d == target))
        d += timedelta(days=1)
    db.set_setting(conn, "sim_today", target.isoformat())


def receive_letter(conn):
    """The state's sample letter arrives (dated 23 Nov). Jump to 25 Nov if needed."""
    person = demo_person(conn)
    if db.today(conn) < date(2026, 11, 25):
        advance_to(conn, date(2026, 11, 25))
    doc_id, err = S.store_upload(conn, person["id"], SAMPLES / "sample_state_letter.png", "sample_state_letter.png",
                                 today=db.today(conn))
    if err == "duplicate":
        row = conn.execute("SELECT id FROM letter WHERE document_id=?", (doc_id,)).fetchone()
        return row["id"] if row else None
    doc = conn.execute("SELECT ocr_text FROM document WHERE id=?", (doc_id,)).fetchone()
    if not (doc["ocr_text"] or "").strip():
        # the AI is paused (no key, or its daily limit): use the letter's known text so the demo still works
        conn.execute("UPDATE document SET ocr_text=?, ocr_ok=1 WHERE id=?",
                     ((SAMPLES / "sample_state_letter.txt").read_text(), doc_id))
        conn.commit()
    ctx = S.context(conn, person)
    lid = S.save_letter(conn, person["id"], doc_id, ctx)
    watchdog.rerun_today(conn, person, db.today(conn))
    return lid
