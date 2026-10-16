"""Measure how well NVIDIA Nemotron does each job in Kept, and how often Kept's rules step in.

Runs every sample in samples/ through the real code paths, against Nebius Token Factory,
and compares the result with the known answer for each sample. Prints a Markdown report.

    NEBIUS_API_KEY=... python tools/eval.py

It uses a throwaway data folder, so it never touches your own data. A full run makes about
40 model calls and costs about 3 cents.

How a value is scored (photos, video, notes):
  correct  - saved with the right value
  asked    - Kept wasn't sure, so it asks her to confirm (safe: nothing wrong is saved)
  wrong    - saved with a wrong value, without asking (the failure that matters)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_tmp = Path(tempfile.mkdtemp(prefix="kept-eval-"))
os.environ["KEPT_DATA_DIR"] = str(_tmp)
os.environ["KEPT_AI_USAGE_FILE"] = str(_tmp / "usage.json")

from kept import db, demo  # noqa: E402
from kept import services as S  # noqa: E402
from kept.agent import planner  # noqa: E402
from kept.capture import extract, letters, llm, textlog, vision  # noqa: E402

SAMPLES = ROOT / "samples"
ACTS = [dict(a, id=i + 1) for i, a in enumerate(demo.ACTIVITIES)]
DASHLY, QUICKCART, CORNER, PANTRY, SCHOOL = 1, 2, 3, 4, 5

# ------------------------------------------------------------------ known answers

PHOTOS = {  # file: (start, end, hours, income); None = not on the document
    "proof_corner_note_oct8.jpg": ("2026-10-08", "2026-10-08", 3, 36),       # handwritten
    "proof_dashly_oct16.png": ("2026-10-16", "2026-10-16", 3.5, 64),
    "proof_pantry_oct16.png": ("2026-10-16", "2026-10-16", 4, None),
    "proof_class_oct15.png": ("2026-10-15", "2026-10-15", 2.5, None),
    "sample_gig_statement.png": ("2026-10-05", "2026-10-11", 7.5, 96),
    "sample_pay_stub.png": ("2026-10-19", "2026-10-31", 4, 48),
    "sample_volunteer_sheet.png": ("2026-10-01", "2026-10-31", 20, None),
}
PHOTO_DAY = date(2026, 11, 5)

VIDEO_ROWS = {  # date: (hours, pay), from tools/make_history_video.py
    "2026-10-16": (3.5, 64), "2026-10-13": (4, 50), "2026-10-09": (3.5, 44), "2026-10-06": (4, 52),
    "2026-10-02": (3, 38), "2026-09-29": (3, 41), "2026-09-25": (3.5, 47), "2026-09-22": (3.5, 45),
}
VIDEO_DAY = date(2026, 10, 20)

NOTE_DAY = date(2026, 10, 20)   # a Tuesday
NOTES = [  # what she types or says: (date, hours, income, activity); None = not said, not scored
    ("ayer trabajé en la tienda cuatro horas, me pagaron cuarenta y ocho", "2026-10-19", 4, 48, CORNER),
    ("did dashly yesterday from five to nine, made fifty two dollars", "2026-10-19", 4, 52, DASHLY),
    ("food bank on saturday, four hours", "2026-10-17", 4, None, PANTRY),
    ("fui al banco de comida el sábado, cuatro horas", "2026-10-17", 4, None, PANTRY),
    ("quickcart this morning three hours forty dollars", "2026-10-20", 3, 40, QUICKCART),
    ("clase de GED anoche, tres horas y media", "2026-10-19", 3.5, None, SCHOOL),
    ("helped at the store on sunday two hours, got twenty four cash", "2026-10-18", 2, 24, CORNER),
    ("hice entregas el viernes, gané sesenta y cuatro dólares", "2026-10-16", None, 64, DASHLY),
    ("Dashly 5-8pm, made $38", "2026-10-20", 3, 38, DASHLY),
    ("grocery deliveries last friday 2.5 hours $34", "2026-10-16", 2.5, 34, QUICKCART),
]

LETTERS = [  # (name, text source, letter's own date, respond by, months)
    ("State letter (photo)", "sample_state_letter.png", date(2026, 11, 25), "2026-12-23", ["2026-10", "2026-11"]),
    ("Renewal letter (PDF)", "sample_renewal_letter.pdf", date(2026, 10, 20), "2026-12-31",
     ["2026-10", "2026-11", "2026-12"]),
]

BRAIN_DAYS = [date(2026, 10, 20), date(2026, 10, 27), date(2026, 11, 25)]


# ------------------------------------------------------------------ helpers

class Meter:
    """Model calls, model time and cost for one section, from the usage Token Factory returns."""

    def __enter__(self):
        u = llm._load_usage()
        self.u0 = (u.get("total_calls", 0), u.get("total_usd", 0.0),
                   sum(m["seconds"] for m in u.get("by_model", {}).values()))
        return self

    def __exit__(self, *exc):
        u = llm._load_usage()
        self.calls = u.get("total_calls", 0) - self.u0[0]
        self.usd = u.get("total_usd", 0.0) - self.u0[1]
        self.secs = sum(m["seconds"] for m in u.get("by_model", {}).values()) - self.u0[2]


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) < 0.01
    except (TypeError, ValueError):
        return str(a)[:10] == str(b)[:10]   # dates, as date objects or "YYYY-MM-DD"


def tally(rows) -> dict:
    t = {"correct": 0, "asked": 0, "wrong": 0}
    for r in rows:
        t[r] += 1
    return t


def score(value, conf, truth, ask_below=extract.ASK_BELOW) -> str:
    if conf < ask_below or value is None:
        return "asked"
    return "correct" if same(value, truth) else "wrong"


# ------------------------------------------------------------------ the jobs

def eval_photos(log):
    model_ok = model_n = 0
    final, rules_fixed, detail = [], 0, []
    for name, truth in PHOTOS.items():
        t0 = time.time()
        r = vision.read_photo(SAMPLES / name, PHOTO_DAY)
        if not r:
            detail.append(f"{name}: no answer")
            final += ["asked"] * sum(v is not None for v in truth)
            continue
        out = extract.extract(r["text"], PHOTO_DAY, ACTS, ai=r["ai"], ai_kind=r["kind"])
        f, c = out["fields"], out["confidence"]
        marks = []
        for k, tv in zip(("period_start", "period_end", "hours", "income"), truth):
            if tv is None:
                continue
            mv = r["ai"].get(k)
            model_n += 1
            m_ok = same(mv, tv)
            model_ok += m_ok
            s = score(f.get(k), c.get(k, 0), tv)
            final.append(s)
            if not m_ok and s != "wrong":
                rules_fixed += 1
            marks.append(f"{k}={'ok' if s == 'correct' else s.upper()}" + ("" if m_ok else f" (model said {mv})"))
        detail.append(f"{name} ({time.time() - t0:.1f}s): " + ", ".join(marks))
    log += detail
    return {"model": f"{model_ok}/{model_n}", "final": tally(final), "rules_caught": rules_fixed}


def eval_video(log):
    t0 = time.time()
    frames = vision.sample_frames(SAMPLES / "sample_dashly_history.mp4", _tmp / "frames")
    out = vision.read_history(frames, VIDEO_DAY) or {"rows": []}
    rows = {r["start"]: r for r in out["rows"]}
    marks = []
    for d, (h, p) in VIDEO_ROWS.items():
        r = rows.get(d)
        if not r:
            marks += ["asked", "asked"]   # not found: she adds it by hand
            log.append(f"video {d}: not found")
            continue
        good = same(r["hours"], h) and same(r["pay"], p)
        if not r["ok"]:
            marks += ["asked", "asked"]
        else:
            marks += ["asked" if got is None else ("correct" if same(got, want) else "wrong")
                      for got, want in ((r["hours"], h), (r["pay"], p))]
        if not good or not r["ok"]:
            log.append(f"video {d}: hours {r['hours']} pay {r['pay']} checks {r['problems']}")
    extra = [d for d in rows if d not in VIDEO_ROWS]
    flagged_extra = sum(1 for d in extra if not rows[d]["ok"])
    log.append(f"video: {len(frames)} frames kept, {len(out['rows'])} rows, {time.time() - t0:.1f}s")
    if extra:
        log.append(f"video: rows that aren't in the video: {extra} ({flagged_extra} flagged by the rules)")
    return {"frames": len(frames), "found": sum(1 for d in VIDEO_ROWS if d in rows), "final": tally(marks),
            "invented": len(extra) - flagged_extra}


def eval_notes(log):
    def run(use_ai):
        llm.set_enabled(use_ai)
        res = []
        for text, d, h, inc, act in NOTES:
            r = textlog.parse(text, NOTE_DAY, ACTS)
            # a note becomes a draft she checks before saving: a blank is "asked", a filled-in value is scored
            marks = [score(r["date"], 1, d)]
            if h is not None:
                marks.append(score(r["hours"], 1, h))
            if inc is not None:
                marks.append(score(r["income"], 1, inc))
            marks.append(score(r["activity_id"], 1, act))
            res.append(marks)
            if use_ai and any(m != "correct" for m in marks):
                log.append(f'note "{text}": {r["date"]} {r["hours"]}h ${r["income"]} act {r["activity_id"]} -> {marks}')
        llm.set_enabled(True)
        return res

    rules = run(False)
    ai = run(True)
    whole = lambda res: sum(all(m == "correct" for m in x) for x in res)
    return {"n": len(NOTES), "rules_whole": whole(rules), "ai_whole": whole(ai),
            "final": tally(m for x in ai for m in x), "rules": tally(m for x in rules for m in x)}


def eval_letters(log):
    res = []
    for name, src, today, due, months in LETTERS:
        if src.endswith(".pdf"):
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(str(SAMPLES / src)).pages)
        else:
            r = vision.read_photo(SAMPLES / src, today)
            text = r["text"] if r else (SAMPLES / "sample_state_letter.txt").read_text()
        out = letters.read_letter(text, today)
        ok = {"explained by Nemotron": out["read_by"] == "ai", "Spanish": bool(out["explanation_es"]),
              "deadline right": str(out["respond_by"]) == due, "months right": out["months"] == months}
        res.append(ok)
        log.append(f"{name}: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in ok.items()))
        log.append(f"   EN: {out['explanation']}")
        log.append(f"   ES: {out['explanation_es']}")
    return {"n": len(res), "explained": sum(r["explained by Nemotron"] for r in res),
            "spanish": sum(r["Spanish"] for r in res),
            "deadlines": sum(r["deadline right"] and r["months right"] for r in res)}


def _demo(day):
    conn = db.connect(_tmp / f"demo-{day}.sqlite3")
    with llm.paused():
        demo.reset(conn, date(2026, 10, 1))
    return conn


def eval_plans(log):
    res = []
    for day in (date(2026, 10, 20), date(2026, 10, 27)):
        conn = _demo(day)
        with llm.paused():
            demo.advance_to(conn, day)
        person = demo.demo_person(conn)
        ctx = S.context(conn, person, day)
        p = planner.make_plan(conn, ctx, day, fresh=True)
        res.append(p)
        log.append(f"plan {day}: by {p['read_by']}, {p['hours_total']}h now + {p['planned_hours']}h planned, "
                   f"reaches 80: {p.get('reaches_hours')}; rules dropped {len(p.get('dropped', []))}, "
                   f"changed {len(p.get('adjusted', []))}: {p.get('dropped', [])} {p.get('adjusted', [])}")
        log.append(f"   message: {p.get('message')}")
    return {"n": len(res), "by_ai": sum(p["read_by"] == "ai" for p in res),
            "reach": sum(bool(p.get("reaches_hours")) for p in res),
            "dropped": sum(len(p.get("dropped", [])) for p in res),
            "adjusted": sum(len(p.get("adjusted", [])) for p in res)}


def eval_brain(log):
    days = []
    for day in BRAIN_DAYS:
        conn = _demo(day)
        person = demo.demo_person(conn)
        if day == date(2026, 11, 25):
            with llm.paused():
                demo.advance_to(conn, date(2026, 11, 24))
            demo.receive_letter(conn)   # moves to Nov 25, the brain thinks with the letter in hand
        else:
            with llm.paused():
                demo.advance_to(conn, date.fromordinal(day.toordinal() - 1))
            demo.advance_to(conn, day)
        rows = conn.execute("SELECT decision, detail FROM agent_log WHERE person_id=? AND day=? AND step='brain'",
                            (person["id"], day.isoformat())).fetchall()
        kinds = [r["decision"] for r in rows]
        sent = conn.execute("SELECT title, body, reason FROM alert WHERE person_id=? AND day=? AND status='sent'",
                            (person["id"], day.isoformat())).fetchall()
        days.append({"thought": "think" in kinds, "sent_by_ai": sum("NVIDIA Nemotron" in (a["reason"] or "") for a in sent),
                     "refused": kinds.count("rules") + kinds.count("refused"), "held": kinds.count("hold")})
        log.append(f"brain {day}:")
        log += [f"   [{r['decision']}] {r['detail']}" for r in rows]
        log += [f"   SENT: {a['title']} - {a['body']}" for a in sent]
    return {"n": len(days), "decided": sum(d["thought"] for d in days), "sent_by_ai": sum(d["sent_by_ai"] for d in days),
            "refused": sum(d["refused"] for d in days), "held": sum(d["held"] for d in days)}


# ------------------------------------------------------------------ report

def main():
    if llm.provider() != "nvidia":
        sys.exit("Set NEBIUS_API_KEY first (see env.example).")
    log, out = [], {}
    for name, fn in (("photos", eval_photos), ("video", eval_video), ("notes", eval_notes),
                     ("letters", eval_letters), ("plans", eval_plans), ("brain", eval_brain)):
        print(f"... {name}", file=sys.stderr)
        with Meter() as m:
            out[name] = fn(log)
        out[name]["meter"] = m
    fmt = lambda t: f"{t['correct']} correct, {t['asked']} asked, **{t['wrong']} wrong**"
    cost = lambda m: f"{m.calls} calls, {m.secs / max(m.calls, 1):.1f} s each, ${m.usd:.4f}"
    p, v, n, le, pl, b = (out[k] for k in ("photos", "video", "notes", "letters", "plans", "brain"))
    rows = [
        ("Read 7 photos: pay stub, app screens, handwritten note, sign-in sheets (values)", "Super VL",
         f"Model alone: {p['model']} right. After the rules: {fmt(p['final'])}", cost(p["meter"])),
        (f"Read an app-history screen recording ({v['frames']} frames kept, 8 shifts)", "Super VL",
         f"{v['found']}/8 shifts found. Values (hours and pay): {fmt(v['final'])}; {v['invented']} invented rows", cost(v["meter"])),
        (f"Understand {n['n']} typed or spoken notes, English and Spanish", "Nano",
         f"Whole note right: rules alone {n['rules_whole']}/{n['n']}, with Nemotron {n['ai_whole']}/{n['n']}. "
         f"Values: {fmt(n['final'])}", cost(n["meter"])),
        (f"Explain {le['n']} state letters in plain English and Spanish", "Ultra",
         f"{le['explained']}/{le['n']} explanations passed the no-invented-date check, {le['spanish']}/{le['n']} in Spanish; "
         f"deadline and months (rules) {le['deadlines']}/{le['n']}", cost(le["meter"])),
        (f"Plan the rest of the month ({pl['n']} days)", "Super + Nano",
         f"{pl['reach']}/{pl['n']} plans reach 80 h; the rules removed {pl['dropped']} lines and made {pl['adjusted']} fixes",
         cost(pl["meter"])),
        (f"Decide each day what to tell her ({b['n']} days, tool calling)", "Super",
         f"{b['decided']}/{b['n']} days decided; {b['sent_by_ai']} messages written by Nemotron sent, "
         f"{b['refused']} refused or replaced by the rules, {b['held']} findings held for later", cost(b["meter"])),
    ]
    total = sum(o["meter"].usd for o in out.values())
    calls = sum(o["meter"].calls for o in out.values())
    print(f"Run on {date.today().isoformat()} against Nebius Token Factory: {calls} model calls, ${total:.3f} in total.\n")
    print("| Job | NVIDIA model | Result | Calls, speed, cost |")
    print("|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("\nDetails\n")
    for line in log:
        print("    " + line)
    print(json.dumps({k: {kk: vv for kk, vv in o.items() if kk != "meter"} for k, o in out.items()}, default=str),
          file=sys.stderr)


if __name__ == "__main__":
    main()
