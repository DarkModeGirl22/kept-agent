"""The NVIDIA layer: model routing, spending limits, and Kept's rules checking every answer.
No network: each test stands in for Nemotron with a fixed answer."""

import io
import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

os.environ.setdefault("KEPT_DATA_DIR", tempfile.mkdtemp())

from kept import db, demo  # noqa: E402
from kept import services as S  # noqa: E402
from kept.agent import planner  # noqa: E402
from kept.capture import extract, letters, llm, textlog, vision  # noqa: E402
from kept.web.app import create_app  # noqa: E402
from tests import fakevision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
ACTS = [
    {"id": 1, "name": "Dashly", "type": "gig", "keywords": "dashly"},
    {"id": 2, "name": "Corner Market", "type": "cash", "keywords": "store,corner market,tienda"},
]


class Client(unittest.TestCase):
    def test_json_repair(self):
        self.assertEqual(llm._parse_json('{"a": [{"b": 1}]'), {"a": [{"b": 1}]})
        self.assertEqual(llm._parse_json('Sure! ```json\n{"x": 2}\n```'), {"x": 2})
        self.assertEqual(llm._parse_json("<think>hmm</think>{\"y\": 3}"), {"y": 3})
        self.assertIsNone(llm._parse_json("no json here"))

    def test_spending_limit_stops_calls(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "u.json"
            f.write_text(json.dumps({"days": {date.today().isoformat(): {"usd": 5.0, "calls": 9}}, "total_usd": 5.0}))
            with mock.patch.dict(os.environ, {"NEBIUS_API_KEY": "k", "KEPT_AI_USAGE_FILE": str(f)}), \
                    mock.patch.object(llm, "DAILY_USD", 1.0):
                self.assertFalse(llm.available())
                with mock.patch("urllib.request.urlopen") as up:
                    self.assertIsNone(llm.chat_json("s", "u"))
                    up.assert_not_called()

    def test_visitor_guard_and_switch(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"NEBIUS_API_KEY": "k", "KEPT_AI_USAGE_FILE": str(Path(d) / "u.json")}):
            llm.set_request(off=False, guard=lambda: False)
            with mock.patch("urllib.request.urlopen") as up:
                self.assertIsNone(llm.chat_json("s", "u"))
                up.assert_not_called()
            llm.set_request(off=True)
            self.assertFalse(llm.available())
            llm.set_request(off=False, guard=None)

    def test_model_routing(self):
        self.assertIn("Super-VL", llm.MODELS["vision"])
        self.assertIn("Nano", llm.MODELS["fast"])
        self.assertIn("Ultra", llm.MODELS["deep"])
        self.assertTrue(all(m.startswith("nvidia/") for m in llm.MODELS.values()))


class VisionMerge(unittest.TestCase):
    """Super VL's values against Kept's rules re-reading the same transcription."""
    TEXT = (FIX / "proof_dashly_oct16.txt").read_text()

    def run_extract(self, ai):
        return extract.extract(self.TEXT, date(2026, 10, 20), ACTS, ai=ai)

    def test_agreement_is_trusted(self):
        r = self.run_extract({"period_start": "2026-10-16", "period_end": "2026-10-16", "hours": 3.5, "income": 64})
        self.assertEqual(r["confidence"]["income"], extract.VERIFIED)
        self.assertEqual(r["read_by"], "ai")
        self.assertEqual(r["ask"], [])

    def test_disagreement_is_asked(self):
        # the model says 46 (the delivery pay line), the rules say 64 (total earnings): both are written
        r = self.run_extract({"income": 46})
        self.assertIn("income", r["ask"])

    def test_invented_value_is_ignored(self):
        r = self.run_extract({"income": 99})
        self.assertEqual(r["fields"]["income"], 64.0)

    def test_handwritten_hours_written_before_the_word(self):
        text = (FIX / "proof_corner_note_oct8.txt").read_text() if (FIX / "proof_corner_note_oct8.txt").exists() else \
            "Oct 8, 2026\nKeisha Moore worked at\nCorner Market on Thursday\nOct 8 from 12 to 3 pm.\n3 hours. Paid $36 cash."
        r = extract.extract(text, date(2026, 10, 20), ACTS, ai={"hours": 3, "income": 36})
        self.assertEqual(r["fields"]["hours"], 3.0)
        self.assertEqual(r["fields"]["income"], 36.0)
        self.assertNotIn("hours", r["ask"])


class History(unittest.TestCase):
    def read(self, rows, app="Dashly"):
        with mock.patch.object(llm, "vision_json", return_value={"app_name": app, "rows": rows}):
            return vision.read_history(["f1.jpg"], date(2026, 10, 20))

    def test_rows_are_checked(self):
        r = self.read([
            {"date": "2026-10-16", "hours": 3.5, "pay": 64, "quote": "Fri, Oct 16 Active time 3h 30m $64.00"},
            {"date": "2026-10-16", "hours": 3.5, "pay": 64, "quote": "Fri, Oct 16 Active time 3h 30m $64.00"},
            {"date": "2026-10-25", "hours": 4, "pay": 50, "quote": "Sun, Oct 25 4h 00m $50.00"},
            {"date": "2026-10-13", "hours": 4, "pay": 80, "quote": "Tue, Oct 13 Active time 4h 00m $50.00"},
            {"date": "2026-10-09", "hours": 3.5, "pay": 44, "quote": "Fri, Oct 9 Active time 3h 30m $44.00"},
        ])
        rows = {x["start"]: x for x in r["rows"]}
        self.assertEqual(len(r["rows"]), 4)                        # the duplicate frame row counted once
        self.assertTrue(rows["2026-10-16"]["ok"])                  # 3.5 written as 3h 30m
        self.assertIn("date in the future", rows["2026-10-25"]["problems"])
        self.assertIn("numbers not in the row it quoted", rows["2026-10-13"]["problems"])
        self.assertTrue(rows["2026-10-09"]["ok"])

    def test_batches_of_eight(self):
        calls = []

        def fake(prompt, frames, **kw):
            calls.append(len(frames))
            return {"app_name": "Dashly", "rows": []}
        with mock.patch.object(llm, "vision_json", side_effect=fake):
            vision.read_history([f"f{i}.jpg" for i in range(13)], date(2026, 10, 20))
        self.assertEqual(calls, [8, 5])


class Words(unittest.TestCase):
    def test_amount_said_in_words(self):
        self.assertTrue(textlog._money_said(38, "made thirty eight"))
        self.assertTrue(textlog._money_said(38, "gané treinta y ocho"))
        self.assertTrue(textlog._money_said(48, "me pagaron 48"))
        self.assertFalse(textlog._money_said(40, "made thirty eight"))

    def test_spanish_voice_note_with_ai(self):
        fake = {"date": "2026-10-19", "hours": 4, "income": 48, "activity_id": 2}
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "chat_json", return_value=fake):
            r = textlog.parse("ayer trabajé en la tienda cuatro horas, me pagaron cuarenta y ocho", date(2026, 10, 20), ACTS)
        self.assertEqual((r["date"], r["hours"], r["income"], r["activity_id"]), ("2026-10-19", 4.0, 48.0, 2))


class LetterExplain(unittest.TestCase):
    TEXT = (FIX / "sample_state_letter.txt").read_text()

    def test_spanish_kept_when_clean(self):
        fake = {"explanation_en": "The state wants proof of your hours for October and November 2026. Answer by December 23.",
                "explanation_es": "El estado pide prueba de sus horas de octubre y noviembre de 2026. Responda antes del 23 de diciembre."}
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "chat_json", return_value=fake):
            r = letters.read_letter(self.TEXT, date(2026, 11, 25))
        self.assertEqual(r["read_by"], "ai")
        self.assertIn("octubre", r["explanation_es"])
        self.assertEqual(r["respond_by"], date(2026, 12, 23))   # the deadline still comes from the rules

    def test_invented_date_refused(self):
        fake = {"explanation_en": "Send your proof by January 15, or your coverage ends on January 31.",
                "explanation_es": "Envíe prueba antes del 15 de enero."}
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "chat_json", return_value=fake):
            r = letters.read_letter(self.TEXT, date(2026, 11, 25))
        self.assertEqual(r["read_by"], "rules")
        self.assertIsNone(r["explanation_es"])


class Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = db.connect(Path(tempfile.mkdtemp()) / "p.sqlite3")
        demo.reset(cls.conn)
        cls.person = demo.demo_person(cls.conn)
        cls.today = db.today(cls.conn)            # Tue Oct 20
        cls.ctx = S.context(cls.conn, cls.person, cls.today)
        cls.ids = {a["name"]: a["id"] for a in cls.ctx.activities}

    def verify(self, items):
        s = self.ctx.summary("2026-10")
        return planner._verify({"items": items}, self.ctx, self.today, date(2026, 10, 31), s)

    def test_rules_fix_the_models_plan(self):
        d, pantry, school = self.ids["Dashly"], self.ids["Riverside Community Pantry"], self.ids["Adult Learning Center"]
        v = self.verify([
            {"date": "2026-10-21", "activity_id": school, "hours": 3},     # class isn't on Wednesdays
            {"date": "2026-10-19", "activity_id": d, "hours": 4},          # already past
            {"date": "2026-10-22", "activity_id": d, "hours": 14},         # impossible shift
            {"date": "2026-10-22", "activity_id": 999, "hours": 4},        # not her activity
        ] + [{"date": f"2026-10-{n}", "activity_id": d, "hours": 4} for n in range(21, 31)])
        reasons = " ".join(w for _, w in v["dropped"])
        for r in ("only happens on Tuesdays", "not a day left", "not a believable shift", "not one of her activities"):
            self.assertIn(r, reasons)
        planned = {(i["date"], i["activity"]) for i in v["items"]}
        self.assertIn(("2026-10-24", "Riverside Community Pantry"), planned)   # her Saturday, added if missing
        self.assertIn(("2026-10-27", "Adult Learning Center"), planned)
        days = sorted(date.fromisoformat(i["date"]) for i in v["items"])
        run = longest = 1
        for a, b in zip(days, days[1:]):
            run = run + 1 if (b - a).days == 1 else 1
            longest = max(longest, run)
        self.assertLess(longest, 7)                                             # a free day in any 7
        self.assertTrue(v["reaches_hours"])
        self.assertLessEqual(v["total_after"], 84)                              # just enough
        del pantry

    def test_without_ai_the_rules_plan(self):
        with mock.patch.object(llm, "available", return_value=False):
            p = planner.make_plan(self.conn, self.ctx, self.today, fresh=True)
        self.assertEqual(p["read_by"], "rules")
        self.assertTrue(all(i["usual"] for i in p["items"]))

    def test_message_numbers_must_come_from_the_plan(self):
        s = self.ctx.summary("2026-10")
        v = {"items": [{"date": "2026-10-24", "hours": 4.0, "pay": None}], "planned_hours": 4.0, "planned_pay": 0,
             "total_after": s.hours_total + 4}
        self.assertTrue(planner._message_ok("Your pantry shift on Oct 24 adds 4 hours. Keep going!", v, s))
        self.assertFalse(planner._message_ok("Work 12 more hours on Oct 24 and you'll make $900.", v, s))
        # found by tools/eval.py: an empty "you're doing great" while the month still needs hours
        self.assertFalse(planner._message_ok("You're doing great! You'll work at the pantry. Keep up the good work!", v, s))
        self.assertFalse(planner._message_ok("You'll work at the pantry on Saturday and at the store.", v, s))


class Brain(unittest.TestCase):
    """Nemotron as the agent's brain, and the rules checking its decision."""

    def cands(self):
        from kept.agent.watchdog import Candidate
        return [Candidate("letter:1:7", "letter", 0, "7 days left to answer the state", "Deadline: Dec 23.", "/letters/1",
                          "Open the letter", urgent=True, cooldown_days=0),
                Candidate("month_risk:2026-10:20", "month", 2, "October needs 30.5 more hours",
                          "You're at 49.5 of 80 hours with 12 days left.", "/plan", "Plan the rest"),
                Candidate("proof:2026-43", "proof", 5, "Some work has no proof yet", "Dashly (5).", "/add", "Add proof")]

    RESULTS = {"get_situation": {"hours_logged": 49.5, "hours_left": 30.5, "days_left_in_month": 12},
               "get_week": {"next_7_days_usual": [{"date": "2026-10-24", "activity": "Riverside Community Pantry",
                                                    "usual_hours": 4.0}]}}

    def test_rules_check_the_decision(self):
        from kept.agent import brain
        decision = {
            "messages": [
                {"about": "month_risk:2026-10:20", "title": "October: 30.5 hours to go",
                 "body": "You have 49.5 of 80 hours and 12 days left. Your Saturday pantry shift on Oct 24 adds 4.",
                 "action": "/plan", "why": "She can still fix October."},
                {"about": "proof:2026-43", "title": "Proof", "body": "Upload 9 Dashly screenshots to earn $900.",
                 "why": "x"},
                {"about": "new", "title": "Ask Diane on Saturday", "body": "On Oct 24 at the pantry, ask Diane to sign the October page.",
                 "action": "/folder", "why": "She will be there anyway."},
                {"about": "new", "title": "Another idea", "body": "Rest on Sunday.", "why": "y"},
                {"about": "nonsense", "title": "?", "body": "?", "why": "z"},
            ],
            "hold": [{"about": "letter:1:7", "why": "Already told her yesterday."}],
        }
        out, notes = brain.check_decision(decision, self.cands(), self.RESULTS)
        by_key = {c.key: c for c in out}
        self.assertEqual(by_key["month_risk:2026-10:20"].title, "October: 30.5 hours to go")   # Nemotron's words
        self.assertEqual(by_key["month_risk:2026-10:20"].by, "nemotron")
        self.assertEqual(by_key["proof:2026-43"].title, "Some work has no proof yet")         # invented $900: rules' text
        self.assertIn("letter:1:7", by_key)                                                    # urgent: sent anyway
        self.assertEqual(sum(1 for c in out if c.kind == "coach"), 1)                          # one idea of its own
        text = " ".join(t for _, t in notes)
        self.assertIn("urgent: sent anyway", text)
        self.assertIn("only one idea", text)
        self.assertIn("isn't a finding today", text)

    def test_no_false_reassurance(self):
        from kept.agent import brain
        res = dict(self.RESULTS, get_situation={"status": "At risk", "hours_logged": 49.5})
        d = {"messages": [{"about": "month_risk:2026-10:20", "title": "Great work",
                           "body": "You have 49.5 hours, you're on track!", "why": "x"}]}
        out, notes = brain.check_decision(d, self.cands(), res)
        month = next(c for c in out if c.key == "month_risk:2026-10:20")
        self.assertEqual(month.title, "October needs 30.5 more hours")   # the rules' honest text
        self.assertIn("at risk", " ".join(t for _, t in notes))

    def test_private_words_refused(self):
        from kept.agent import brain
        d = {"messages": [{"about": "new", "title": "Your pregnancy", "body": "Because you are pregnant, rest.", "why": "x"}]}
        out, notes = brain.check_decision(d, [], self.RESULTS)
        self.assertEqual(out, [])
        self.assertIn("private", notes[0][1])

    def test_tool_loop_and_log(self):
        conn = db.connect(Path(tempfile.mkdtemp()) / "b.sqlite3")
        with llm.paused():
            demo.reset(conn)
        person = demo.demo_person(conn)
        replies = iter([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "get_rule_findings", "arguments": "{}"}},
                {"id": "2", "type": "function", "function": {"name": "get_week", "arguments": "{}"}}]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "3", "type": "function", "function": {"name": "decide", "arguments": json.dumps({
                    "messages": [{"about": "new", "title": "Saturday at the pantry",
                                  "body": "You'll be at the pantry on Oct 24: ask for the October page to be signed.",
                                  "action": "/folder", "why": "She'll be there."}],
                    "summary": "On track if she keeps her routine."})}}]},
        ])
        from kept.agent import watchdog
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "provider", return_value="nvidia"), \
                mock.patch.object(llm, "chat_tools", side_effect=lambda *a, **k: next(replies)):
            watchdog.rerun_today(conn, person, db.today(conn))
        logs = conn.execute("SELECT decision, detail FROM agent_log WHERE step='brain'").fetchall()
        self.assertIn("get_rule_findings, get_week", logs[0]["detail"])
        alert = conn.execute("SELECT * FROM alert WHERE kind='coach'").fetchone()
        self.assertEqual(alert["title"], "Saturday at the pantry")
        self.assertTrue(alert["reason"].startswith("Chosen and written by NVIDIA Nemotron"))

    def test_no_decision_means_rules(self):
        conn = db.connect(Path(tempfile.mkdtemp()) / "c.sqlite3")
        with llm.paused():
            demo.reset(conn)
        person = demo.demo_person(conn)
        from kept.agent import watchdog
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(llm, "provider", return_value="nvidia"), \
                mock.patch.object(llm, "chat_tools", return_value=None):
            watchdog.rerun_today(conn, person, db.today(conn))
        self.assertTrue(conn.execute("SELECT 1 FROM agent_log WHERE step='brain' AND decision='fallback'").fetchone())
        self.assertTrue(conn.execute("SELECT 1 FROM alert WHERE day=?", (db.today(conn).isoformat(),)).fetchone())


class Screens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        llm.set_enabled(False)
        cls.vision = fakevision.install()
        cls.conn = db.connect(Path(tempfile.mkdtemp()) / "w.sqlite3")
        cls.app = create_app(cls.conn)
        cls.app.testing = True
        cls.c = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        llm.set_enabled(True)
        cls.vision.stop()

    def test_video_flow(self):
        self.c.post("/demo/start")
        pid = self.conn.execute("SELECT value FROM settings WHERE key='active_person'").fetchone()["value"]
        dashly = self.conn.execute("SELECT id FROM activity WHERE name='Dashly' AND person_id=?", (pid,)).fetchone()["id"]
        frame_dir = Path(tempfile.mkdtemp())
        frames = []
        for i in range(2):
            p = frame_dir / f"f{i}.jpg"
            p.write_bytes((ROOT / "samples" / "sample_dashly_history_poster.jpg").read_bytes())
            frames.append(p)
        rows = [{"start": "2026-10-06", "end": "2026-10-06", "hours": 4.0, "pay": 52.0, "quote": "Tue, Oct 6 4h 00m $52.00",
                 "ok": True, "problems": []},
                {"start": "2026-10-13", "end": "2026-10-13", "hours": 4.0, "pay": 50.0, "quote": "Tue, Oct 13 4h 00m $50.00",
                 "ok": True, "problems": []},
                {"start": "2026-09-29", "end": "2026-09-29", "hours": 3.0, "pay": 41.0, "quote": "Tue, Sep 29 3h 00m $41.00",
                 "ok": True, "problems": []},
                {"start": "2026-10-25", "end": "2026-10-25", "hours": 4.0, "pay": 50.0, "quote": "?",
                 "ok": False, "problems": ["date in the future"]}]
        with mock.patch.object(vision, "sample_frames", return_value=frames), \
                mock.patch.object(vision, "read_history", return_value={"app_name": "Dashly", "rows": rows}):
            data = {"video_file": (io.BytesIO(b"fake video bytes"), "history.mp4")}
            r = self.c.post("/add/video", data=data, content_type="multipart/form-data")
        self.assertIn("/review/history/", r.headers["Location"])
        html = self.c.get(r.headers["Location"]).get_data(as_text=True)
        self.assertIn("Matches what you logged", html)
        self.assertIn("New: you hadn't logged this day", html)
        self.assertIn("Not added: date in the future", html)
        self.assertIn("/frame/1", html)
        doc_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
        self.assertEqual(self.c.get(f"/files/{doc_id}/frame/0").status_code, 200)
        before = self.conn.execute("SELECT COUNT(*) n FROM entry WHERE activity_id=? AND superseded_by IS NULL", (dashly,)).fetchone()["n"]
        html = self.c.post(r.headers["Location"], data={"action": "save", "activity_id": dashly, "row": ["0", "1", "2", "3"]},
                           follow_redirects=True).get_data(as_text=True)
        self.assertIn("Saved 3 shifts with the video as proof", html)
        self.assertIn("It replaces 2 lines you typed", html)
        after = self.conn.execute("SELECT COUNT(*) n FROM entry WHERE activity_id=? AND superseded_by IS NULL", (dashly,)).fetchone()["n"]
        self.assertEqual(after, before + 1)            # two replaced, one new (Sep 29), the future one refused

    def test_plan_ai_and_demo_pages(self):
        self.c.post("/demo/start")
        html = self.c.get("/plan").get_data(as_text=True)
        self.assertIn("Plan the rest of October", html)
        self.assertIn("Kept's rules: your usual days left", html)
        html = self.c.get("/ai").get_data(as_text=True)
        self.assertIn("Nemotron 3.5 Super VL", html)
        self.assertIn("Nemotron Nano Omni", html)
        html = self.c.get("/demo").get_data(as_text=True)
        self.assertIn("Handwritten note", html)
        self.assertEqual(self.c.get("/samples/sample_renewal_letter.pdf").status_code, 200)
        self.assertEqual(self.c.get("/samples/../kept/db.py").status_code, 404)

    def test_add_page_offers_video_and_voice(self):
        self.c.post("/demo/start")
        html = self.c.get("/add").get_data(as_text=True)
        self.assertIn('name="video_file"', html)
        self.assertIn("SpeechRecognition", html)
        self.assertIn("es-US", html)


if __name__ == "__main__":
    unittest.main()
