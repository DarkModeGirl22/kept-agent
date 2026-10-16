import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

from kept.capture import llm

_tmp = tempfile.mkdtemp()
os.environ["KEPT_DATA_DIR"] = _tmp

from kept import db, demo, services as S  # noqa: E402
from kept.agent import watchdog  # noqa: E402
from kept.capture import textlog  # noqa: E402


def fresh():
    conn = db.connect(Path(tempfile.mkdtemp()) / "t.sqlite3")
    return conn


class DemoScript(unittest.TestCase):
    def test_every_demo_text_parses(self):
        acts = [dict(a, id=i) for i, a in enumerate(demo.ACTIVITIES, 1)]
        for d, msg in demo.SCRIPT:
            r = textlog.parse(msg, date.fromisoformat(d), acts)
            self.assertIsNotNone(r["activity_id"], msg)
            self.assertIsNotNone(r["hours"], msg)
            self.assertEqual(r["date"], d, msg)


class AgentStory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        llm.set_enabled(False)
        cls.conn = fresh()
        demo.reset(cls.conn)  # runs the agent from Oct 1 to Oct 20

    @classmethod
    def tearDownClass(cls):
        llm.set_enabled(True)

    def alerts(self, day=None):
        q = "SELECT * FROM alert"
        args = ()
        if day:
            q += " WHERE day=?"
            args = (day,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY id", args)]

    def test_1_oct20_at_risk_with_concrete_fixes(self):
        p = db.get_person(self.conn)
        s = S.context(self.conn, p).summary("2026-10")
        self.assertEqual(s.status, "at_risk")
        a = [x for x in self.alerts("2026-10-20") if x["kind"] == "month"]
        self.assertEqual(len(a), 1)
        self.assertIn("Oct 24, Oct 31", a[0]["body"])   # her Saturday pantry shifts
        self.assertIn("Oct 27", a[0]["body"])           # her Tuesday class
        self.assertEqual(a[0]["status"], "sent")

    def test_2_no_september_close_before_tracking(self):
        self.assertFalse([x for x in self.alerts() if x["title"].startswith("September")])

    def test_3_every_decision_is_logged(self):
        steps = {r["step"] for r in self.conn.execute("SELECT step FROM agent_log WHERE day='2026-10-20'")}
        for s in ("letters", "exemption", "month", "check", "close", "proof", "nudge", "reconfirm", "deliver"):
            self.assertIn(s, steps)

    def test_4_october_met_and_page_prepared_on_nov1(self):
        demo.advance_to(self.conn, date(2026, 11, 1))
        p = db.get_person(self.conn)
        s = S.context(self.conn, p).summary("2026-10")
        self.assertEqual(s.hours_total, 81)
        self.assertEqual(s.status, "met")
        so = self.conn.execute("SELECT * FROM signoff").fetchone()
        self.assertEqual((so["month"], so["hours"], so["status"]), ("2026-10", 20, "prepared"))
        titles = [x["title"] for x in self.alerts("2026-11-01")]
        self.assertIn("October is done", titles)
        self.assertIn("A page to get signed", titles)

    def test_5_letter_is_urgent_with_deadline(self):
        lid = demo.receive_letter(self.conn)
        L = self.conn.execute("SELECT * FROM letter WHERE id=?", (lid,)).fetchone()
        if L["respond_by"]:  # read from the photo by Tesseract
            self.assertEqual(L["deadline"], "2026-12-23")
        a = [x for x in self.alerts("2026-11-25") if x["kind"] == "letter"]
        self.assertTrue(a and a[0]["urgent"] == 1 and a[0]["status"] == "sent")

    def test_6_statement_replaces_typed_entries(self):
        p = db.get_person(self.conn)
        doc_id, err = S.store_upload(self.conn, p["id"], demo.SAMPLES / "sample_gig_statement.png", "gig.png")
        self.assertIsNone(err)
        dashly = self.conn.execute("SELECT id FROM activity WHERE name='Dashly' AND person_id=?", (p["id"],)).fetchone()["id"]
        before = S.context(self.conn, p).summary("2026-10")
        res = S.confirm_document(self.conn, p["id"], doc_id, dashly, date(2026, 10, 5), date(2026, 10, 11), 7.5, 96.0)
        self.assertEqual(len(res["replaced"]), 2)  # Oct 6 and Oct 9 texts
        after = S.context(self.conn, p).summary("2026-10")
        self.assertEqual(after.hours_total, before.hours_total)   # not counted twice
        self.assertEqual(after.hours_proven, before.hours_proven + 7.5)
        again, err = S.store_upload(self.conn, p["id"], demo.SAMPLES / "sample_gig_statement.png", "gig.png")
        self.assertEqual(err, "duplicate")

    def test_7_signed_volunteer_sheet_proves_the_month(self):
        p = db.get_person(self.conn)
        doc_id, _ = S.store_upload(self.conn, p["id"], demo.SAMPLES / "sample_volunteer_sheet.png", "vol.png")
        pantry = self.conn.execute("SELECT id FROM activity WHERE type='volunteer' AND person_id=?", (p["id"],)).fetchone()["id"]
        S.confirm_document(self.conn, p["id"], doc_id, pantry, date(2026, 10, 1), date(2026, 10, 31), 20, None)
        so = self.conn.execute("SELECT status FROM signoff WHERE month='2026-10'").fetchone()
        self.assertEqual(so["status"], "signed")
        s = S.context(self.conn, p).summary("2026-10")
        self.assertEqual(s.hours_total, 81)


class Budget(unittest.TestCase):
    def test_non_urgent_messages_are_rationed_and_logged(self):
        conn = fresh()
        llm.set_enabled(False)
        demo.reset(conn, date(2026, 10, 1))
        p = db.get_person(conn)
        conn.execute("UPDATE person SET max_alerts_week=1 WHERE id=?", (p["id"],))
        conn.commit()
        demo.advance_to(conn, date(2026, 10, 20))
        weeks = {}
        for r in conn.execute("SELECT day FROM alert WHERE status='sent' AND urgent=0"):
            wk = date.fromisoformat(r["day"]).isocalendar()[1]
            weeks[wk] = weeks.get(wk, 0) + 1
        self.assertTrue(all(v <= 1 for v in weeks.values()), weeks)
        held = conn.execute("SELECT reason FROM alert WHERE status='held'").fetchall()
        self.assertTrue(held and "Weekly limit" in held[0]["reason"])
        llm.set_enabled(True)


class Modes(unittest.TestCase):
    def test_exempt_person_gets_no_hour_alerts(self):
        conn = fresh()
        llm.set_enabled(False)
        demo.reset(conn, date(2026, 10, 1))
        p = db.get_person(conn)
        conn.execute("UPDATE person SET answers=? WHERE id=?",
                     (json.dumps({"caregiver_child": True, "child_birth_date": "2020-05-01"}), p["id"]))
        conn.commit()
        p = db.get_person(conn)
        self.assertEqual(S.mode(S.context(conn, p)), "exempt")
        demo.advance_to(conn, date(2026, 10, 20))
        self.assertFalse(conn.execute("SELECT 1 FROM alert WHERE kind='month' AND day>'2026-10-01'").fetchall())
        llm.set_enabled(True)


if __name__ == "__main__":
    unittest.main()
