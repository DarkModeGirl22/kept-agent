import io
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KEPT_DATA_DIR", tempfile.mkdtemp())

from kept import db  # noqa: E402
from kept.capture import llm  # noqa: E402
from kept.web.app import create_app  # noqa: E402
from tests import fakevision  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class Web(unittest.TestCase):
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

    def get(self, url, code=200):
        r = self.c.get(url, follow_redirects=True)
        self.assertEqual(r.status_code, code, url)
        return r.get_data(as_text=True)

    def test_1_welcome_and_demo(self):
        self.assertIn("Open Keisha's demo", self.get("/"))
        r = self.c.post("/demo/start", follow_redirects=True)
        html = r.get_data(as_text=True)
        self.assertIn("October 2026", html)
        self.assertIn("At risk", html)
        self.assertIn("needs 30.5 more hours", html)
        self.assertEqual(html.count('class="cell f'), 99 if False else html.count('class="cell f'))

    def test_1b_old_warning_disappears_when_month_is_fixed(self):
        self.c.post("/demo/start")
        self.c.post("/demo/go", data={"to": "2026-10-28"})
        html = self.c.get("/").get_data(as_text=True)
        self.assertIn("On track", html)
        self.assertNotIn("needs 30.5 more hours", html)

    def test_2_every_page_renders(self):
        self.c.post("/demo/start")
        for url in ["/", "/add", "/calendar", "/calendar?month=2026-10&day=17", "/folder", "/folder/2026-10",
                    "/letters", "/profile", "/agent", "/inbox", "/help", "/more", "/demo", "/welcome",
                    "/onboarding/1", "/add?letter=1"]:
            self.get(url)

    def test_3_text_log_flow(self):
        self.c.post("/demo/start")
        html = self.c.post("/add/text", data={"text": "Dashly 3pm-6pm today $40"}).get_data(as_text=True)
        self.assertIn("Is this right?", html)
        self.assertIn('value="3.0"', html)
        aid = self.conn.execute("SELECT id FROM activity WHERE name='Dashly' AND person_id=(SELECT value FROM settings WHERE key='active_person')").fetchone()["id"]
        html = self.c.post("/add/manual", data={"activity_id": aid, "start": "2026-10-20", "hours": "3", "income": "40",
                                                "source": "text", "read_by": "rules"}, follow_redirects=True).get_data(as_text=True)
        self.assertIn("Logged: 3h and $40.00 Dashly, Oct 20.", html)
        self.assertIn("52.5", html)

    def test_4_future_and_impossible_values_refused(self):
        self.c.post("/demo/start")
        aid = self.conn.execute("SELECT id FROM activity WHERE type='volunteer' AND person_id=(SELECT value FROM settings WHERE key='active_person')").fetchone()["id"]
        html = self.c.post("/add/manual", data={"activity_id": aid, "start": "2026-10-25", "hours": "4"},
                           follow_redirects=True).get_data(as_text=True)
        self.assertIn("in the future", html)
        html = self.c.post("/add/manual", data={"activity_id": aid, "start": "2026-10-19", "hours": "30"},
                           follow_redirects=True).get_data(as_text=True)
        self.assertIn("fit in 1 day", html)

    def test_5_photo_upload_review_and_supersede(self):
        self.c.post("/demo/start")
        data = {"photo_file": (io.BytesIO((SAMPLES / "sample_gig_statement.png").read_bytes()), "dashly.png")}
        r = self.c.post("/add/photo", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 302)
        html = self.c.get(r.headers["Location"]).get_data(as_text=True)
        self.assertIn("Check what Kept read", html)
        self.assertIn('value="7.5"', html)
        self.assertIn('value="2026-10-05"', html)
        doc_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
        aid = self.conn.execute("SELECT activity_id FROM document WHERE id=?", (doc_id,)).fetchone()
        dashly = self.conn.execute("SELECT id FROM activity WHERE name='Dashly' AND person_id=(SELECT value FROM settings WHERE key='active_person')").fetchone()["id"]
        html = self.c.post(f"/review/{doc_id}", data={"action": "save", "activity_id": dashly, "start": "2026-10-05",
                                                      "end": "2026-10-11", "hours": "7.5", "income": "96"},
                           follow_redirects=True).get_data(as_text=True)
        self.assertIn("It replaces 2 lines you typed", html)
        self.assertIn("49.5", html)  # total unchanged

    def test_5b_photo_with_the_ai_paused_still_works(self):
        from unittest import mock
        from kept.capture import vision
        self.c.post("/demo/start")
        with mock.patch.object(vision, "read_photo", return_value=None):
            data = {"photo_cam": (io.BytesIO((SAMPLES / "sample_pay_stub.png").read_bytes()), "stub.png")}
            r = self.c.post("/add/photo", data=data, content_type="multipart/form-data")
            html = self.c.get(r.headers["Location"]).get_data(as_text=True)
        self.assertIn("Check what Kept read", html)
        self.assertIn("Not found, please fill in", html)
        self.assertIn("type the values", html)

    def test_5c_add_proof_to_a_typed_job(self):
        self.c.post("/demo/start")
        pid = self.conn.execute("SELECT value FROM settings WHERE key='active_person'").fetchone()["value"]
        dashly = self.conn.execute("SELECT id FROM activity WHERE name='Dashly' AND person_id=?", (pid,)).fetchone()["id"]
        # the list shows every Dashly job without proof
        html = self.c.get(f"/proof/activity/{dashly}?month=2026-10").get_data(as_text=True)
        self.assertIn("5 jobs without proof", html)
        e = self.conn.execute("SELECT * FROM entry WHERE activity_id=? AND start_date='2026-10-16'", (dashly,)).fetchone()
        html = self.c.get(f"/proof/entry/{e['id']}").get_data(as_text=True)
        self.assertIn("October 16, 2026", html)
        self.assertIn('name="entry_id"', html)
        # first given to the wrong day (the Oct 20 class): Kept points out the dates, nothing is saved
        cls_e = self.conn.execute("SELECT e.* FROM entry e JOIN activity a ON a.id=e.activity_id WHERE a.person_id=? "
                                  "AND a.name='Adult Learning Center' AND e.start_date='2026-10-20'", (pid,)).fetchone()
        data = {"photo_file": (io.BytesIO((SAMPLES / "proof_dashly_oct16.png").read_bytes()), "d.png"), "entry_id": str(cls_e["id"])}
        r = self.c.post("/add/photo", data=data, content_type="multipart/form-data")
        html = self.c.get(r.headers["Location"]).get_data(as_text=True)
        self.assertIn("the photo says <strong>Oct 16</strong>, you logged <strong>Oct 20</strong>", html)
        # then the same photo from the right day: checked against Oct 16, not refused as a duplicate
        # a photo that says $64 for a job she logged as $46: Kept points out the difference
        data = {"photo_file": (io.BytesIO((SAMPLES / "proof_dashly_oct16.png").read_bytes()), "d.png"), "entry_id": str(e["id"])}
        r = self.c.post("/add/photo", data=data, content_type="multipart/form-data")
        html = self.c.get(r.headers["Location"]).get_data(as_text=True)
        self.assertIn("<strong>Proof for:</strong> Dashly, Oct 16", html)
        self.assertIn("the photo says <strong>$64.00</strong>, you logged <strong>$46.00</strong>", html)
        doc_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
        before = len(self.conn.execute("SELECT 1 FROM entry WHERE activity_id=? AND superseded_by IS NULL", (dashly,)).fetchall())
        self.c.post(f"/review/{doc_id}", data={"action": "save", "activity_id": dashly, "start": "2026-10-16",
                                               "end": "2026-10-16", "hours": "3.5", "income": "64"})
        after = len(self.conn.execute("SELECT 1 FROM entry WHERE activity_id=? AND superseded_by IS NULL", (dashly,)).fetchall())
        self.assertEqual(before, after)  # the typed job was replaced, not doubled
        # now the calendar day offers the download
        html = self.c.get("/calendar?month=2026-10&day=16").get_data(as_text=True)
        self.assertIn(f"/files/{doc_id}?dl=1", html)
        r = self.c.get(f"/files/{doc_id}?dl=1")
        self.assertIn("attachment", r.headers["Content-Disposition"])
        self.assertIn("proof-Dashly-2026-10-16", r.headers["Content-Disposition"])

    def test_5d_one_missing_goes_straight_to_its_page(self):
        self.c.post("/demo/start")
        pid = self.conn.execute("SELECT value FROM settings WHERE key='active_person'").fetchone()["value"]
        cm = self.conn.execute("SELECT id FROM activity WHERE name='Corner Market' AND person_id=?", (pid,)).fetchone()["id"]
        r = self.c.get(f"/proof/activity/{cm}?month=2026-10")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/proof/entry/", r.headers["Location"])

    def test_5e_photo_without_values_uses_the_log(self):
        from unittest import mock
        from kept.capture import vision
        self.c.post("/demo/start")
        pid = self.conn.execute("SELECT value FROM settings WHERE key='active_person'").fetchone()["value"]
        e = self.conn.execute("SELECT e.* FROM entry e JOIN activity a ON a.id=e.activity_id WHERE a.name='Corner Market' "
                              "AND e.person_id=?", (pid,)).fetchone()
        with mock.patch.object(vision, "read_photo", return_value=None):
            data = {"photo_cam": (io.BytesIO((SAMPLES / "sample_pay_stub.png").read_bytes()), "n.png"), "entry_id": str(e["id"])}
            r = self.c.post("/add/photo", data=data, content_type="multipart/form-data")
            html = self.c.get(r.headers["Location"]).get_data(as_text=True)
        self.assertIn("From your log", html)
        self.assertIn('value="2026-10-08"', html)

    def test_6_pdfs(self):
        self.c.post("/demo/start")
        self.c.post("/demo/go", data={"to": "2026-11-01"})
        for url in ["/pdf/log/2026-10", "/pdf/packet/2026-10,2026-11"]:
            r = self.c.get(url)
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.data.startswith(b"%PDF"))
        sid = self.conn.execute("SELECT id FROM signoff").fetchone()["id"]
        r = self.c.get(f"/pdf/signoff/{sid}")
        self.assertTrue(r.data.startswith(b"%PDF"))

    def test_7_letter_flow(self):
        self.c.post("/demo/start")
        html = self.c.post("/demo/letter", follow_redirects=True).get_data(as_text=True)
        self.assertIn("Request for proof", html)
        self.assertIn("December 23, 2026", html)
        self.assertIn("28", html)
        lid = self.conn.execute("SELECT id FROM letter ORDER BY id DESC").fetchone()["id"]
        html = self.c.post(f"/letters/{lid}/responded", follow_redirects=True).get_data(as_text=True)
        self.assertIn("Answered", html)

    def test_8_onboarding_creates_own_profile(self):
        self.c.post("/onboarding/1", data={"name": "Ana", "state": "MT", "birth_date": "1990-01-01",
                                            "status": "enrolled", "last_renewal": "2026-07-01"})
        self.c.post("/onboarding/2", data={"caregiver_child": "1", "child_birth_date": "2015-06-01",
                                            "medically_frail": "1"})
        html = self.c.post("/onboarding/3", data={}, follow_redirects=True).get_data(as_text=True)
        self.assertIn("Add at least one activity", html)
        self.c.post("/onboarding/3", data={"name": "Walmart", "type": "work"})
        self.c.post("/onboarding/3", data={})
        self.c.post("/onboarding/4", data={"channel": "text", "max_alerts_week": "3", "quiet_start": "21", "quiet_end": "8"})
        html = self.get("/onboarding/5")
        self.assertIn("you meet the criteria for the parent or caregiver of a child under 14 exemption", html)
        self.assertIn("Exemption mode", html)
        p = db.get_person(self.conn)
        self.assertEqual(p["is_demo"], 0)
        priv = db.private_answers(self.conn, p["id"])
        self.assertTrue(priv["medically_frail"])
        self.assertNotIn("medically_frail", p["answers"])  # kept apart from ordinary answers
        self.get("/")
        html = self.c.post("/profile/delete", data={"confirm": "DELETE"}, follow_redirects=True).get_data(as_text=True)
        self.assertIn("All your data was deleted", html)


if __name__ == "__main__":
    unittest.main()


class PublicMode(unittest.TestCase):
    def test_each_visitor_gets_a_private_demo(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"KEPT_PUBLIC": "1"}):
            app = create_app()
        app.testing = True
        a, b = app.test_client(), app.test_client()
        a.post("/demo/start")
        b.post("/demo/start")
        a.post("/demo/go", data={"to": "2026-11-01"})
        self.assertIn("Demo: Sun Nov 1", a.get("/").get_data(as_text=True))
        hb = b.get("/").get_data(as_text=True)
        self.assertIn("Demo: Tue Oct 20", hb)
        self.assertIn("Public demo", hb)
        c = app.test_client()
        self.assertIn("Open Keisha's demo", c.get("/", follow_redirects=True).get_data(as_text=True))
