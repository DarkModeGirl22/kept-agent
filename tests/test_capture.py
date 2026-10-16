import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from kept.capture import extract, letters, llm, textlog

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 10, 20)  # a Tuesday
ACTS = [
    {"id": 1, "name": "Dashly", "type": "gig", "keywords": "dashly,delivery"},
    {"id": 2, "name": "Corner Market", "type": "cash", "keywords": "store"},
    {"id": 3, "name": "Riverside Community Pantry", "type": "volunteer", "keywords": "pantry,food bank"},
    {"id": 4, "name": "Adult Learning Center", "type": "education", "keywords": "ged,class"},
]


class AIOff(unittest.TestCase):
    """Every capture path must work with the AI switched off."""

    def setUp(self):
        llm.set_enabled(False)

    def tearDown(self):
        llm.set_enabled(True)


class TextLog(AIOff):
    def p(self, text):
        return textlog.parse(text, TODAY, ACTS)

    def test_hours_activity_today(self):
        r = self.p("4h food bank today")
        self.assertEqual((r["hours"], r["activity_id"], r["date"]), (4, 3, "2026-10-20"))
        self.assertEqual(r["read_by"], "rules")
        self.assertEqual(r["missing"], [])

    def test_time_range_and_money(self):
        r = self.p("Dashly 3pm-8pm yesterday, made $85")
        self.assertEqual((r["hours"], r["income"], r["date"], r["activity_id"]), (5, 85, "2026-10-19", 1))

    def test_range_without_ampm_on_start(self):
        self.assertEqual(self.p("delivery 3-8pm")["hours"], 5)

    def test_minutes_formats(self):
        self.assertEqual(self.p("class 2h30 today")["hours"], 2.5)
        self.assertEqual(self.p("class 2h 30m today")["hours"], 2.5)
        self.assertEqual(self.p("class 90 min")["hours"], 1.5)

    def test_weekday_means_most_recent(self):
        self.assertEqual(self.p("pantry 4 hours saturday")["date"], "2026-10-17")
        self.assertEqual(self.p("class 3h last tuesday")["date"], "2026-10-13")

    def test_explicit_dates_never_future(self):
        self.assertEqual(self.p("store 5h 10/14")["date"], "2026-10-14")
        self.assertEqual(self.p("store 5h Oct 3")["date"], "2026-10-03")
        self.assertEqual(self.p("store 5h 3 october")["date"], "2026-10-03")
        self.assertEqual(self.p("store 5h dec 1")["date"], "2025-12-01")

    def test_missing_fields_are_asked_not_guessed(self):
        r = self.p("worked today")
        self.assertIn("hours", r["missing"])
        self.assertIsNone(r["hours"])

    def test_gig_with_money_only_is_fine(self):
        r = self.p("dashly made $60 today")
        self.assertEqual(r["income"], 60)
        self.assertNotIn("hours", r["missing"])

    def test_volunteer_needs_hours(self):
        self.assertIn("hours", self.p("pantry today")["missing"])

    def test_unknown_activity_is_asked(self):
        self.assertIn("activity", self.p("4h babysitting today")["missing"])


class Documents(AIOff):
    def read(self, name, ai=None):
        # the transcription Nemotron Super VL returns for this sample (saved, so tests run offline)
        text = (ROOT / "tests" / "fixtures" / name.replace(".png", ".txt")).read_text()
        return extract.extract(text, date(2026, 11, 5), ACTS, ai=ai)

    def test_gig_statement(self):
        r = self.read("sample_gig_statement.png")
        f = r["fields"]
        self.assertEqual(r["kind"], "gig")
        self.assertEqual((f["period_start"], f["period_end"]), (date(2026, 10, 5), date(2026, 10, 11)))
        self.assertEqual((f["hours"], f["income"], f["activity_id"]), (7.5, 96.0, 1))
        self.assertEqual(r["ask"], [])

    def test_pay_stub_reads_gross_not_net(self):
        f = self.read("sample_pay_stub.png")["fields"]
        self.assertEqual((f["hours"], f["income"], f["activity_id"]), (4.0, 48.0, 2))

    def test_volunteer_sheet(self):
        r = self.read("sample_volunteer_sheet.png")
        self.assertEqual(r["kind"], "volunteer")
        self.assertEqual(r["fields"]["hours"], 20.0)

    def test_blank_text_asks_everything(self):
        r = extract.extract("", date(2026, 11, 5), ACTS)
        self.assertIn("hours", r["ask"])
        self.assertIn("activity_id", r["ask"])

    def test_impossible_hours_get_low_confidence(self):
        txt = "Dashly\nPeriod: 10/05/2026 - 10/05/2026\nActive time 30h\nTotal earnings $50.00"
        r = extract.extract(txt, date(2026, 11, 5), ACTS)
        self.assertIn("hours", r["ask"])


class Letters(AIOff):
    def test_sample_letter(self):
        text = (ROOT / "tests" / "fixtures" / "sample_state_letter.txt").read_text()
        r = letters.read_letter(text, date(2026, 11, 25))
        self.assertEqual(r["kind"], "verification_request")
        self.assertEqual(r["respond_by"], date(2026, 12, 23))
        self.assertEqual(r["months"], ["2026-10", "2026-11"])

    def test_pdf_letter_text_layer(self):
        from kept.services import _pdf_text
        text, ok = _pdf_text(ROOT / "samples" / "sample_renewal_letter.pdf")
        self.assertTrue(ok)
        r = letters.read_letter(text, date(2026, 10, 20))
        self.assertEqual(r["kind"], "renewal")
        self.assertEqual(r["letter_date"], date(2026, 10, 19))
        # "due for renewal on January 1" is not the deadline; "respond by December 31" is
        self.assertEqual(r["respond_by"], date(2026, 12, 31))
        self.assertEqual(r["months"], ["2026-10", "2026-11", "2026-12"])

    def test_noncompliance(self):
        r = letters.read_letter("We could not verify your hours. Your coverage will end.", date(2026, 11, 25))
        self.assertEqual(r["kind"], "noncompliance")
        self.assertIsNone(r["respond_by"])


class AIGuardrails(unittest.TestCase):
    """When the AI answers, its values are checked by code."""

    def test_ai_value_not_in_document_is_thrown_away(self):
        txt = "Some app\nweekly summary\nOct 5 - Oct 11, 2026\nTrips 12\nPaid out $96.00"
        fake = {"period_start": "2026-10-05", "period_end": "2026-10-11", "hours": 14, "income": 96.0}
        with mock.patch.object(llm, "available", return_value=True), \
             mock.patch.object(llm, "chat_json", return_value=fake):
            r = extract.extract(txt, date(2026, 11, 5), ACTS)
        self.assertNotEqual(r["fields"].get("hours"), 14)  # "14" is nowhere in the text
        self.assertIn("hours", r["ask"])

    def test_ai_grounded_value_is_used(self):
        txt = "Some app\nOct 5 - Oct 11, 2026\nWorked 7.5\nPaid out 96.00"
        fake = {"hours": 7.5, "income": 96.0}
        with mock.patch.object(llm, "available", return_value=True), \
             mock.patch.object(llm, "chat_json", return_value=fake):
            r = extract.extract(txt, date(2026, 11, 5), ACTS)
        self.assertEqual(r["fields"]["hours"], 7.5)
        self.assertEqual(r["read_by"], "ai")

    def test_ai_text_log_cannot_invent_money(self):
        fake = {"date": "2026-10-20", "hours": None, "income": 120, "activity_id": 1}
        with mock.patch.object(llm, "available", return_value=True), \
             mock.patch.object(llm, "chat_json", return_value=fake):
            r = textlog.parse("did some dashly runs", TODAY, ACTS)
        self.assertIsNone(r["income"])

    def test_ai_bad_answer_falls_back(self):
        with mock.patch.object(llm, "available", return_value=True), \
             mock.patch.object(llm, "chat_json", return_value=None):
            r = textlog.parse("4h food bank today", TODAY, ACTS)
        self.assertEqual(r["hours"], 4)
        self.assertEqual(r["read_by"], "rules")

    def test_ai_letter_explanation_with_invented_date_is_refused(self):
        fake = {"explanation": "You must reply by March 3 2027 or lose coverage forever, so act quickly please."}
        with mock.patch.object(llm, "available", return_value=True), \
             mock.patch.object(llm, "chat_json", return_value=fake):
            r = letters.read_letter("Request for verification. Please respond by December 23, 2026.", date(2026, 11, 25))
        self.assertEqual(r["read_by"], "rules")

    def test_ollama_off_means_unavailable(self):
        with mock.patch.object(llm, "status", return_value={"ok": False, "models": []}):
            self.assertFalse(llm.available())
            self.assertIsNone(llm.chat_json("s", "u"))


if __name__ == "__main__":
    unittest.main()
