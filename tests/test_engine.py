import unittest
from datetime import date

from kept.rules import engine as E

MT = E.load_rules("MT")


def entry(i, t, start, end=None, hours=None, income=None, proven=False, aid=None):
    return E.Entry(id=i, activity_id=aid or i, activity_type=t, start=start, end=end or start,
                   hours=hours, income=income, proven=proven)


class Thresholds(unittest.TestCase):
    def test_80_hours_meets(self):
        es = [entry(1, "gig", date(2026, 10, d), hours=8, aid=1) for d in range(1, 11)]
        s = E.summarize_month(es, "2026-10", date(2026, 11, 2), MT)
        self.assertEqual(s.hours_total, 80)
        self.assertEqual(s.status, "met")
        self.assertIn("hours", s.met_by)

    def test_79_hours_and_low_income_is_short_after_month_ends(self):
        es = [entry(1, "volunteer", date(2026, 10, 3), hours=79)]
        s = E.summarize_month(es, "2026-10", date(2026, 11, 1), MT)
        self.assertEqual(s.status, "short")
        self.assertEqual(s.hours_remaining, 1)

    def test_income_path_580(self):
        es = [entry(1, "gig", date(2026, 10, 5), hours=20, income=580)]
        s = E.summarize_month(es, "2026-10", date(2026, 11, 1), MT)
        self.assertEqual(s.met_by, ["income"])
        self.assertEqual(s.status, "met")

    def test_income_threshold_is_80_times_minimum_wage(self):
        self.assertAlmostEqual(MT["income_per_month"], 80 * MT["federal_minimum_wage"])

    def test_volunteer_hours_count_for_hours_not_income(self):
        es = [entry(1, "volunteer", date(2026, 10, 3), hours=10, income=999)]
        s = E.summarize_month(es, "2026-10", date(2026, 10, 3), MT)
        self.assertEqual(s.hours_total, 10)
        self.assertEqual(s.income_total, 0)

    def test_mixing_activities(self):
        es = [entry(1, "gig", date(2026, 10, 2), hours=40), entry(2, "volunteer", date(2026, 10, 3), hours=20),
              entry(3, "education", date(2026, 10, 4), hours=20)]
        s = E.summarize_month(es, "2026-10", date(2026, 11, 1), MT)
        self.assertEqual(s.status, "met")
        self.assertEqual(s.hours_by_type, {"gig": 40, "volunteer": 20, "education": 20})

    def test_half_time_student_meets(self):
        s = E.summarize_month([], "2026-10", date(2026, 11, 1), MT, half_time_student=True)
        self.assertEqual(s.met_by, ["school"])

    def test_closer_path(self):
        es = [entry(1, "gig", date(2026, 10, 2), hours=30, income=570)]
        s = E.summarize_month(es, "2026-10", date(2026, 10, 20), MT)
        self.assertEqual(s.closer_path, "income")
        self.assertEqual(s.income_remaining, 10)


class Seasonal(unittest.TestCase):
    def test_six_month_average_income(self):
        # A summer worker: big months May-August, nothing in September and October.
        es = [entry(i, "work", date(2026, m, 15), income=900, hours=100) for i, m in enumerate([5, 6, 7, 8], 1)]
        s = E.summarize_month(es, "2026-10", date(2026, 11, 1), MT, seasonal=True)
        self.assertEqual(s.seasonal_average, 600.0)  # 3600 / 6
        self.assertIn("seasonal_average", s.met_by)
        self.assertEqual(s.status, "met")

    def test_seasonal_off_by_default(self):
        es = [entry(1, "work", date(2026, 6, 15), income=4000)]
        s = E.summarize_month(es, "2026-10", date(2026, 11, 1), MT)
        self.assertIsNone(s.seasonal_average)
        self.assertEqual(s.status, "short")


class PayPeriods(unittest.TestCase):
    def test_split_across_months_by_days(self):
        e = entry(1, "work", date(2026, 6, 25), date(2026, 7, 8), hours=56, income=700)
        parts = E.split_by_month(e)
        self.assertEqual(parts["2026-06"].hours, 24.0)   # 6 of 14 days
        self.assertEqual(parts["2026-07"].hours, 32.0)   # 8 of 14 days
        self.assertEqual(parts["2026-06"].income, 300.0)
        self.assertEqual(parts["2026-07"].income, 400.0)

    def test_split_shows_in_summary(self):
        e = entry(1, "work", date(2026, 9, 28), date(2026, 10, 4), hours=35)
        s = E.summarize_month([e], "2026-10", date(2026, 10, 10), MT)
        self.assertEqual(s.hours_total, 20.0)  # 4 of 7 days


class Duplicates(unittest.TestCase):
    def test_same_shift_counted_once_and_proof_wins(self):
        a = entry(1, "gig", date(2026, 10, 5), hours=5, income=80, aid=7)
        b = entry(2, "gig", date(2026, 10, 5), hours=5, income=80, proven=True, aid=7)
        out = E.dedupe([a, b])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].proven)

    def test_different_activities_not_merged(self):
        a = entry(1, "gig", date(2026, 10, 5), hours=5, aid=1)
        b = entry(2, "gig", date(2026, 10, 5), hours=5, aid=2)
        self.assertEqual(len(E.dedupe([a, b])), 2)


class Proof(unittest.TestCase):
    def test_proven_and_logged_tracked_separately(self):
        es = [entry(1, "gig", date(2026, 10, 2), hours=30, proven=True), entry(2, "cash", date(2026, 10, 3), hours=10)]
        s = E.summarize_month(es, "2026-10", date(2026, 10, 10), MT)
        self.assertEqual(s.hours_total, 40)
        self.assertEqual(s.hours_proven, 30)
        self.assertFalse(s.proof_complete)
        self.assertEqual(s.unproven_entries, 1)


class Projection(unittest.TestCase):
    def test_at_risk_mid_month(self):
        es = [entry(1, "gig", date(2026, 10, 2), hours=40)]
        s = E.summarize_month(es, "2026-10", date(2026, 10, 20), MT)
        self.assertEqual(s.status, "at_risk")
        self.assertEqual(s.days_left, 12)
        self.assertAlmostEqual(s.hours_needed_per_day, 3.3)

    def test_on_track(self):
        es = [entry(1, "gig", date(2026, 10, 2), hours=60)]
        s = E.summarize_month(es, "2026-10", date(2026, 10, 20), MT)
        self.assertEqual(s.status, "on_track")

    def test_upcoming_month(self):
        s = E.summarize_month([], "2026-12", date(2026, 10, 20), MT)
        self.assertEqual(s.status, "upcoming")


class Checks(unittest.TestCase):
    def test_next_check_every_six_months(self):
        self.assertEqual(E.next_check(date(2026, 7, 1), date(2026, 10, 15), MT), date(2027, 1, 1))
        self.assertEqual(E.next_check(date(2026, 1, 1), date(2026, 10, 15), MT), date(2027, 1, 1))

    def test_montana_checks_three_months_before(self):
        self.assertEqual(E.months_checked(date(2027, 1, 1), MT), ["2026-10", "2026-11", "2026-12"])

    def test_generic_checks_one_month(self):
        g = E.load_rules("GENERIC")
        self.assertEqual(E.months_checked(date(2027, 1, 15), g), ["2026-12"])

    def test_application_lookback(self):
        self.assertEqual(E.months_checked(date(2026, 11, 3), MT, applying=True), ["2026-10"])

    def test_letter_30_days(self):
        self.assertEqual(E.letter_deadline(date(2026, 10, 20), None, MT), date(2026, 11, 19))
        self.assertEqual(E.letter_deadline(date(2026, 10, 20), date(2026, 11, 12), MT), date(2026, 11, 12))

    def test_unknown_state_uses_generic(self):
        self.assertEqual(E.load_rules("ZZ")["state"], "GENERIC")


class Exemptions(unittest.TestCase):
    def test_caregiver_ends_day_before_14th_birthday(self):
        r = E.evaluate_exemptions({"caregiver_child": True, "child_birth_date": "2013-03-10"}, date(2026, 10, 1), MT)
        self.assertTrue(r[0].meets)
        self.assertEqual(r[0].ends_on, date(2027, 3, 9))
        self.assertIn("you meet the criteria", r[0].wording("Montana"))

    def test_caregiver_expired(self):
        r = E.evaluate_exemptions({"caregiver_child": True, "child_birth_date": "2010-01-01"}, date(2026, 10, 1), MT)
        self.assertFalse(r[0].meets)
        self.assertIn("don't meet", r[0].wording("Montana"))

    def test_postpartum_12_months(self):
        r = E.evaluate_exemptions({"baby_birth_date": "2026-03-15"}, date(2026, 10, 1), MT)
        self.assertEqual(r[0].ends_on, date(2027, 3, 14))

    def test_exempt_month_status(self):
        s = E.summarize_month([], "2026-10", date(2026, 11, 2), MT, exemption_answers={"pregnant": True})
        self.assertEqual(s.status, "exempt")

    def test_no_answers_no_exemptions(self):
        self.assertEqual(E.evaluate_exemptions({}, date(2026, 10, 1), MT), [])


if __name__ == "__main__":
    unittest.main()
