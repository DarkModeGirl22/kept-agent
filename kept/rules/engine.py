"""The rules engine.

Everything that decides whether a month counts lives here, in plain Python, driven by a
versioned rules file per state. The AI never computes hours, income, deadlines or
exemptions: it can only help read documents. Every result carries the rules version that
produced it, so the same facts always give the same answer.
"""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

STATES_DIR = Path(__file__).parent / "states"

ACTIVITY_LABELS = {
    "work": "Job (payroll)",
    "gig": "Gig app",
    "cash": "Cash job",
    "self_employment": "Self-employed",
    "volunteer": "Volunteering",
    "education": "School",
    "job_training": "Job training",
    "work_program": "Work program",
}


# ---------------------------------------------------------------- rules files

def load_rules(state: str) -> dict:
    path = STATES_DIR / f"{state.upper()}.json"
    if not path.exists():
        path = STATES_DIR / "GENERIC.json"
    return json.loads(path.read_text(encoding="utf-8"))


def supported_states() -> list[tuple[str, str]]:
    out = []
    for p in sorted(STATES_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append((d["state"], d["state_name"]))
    out.sort(key=lambda s: (s[0] == "GENERIC", s[1]))
    return out


# ---------------------------------------------------------------- dates

def month_start(d: date) -> date:
    return d.replace(day=1)


def month_end(d: date) -> date:
    return d.replace(day=calendar.monthrange(d.year, d.month)[1])


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def parse_month(key: str) -> date:
    y, m = key.split("-")
    return date(int(y), int(m), 1)


def month_label(key_or_date) -> str:
    d = parse_month(key_or_date) if isinstance(key_or_date, str) else key_or_date
    return d.strftime("%B %Y")


def days_in_month(d: date) -> int:
    return calendar.monthrange(d.year, d.month)[1]


# ---------------------------------------------------------------- entries

@dataclass
class Entry:
    """One logged piece of activity. A pay stub covering two weeks is one entry with a
    start and end date; a single shift has start == end."""
    id: int
    activity_id: int
    activity_type: str
    start: date
    end: date
    hours: float | None = None
    income: float | None = None
    proven: bool = False
    activity_name: str = ""

    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass
class Portion:
    """The share of an entry that falls inside one calendar month."""
    entry: Entry
    hours: float
    income: float


def split_by_month(entry: Entry) -> dict[str, Portion]:
    """Split an entry across calendar months in proportion to its days.

    A pay period from 25 June to 8 July has 6 days in June and 8 in July, so 6/14 of
    its hours and pay belong to June. The state looks at months, so this matters.
    """
    total_days = entry.days()
    out: dict[str, Portion] = {}
    cur = entry.start
    while cur <= entry.end:
        seg_end = min(month_end(cur), entry.end)
        share = ((seg_end - cur).days + 1) / total_days
        out[month_key(cur)] = Portion(
            entry=entry,
            hours=round((entry.hours or 0.0) * share, 2),
            income=round((entry.income or 0.0) * share, 2),
        )
        cur = seg_end + timedelta(days=1)
    return out


def dedupe(entries: list[Entry]) -> list[Entry]:
    """Count the same work once, even if it was logged twice (a text log and then the
    pay stub for the same shift, or the same screenshot uploaded twice). When two
    entries match, the one with proof wins."""
    best: dict[tuple, Entry] = {}
    for e in entries:
        k = (e.activity_id, e.start, e.end, round(e.hours or 0, 2), round(e.income or 0, 2))
        if k not in best or (e.proven and not best[k].proven):
            best[k] = e
    return list(best.values())


# ---------------------------------------------------------------- exemptions

@dataclass
class ExemptionResult:
    code: str
    label: str
    meets: bool
    ends_on: date | None
    proof: str
    sensitive: bool = False

    def wording(self, state_name: str) -> str:
        if self.meets:
            until = f" until {self.ends_on.strftime('%B %d, %Y')}" if self.ends_on else ""
            return (f"Based on your answers, you meet the criteria for the {self.label} exemption "
                    f"under {state_name}'s rules{until}. The state makes the final decision.")
        return f"Based on your answers, you don't meet the criteria for the {self.label} exemption."


def _years_later(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 February
        return d.replace(year=d.year + years, day=28)


def evaluate_exemptions(answers: dict, today: date, rules: dict) -> list[ExemptionResult]:
    """answers holds the onboarding answers: booleans, plus dates where an exemption ends.

    Dates are ISO strings: birth_date (the person), baby_birth_date, child_birth_date,
    release_date, hospital_end_date."""
    ex = rules["exemptions"]
    results: list[ExemptionResult] = []

    def d(key):
        v = answers.get(key)
        return date.fromisoformat(v) if v else None

    def add(code, meets, ends_on=None):
        cfg = ex[code]
        if ends_on and ends_on < today:
            meets = False
        results.append(ExemptionResult(code, cfg["label"], bool(meets), ends_on if meets else None,
                                       cfg["proof"], cfg.get("sensitive", False)))

    if answers.get("pregnant"):
        add("pregnant", True)
    baby = d("baby_birth_date")
    if baby:
        add("postpartum", True, add_months(baby, ex["postpartum"]["months_after"]) - timedelta(days=1))
    child = d("child_birth_date")
    if answers.get("caregiver_child") and child:
        add("caregiver_child", True, _years_later(child, ex["caregiver_child"]["child_age_limit"]) - timedelta(days=1))
    elif answers.get("caregiver_child"):
        add("caregiver_child", True)
    for code in ("caregiver_disabled", "tribal", "veteran_total_disability", "medically_frail",
                 "treatment", "snap_tanf_work"):
        if answers.get(code):
            add(code, True)
    born = d("birth_date")
    if answers.get("former_foster_youth") and born:
        add("former_foster_youth", True, _years_later(born, ex["former_foster_youth"]["age_limit"]) - timedelta(days=1))
    rel = d("release_date")
    if rel:
        add("released_incarceration", True, add_months(rel, ex["released_incarceration"]["months_after"]))
    hosp = d("hospital_end_date")
    if hosp:
        add("hospital_stay", True, add_months(hosp, ex["hospital_stay"]["months_after"]))
    return results


def active_exemptions(answers: dict, on: date, rules: dict) -> list[ExemptionResult]:
    """Exemptions that cover a given day (used per month: covered if any day of it is)."""
    return [r for r in evaluate_exemptions(answers, on, rules) if r.meets]


def exempt_for_month(answers: dict, month: date, rules: dict) -> list[ExemptionResult]:
    """An exemption counts for a month if it is in force on the month's first day."""
    return active_exemptions(answers, month_start(month), rules)


# ---------------------------------------------------------------- checks

def next_check(last_renewal: date, today: date, rules: dict) -> date:
    """Renewals come every N months after the last one. Return the first one after today."""
    n = rules["renewal_interval_months"]
    d = add_months(last_renewal, n)
    while d <= today:
        d = add_months(d, n)
    return d


def months_checked(check_date: date, rules: dict, applying: bool = False) -> list[str]:
    """The full calendar months the state looks at for a check on check_date: the N
    months before the month of the check. Never a random month, never a future one."""
    n = rules["months_checked_at_application" if applying else "months_checked_at_renewal"]
    first_of_check = month_start(check_date)
    return [month_key(add_months(first_of_check, -i)) for i in range(n, 0, -1)]


def letter_deadline(received: date, respond_by: date | None, rules: dict) -> date:
    """If the letter names a date, that date rules. Otherwise the 30-day window from receipt."""
    return respond_by or (received + timedelta(days=rules["notice_response_days"]))


# ---------------------------------------------------------------- month summary

@dataclass
class MonthSummary:
    month: str
    rules_version: str
    hours_by_type: dict = field(default_factory=dict)
    hours_by_activity: dict = field(default_factory=dict)
    hours_total: float = 0.0
    hours_proven: float = 0.0
    income_total: float = 0.0
    income_proven: float = 0.0
    half_time_student: bool = False
    seasonal_average: float | None = None
    exemptions: list = field(default_factory=list)
    met_by: list = field(default_factory=list)
    status: str = "upcoming"           # exempt | met | on_track | at_risk | short | upcoming
    hours_target: float = 80.0
    income_target: float = 580.0
    hours_remaining: float = 80.0
    income_remaining: float = 580.0
    days_left: int = 0
    projected_hours: float | None = None
    projected_income: float | None = None
    hours_needed_per_day: float | None = None
    closer_path: str = "hours"          # which of hours / income is closer to done
    proof_complete: bool = False        # everything logged has proof
    unproven_entries: int = 0

    @property
    def hours_pct(self) -> int:
        return min(100, round(100 * self.hours_total / self.hours_target)) if self.hours_target else 0

    @property
    def hours_proven_pct(self) -> int:
        return min(100, round(100 * self.hours_proven / self.hours_target)) if self.hours_target else 0

    @property
    def income_pct(self) -> int:
        return min(100, round(100 * self.income_total / self.income_target)) if self.income_target else 0

    @property
    def income_proven_pct(self) -> int:
        return min(100, round(100 * self.income_proven / self.income_target)) if self.income_target else 0

    @property
    def is_met(self) -> bool:
        return self.status in ("met", "exempt")


def month_portions(entries: list[Entry], month: str) -> list[Portion]:
    out = []
    for e in dedupe(entries):
        p = split_by_month(e).get(month)
        if p:
            out.append(p)
    return out


def monthly_income(entries: list[Entry], month: str, rules: dict) -> float:
    inc = set(rules["income_activities"])
    return round(sum(p.income for p in month_portions(entries, month) if p.entry.activity_type in inc), 2)


def summarize_month(entries: list[Entry], month: str, today: date, rules: dict, *,
                    exemption_answers: dict | None = None, seasonal: bool = False,
                    half_time_student: bool = False) -> MonthSummary:
    m0 = parse_month(month)
    m1 = month_end(m0)
    hour_types = set(rules["hour_activities"])
    inc_types = set(rules["income_activities"])

    s = MonthSummary(month=month, rules_version=rules["version"],
                     hours_target=float(rules["hours_per_month"]),
                     income_target=float(rules["income_per_month"]))

    unproven = 0
    for p in month_portions(entries, month):
        e = p.entry
        if e.activity_type in hour_types and p.hours:
            s.hours_by_type[e.activity_type] = round(s.hours_by_type.get(e.activity_type, 0) + p.hours, 2)
            key = (e.activity_id, e.activity_name or ACTIVITY_LABELS.get(e.activity_type, e.activity_type), e.activity_type)
            s.hours_by_activity[key] = round(s.hours_by_activity.get(key, 0) + p.hours, 2)
            s.hours_total += p.hours
            if e.proven:
                s.hours_proven += p.hours
        if e.activity_type in inc_types and p.income:
            s.income_total += p.income
            if e.proven:
                s.income_proven += p.income
        if not e.proven:
            unproven += 1
    s.hours_total = round(s.hours_total, 2)
    s.hours_proven = round(s.hours_proven, 2)
    s.income_total = round(s.income_total, 2)
    s.income_proven = round(s.income_proven, 2)
    s.unproven_entries = unproven
    s.proof_complete = unproven == 0 and (s.hours_total > 0 or s.income_total > 0)

    s.hours_remaining = round(max(0.0, s.hours_target - s.hours_total), 2)
    s.income_remaining = round(max(0.0, s.income_target - s.income_total), 2)
    hours_share = s.hours_total / s.hours_target if s.hours_target else 0
    income_share = s.income_total / s.income_target if s.income_target else 0
    s.closer_path = "income" if income_share > hours_share else "hours"

    s.half_time_student = half_time_student and rules.get("half_time_education_counts", False)
    if seasonal:
        n = rules["seasonal_average_months"]
        months = [month_key(add_months(m0, -i)) for i in range(n - 1, -1, -1)]
        s.seasonal_average = round(sum(monthly_income(entries, k, rules) for k in months) / n, 2)

    if exemption_answers:
        s.exemptions = exempt_for_month(exemption_answers, m0, rules)

    if s.hours_total >= s.hours_target:
        s.met_by.append("hours")
    if s.income_total >= s.income_target:
        s.met_by.append("income")
    if s.half_time_student:
        s.met_by.append("school")
    if s.seasonal_average is not None and s.seasonal_average >= s.income_target:
        s.met_by.append("seasonal_average")

    # time position
    if today < m0:
        s.days_left = days_in_month(m0)
    elif today > m1:
        s.days_left = 0
    else:
        s.days_left = (m1 - today).days + 1  # today still counts

    if s.exemptions:
        s.status = "exempt"
    elif s.met_by:
        s.status = "met"
    elif today > m1:
        s.status = "short"
    elif today < m0:
        s.status = "upcoming"
    else:
        elapsed = (today - m0).days + 1
        dim = days_in_month(m0)
        # A pace projection is noisy in the first days, so only judge from day 5.
        s.projected_hours = round(s.hours_total / elapsed * dim, 1)
        s.projected_income = round(s.income_total / elapsed * dim, 2)
        s.hours_needed_per_day = round(s.hours_remaining / s.days_left, 1) if s.days_left else None
        if elapsed < 5:
            s.status = "on_track" if s.hours_needed_per_day is not None and s.hours_needed_per_day <= 4 else "at_risk"
        elif s.projected_hours >= s.hours_target or s.projected_income >= s.income_target:
            s.status = "on_track"
        else:
            s.status = "at_risk"
    return s


STATUS_TEXT = {
    "exempt": "Exempt",
    "met": "Done",
    "on_track": "On track",
    "at_risk": "At risk",
    "short": "Short",
    "upcoming": "Not started",
}


def remaining_weekdays(today: date, weekday: int) -> list[date]:
    """Dates from today to the end of the month falling on a weekday (0 = Monday)."""
    end = month_end(today)
    out, d = [], today
    while d <= end:
        if d.weekday() == weekday:
            out.append(d)
        d += timedelta(days=1)
    return out
