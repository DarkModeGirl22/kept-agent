"""The agent. Once a day, without being asked, it looks at her situation and decides
what (if anything) is worth telling her.

Each day it runs the same steps in order, and writes down every decision, including the
messages it chose NOT to send and why:

  1. letters      a state letter with a deadline (always urgent)
  2. exemption    an exemption that is about to end
  3. month        the current month at risk, with concrete ways to fix it
  4. check        the next state check: which months count and their status
  5. close        on the 1st: close last month, prepare volunteer pages for signature
  6. proof        work logged without proof
  7. nudge        nothing logged for a few days (full mode only)
  8. reconfirm    recurring activities not confirmed for a long time

Then NVIDIA Nemotron, as the agent's brain, decides which findings are worth a message today
and how to say them (agent/brain.py); its decision is checked by the rules. Finally the rules
ration what's left: urgent messages always go out; the others share a weekly budget she chose,
so she never learns to ignore the app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

from .. import db
from .. import services as S
from ..rules import engine as E

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
RISK_DAYS = (10, 15, 20, 25, 28)
CHECK_MILESTONES = (45, 30, 14, 7, 3, 1)
LETTER_MILESTONES = (21, 14, 7, 3, 1, 0)  # one reminder per band


@dataclass
class Candidate:
    key: str
    kind: str
    priority: int           # 0 = most important
    title: str
    body: str
    action_url: str = "/"
    action_label: str = "Open"
    urgent: bool = False
    cooldown_days: int = 3
    by: str = "rules"       # "nemotron" when the agent's brain chose and wrote it
    why: str = ""


def fmt_h(x: float) -> str:
    return f"{x:g}" if x == int(x) else f"{x:.1f}"


def fmt_d(d: date) -> str:
    return f"{d.strftime('%b')} {d.day}"


class Run:
    def __init__(self, conn, person, day: date, think: bool = False):
        self.conn, self.person, self.day, self.think = conn, person, day, think
        self.ctx = S.context(conn, person, day)
        self.mode = S.mode(self.ctx)
        self.candidates: list[Candidate] = []

    def log(self, step, decision, detail=""):
        self.conn.execute("INSERT INTO agent_log(person_id,day,step,decision,detail) VALUES(?,?,?,?,?)",
                          (self.person["id"], self.day.isoformat(), step, decision, detail))

    # ------------------------------------------------------------ steps

    def step_letters(self):
        rows = self.conn.execute("SELECT * FROM letter WHERE person_id=? AND status='open'", (self.person["id"],)).fetchall()
        if not rows:
            self.log("letters", "nothing to do", "No open letters from the state.")
            return
        for r in rows:
            deadline = date.fromisoformat(r["deadline"])
            left = (deadline - self.day).days
            months = ", ".join(E.month_label(m) for m in json.loads(r["months"] or "[]"))
            if left < 0:
                title, body = "A deadline has passed", f"The deadline for your letter was {fmt_d(deadline)}. Call the state office as soon as you can."
                milestone = "late"
            else:
                bands = [m for m in LETTER_MILESTONES if left <= m]
                milestone = min(bands) if bands else None
                if r["received_date"] == self.day.isoformat():
                    milestone = "new"
                if milestone is None:
                    self.log("letters", "wait", f"Letter #{r['id']}: {left} days left, next reminder later.")
                    continue
                title = f"{left} days left to answer the state" if milestone != "new" else "New letter: the state is asking for proof"
                body = (f"They want proof for {months}. Your folder for those months is ready to check. "
                        f"Deadline: {fmt_d(deadline)}.")
            self.candidates.append(Candidate(f"letter:{r['id']}:{milestone}", "letter", 0, title, body,
                                             f"/letters/{r['id']}", "Open the letter", urgent=True, cooldown_days=0))
            self.log("letters", "alert", f"Letter #{r['id']}: {left} days left (deadline {deadline}).")

    def step_exemption(self):
        res = [r for r in E.evaluate_exemptions(self.ctx.answers, self.day, self.ctx.rules) if r.meets and r.ends_on]
        if not res:
            self.log("exemption", "nothing to do", "No exemption ending soon.")
            return
        for r in res:
            left = (r.ends_on - self.day).days
            if left <= 30:
                self.candidates.append(Candidate(
                    f"exemption_end:{r.code}", "exemption", 1, "Your exemption ends soon",
                    f"Your {r.label} exemption ends on {fmt_d(r.ends_on)}. From then on you'll need 80 hours "
                    f"(or $580) each month. Kept will start tracking for you.", "/profile", "See details",
                    cooldown_days=60))
                self.log("exemption", "alert", f"{r.label} ends in {left} days.")
            else:
                self.log("exemption", "nothing to do", f"{r.label} ends {r.ends_on}, more than 30 days away.")

    def _fixes(self, s: E.MonthSummary) -> tuple[list[str], float]:
        """Concrete ways to close the gap this month, from her own recurring activities."""
        fixes, added = [], 0.0
        tomorrow = self.day + timedelta(days=1)
        for a in self.ctx.activities:
            if not a["active"] or a["usual_day"] is None or not a["usual_hours"]:
                continue
            days = [d for d in E.remaining_weekdays(tomorrow, a["usual_day"]) if d.month == self.day.month] \
                if tomorrow.month == self.day.month else []
            if days:
                h = a["usual_hours"] * len(days)
                added += h
                fixes.append(f"your {WEEKDAY_NAMES[a['usual_day']]} {a['name']} "
                             f"({', '.join(fmt_d(d) for d in days)}) adds {fmt_h(h)}h")
        return fixes, added

    def step_month(self):
        if self.mode in ("exempt", "student", "payroll"):
            self.log("month", "skip", f"Mode is {self.mode}: no monthly hour tracking alerts.")
            return
        month = E.month_key(self.day)
        s = self.ctx.summary(month)
        detail = f"{E.month_label(month)}: {fmt_h(s.hours_total)}h of 80, ${s.income_total:.2f} of $580, status {s.status}."
        if s.status != "at_risk":
            self.log("month", "nothing to do", detail)
            return
        if self.day.day not in RISK_DAYS:
            self.log("month", "wait", detail + " Checking again on a set day to avoid nagging.")
            return
        fixes, added = self._fixes(s)
        parts = [f"You're at {fmt_h(s.hours_total)} of 80 hours ({s.hours_pct}%) with {s.days_left} days left."]
        if fixes:
            parts.append("Planned: " + "; ".join(fixes) + ".")
            still = max(0.0, s.hours_remaining - added)
            if still > 0:
                days_after = max(1, s.days_left)
                parts.append(f"That still leaves {fmt_h(still)}h, about {fmt_h(round(still / days_after, 1))}h a day.")
            else:
                parts.append("If you do those, October is covered." if month.endswith("-10") else "If you do those, this month is covered.")
        else:
            parts.append(f"You need about {fmt_h(s.hours_needed_per_day or 0)}h a day.")
        if s.income_total > 0 and s.income_remaining > 0:
            if s.income_remaining <= 100:
                parts.append(f"Or the income way: you've earned ${s.income_total:.0f}, only ${s.income_remaining:.0f} "
                             f"short of $580. One more paid shift could cover the month.")
            else:
                parts.append(f"Income so far: ${s.income_total:.0f} of $580.")
        self.candidates.append(Candidate(f"month_risk:{month}:{self.day.day}", "month", 2,
                                         f"{E.month_label(month).split()[0]} needs {fmt_h(s.hours_remaining)} more hours",
                                         " ".join(parts), "/plan", "Plan the rest", cooldown_days=2))
        self.log("month", "alert", detail)

    def step_check(self):
        cd = self.ctx.check_date()
        if not cd:
            self.log("check", "nothing to do", "No check date known.")
            return
        left = (cd - self.day).days
        months = self.ctx.checked_months()
        statuses = []
        for m in months:
            s = self.ctx.summary(m)
            proof = "proof complete" if s.proof_complete else (f"{s.unproven_entries} item(s) without proof" if s.unproven_entries else "nothing logged")
            statuses.append(f"{E.month_label(m)}: {E.STATUS_TEXT[s.status].lower()}, {proof}")
        milestone = next((m for m in CHECK_MILESTONES if left == m), None)
        if milestone is None:
            self.log("check", "wait", f"Next check {fmt_d(cd)}, {cd.year} ({left} days). Months that count: {', '.join(E.month_label(m) for m in months)}.")
            return
        urgent = left <= 7
        self.candidates.append(Candidate(
            f"check:{cd}:{milestone}", "check", 3 if not urgent else 1,
            f"State check in {left} days", f"On {fmt_d(cd)} the state looks at: " + "; ".join(statuses) + ".",
            "/folder", "Open my proof folder", urgent=urgent, cooldown_days=0))
        self.log("check", "alert", f"{left} days before the check.")

    def step_close(self):
        if self.day.day != 1:
            self.log("close", "skip", "Months close on the 1st.")
            return
        prev = E.month_key(self.day - timedelta(days=1))
        first = self.conn.execute("SELECT MIN(start_date) d FROM entry WHERE person_id=?", (self.person["id"],)).fetchone()["d"]
        if not first or first[:7] > prev:
            self.log("close", "skip", f"Kept wasn't tracking {E.month_label(prev)} yet.")
            return
        s = self.ctx.summary(prev)
        self.log("close", "closed", f"{E.month_label(prev)} closed: {fmt_h(s.hours_total)}h, ${s.income_total:.2f}, {s.status}.")
        prepared = []
        for (aid, name, typ), hrs in s.hours_by_activity.items():
            if typ == "volunteer" and hrs > 0:
                sid = S.prepare_signoff(self.conn, self.person["id"], aid, prev, hrs)
                prepared.append((sid, name, hrs, aid))
                self.log("close", "prepared", f"Volunteer page for {name}, {E.month_label(prev)}, {fmt_h(hrs)}h.")
        verdict = {"met": "covered", "exempt": "covered by your exemption", "short": "short"}.get(s.status, s.status)
        body = f"{E.month_label(prev)} is {verdict}: {fmt_h(s.hours_total)} hours, ${s.income_total:.0f} earned."
        if s.status == "short":
            body += " Months are checked one by one, so this one may need an explanation if the state asks. Kept keeps everything you logged."
        self.candidates.append(Candidate(f"close:{prev}", "close", 4, f"{E.month_label(prev).split()[0]} is done",
                                         body, f"/folder/{prev}", "See the month", cooldown_days=0))
        for sid, name, hrs, aid in prepared:
            act = self.ctx.activity(aid)
            who = act["contact_name"] if act and act["contact_name"] else "the coordinator"
            self.candidates.append(Candidate(
                f"signoff:{sid}", "signoff", 4, "A page to get signed",
                f"Your {E.month_label(prev)} page for {name} is ready ({fmt_h(hrs)} hours). Ask {who} to sign it now, "
                f"while they remember. Then take a photo of it.", f"/folder/{prev}", "Get the page", cooldown_days=0))

    def step_proof(self):
        rows = self.conn.execute(
            "SELECT e.*, a.name, a.type FROM entry e JOIN activity a ON a.id=e.activity_id "
            "WHERE e.person_id=? AND e.superseded_by IS NULL AND e.end_date<=?",
            (self.person["id"], (self.day - timedelta(days=3)).isoformat())).fetchall()
        missing = {}
        for r in rows:
            if r["type"] == "volunteer":
                continue  # volunteer proof is the monthly signed page
            if not S.is_proven(self.conn, r):
                missing.setdefault(r["name"], 0)
                missing[r["name"]] += 1
        if not missing:
            self.log("proof", "nothing to do", "Everything older than 3 days has proof (volunteer pages handled at month end).")
            return
        names = ", ".join(f"{n} ({c})" for n, c in missing.items())
        week = self.day.isocalendar()[1]
        self.candidates.append(Candidate(
            f"proof:{self.day.year}-{week}", "proof", 5, "Some work has no proof yet",
            f"Logged without proof: {names}. A photo of a pay stub, an app earnings screen, or a note from the "
            f"employer is enough.", "/add", "Add proof", cooldown_days=7))
        self.log("proof", "alert", names)

    def step_nudge(self):
        if self.mode not in ("full", "seasonal"):
            self.log("nudge", "skip", f"Mode is {self.mode}.")
            return
        last = self.conn.execute("SELECT MAX(start_date) m FROM entry WHERE person_id=? AND start_date<=?",
                                 (self.person["id"], self.day.isoformat())).fetchone()["m"]
        gap = (self.day - date.fromisoformat(last)).days if last else 99
        if gap >= 3:
            self.candidates.append(Candidate(f"nudge:{self.day}", "nudge", 6, "Anything to log?",
                                             f"Nothing logged for {gap} days. If you worked, one line is enough: "
                                             f"\"Dashly 3-7pm\" or \"4h pantry\".", "/add", "Log it", cooldown_days=3))
            self.log("nudge", "alert", f"No log for {gap} days.")
        else:
            self.log("nudge", "nothing to do", f"Last log {gap} day(s) ago.")

    def step_reconfirm(self):
        found = False
        for a in self.ctx.activities:
            if not a["active"] or a["usual_day"] is None or not a["last_confirmed"]:
                continue
            age = (self.day - date.fromisoformat(a["last_confirmed"])).days
            if age >= 60:
                found = True
                self.candidates.append(Candidate(
                    f"reconfirm:{a['id']}:{a['last_confirmed']}", "reconfirm", 7, "Still the same?",
                    f"Do you still do {a['name']} on {WEEKDAY_NAMES[a['usual_day']]}s, about {fmt_h(a['usual_hours'])}h? "
                    f"Kept uses this to plan your month.", "/profile", "Update", cooldown_days=14))
        self.log("reconfirm", "alert" if found else "nothing to do", "")

    # ------------------------------------------------------------ deliver

    def deliver(self):
        pid = self.person["id"]
        budget = self.person["max_alerts_week"] or 4
        week_start = self.day - timedelta(days=self.day.weekday())
        used = self.conn.execute(
            "SELECT COUNT(*) c FROM alert WHERE person_id=? AND status='sent' AND urgent=0 AND day>=? AND day<=?",
            (pid, week_start.isoformat(), self.day.isoformat())).fetchone()["c"]
        for c in sorted(self.candidates, key=lambda c: c.priority):
            prev = self.conn.execute("SELECT day FROM alert WHERE person_id=? AND key=? AND status='sent' ORDER BY day DESC LIMIT 1",
                                     (pid, c.key)).fetchone()
            if prev and (c.cooldown_days == 0 or (self.day - date.fromisoformat(prev["day"])).days < c.cooldown_days):
                self.log("deliver", "skip", f"'{c.title}': already told her.")
                continue
            if not c.urgent and used >= budget:
                status, reason = "held", f"Weekly limit reached ({used}/{budget}); not urgent, held back."
            else:
                status = "sent"
                reason = "Urgent: always sent." if c.urgent else f"Within weekly limit ({used + 1}/{budget})."
                if not c.urgent:
                    used += 1
            if c.by == "nemotron":
                reason = f"Chosen and written by NVIDIA Nemotron: {c.why} Checked by Kept's rules. {reason}"
            self.conn.execute(
                "INSERT INTO alert(person_id,key,kind,priority,urgent,title,body,action_url,action_label,day,status,reason)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (pid, c.key, c.kind, c.priority, int(c.urgent), c.title, c.body, c.action_url, c.action_label,
                 self.day.isoformat(), status, reason))
            self.log("deliver", status, f"'{c.title}': {reason}")

    def run(self):
        self.log("start", "run", f"Daily check. Mode: {self.mode}.")
        for step in (self.step_letters, self.step_exemption, self.step_month, self.step_check, self.step_close,
                     self.step_proof, self.step_nudge, self.step_reconfirm):
            step()
        if self.think:
            # NVIDIA Nemotron decides what is worth telling her today; Kept's rules check its decision
            from . import brain
            brain.think(self)
        self.deliver()
        self.conn.execute("INSERT OR IGNORE INTO agent_run(person_id, day) VALUES(?,?)", (self.person["id"], self.day.isoformat()))
        self.conn.commit()


def run_day(conn, person, day: date, think: bool = False):
    Run(conn, person, day, think=think).run()


def catch_up(conn, person, until: date, max_days: int = 120):
    """Run every day the agent hasn't run yet, up to `until` (the app's today)."""
    last = conn.execute("SELECT MAX(day) d FROM agent_run WHERE person_id=?", (person["id"],)).fetchone()["d"]
    start = date.fromisoformat(last) + timedelta(days=1) if last else until
    start = max(start, until - timedelta(days=max_days))
    d = start
    while d <= until:
        run_day(conn, person, d, think=(d == until))
        d += timedelta(days=1)


def rerun_today(conn, person, day: date):
    """After she logs something, re-run today so the screen reflects it."""
    # the brain already thought about today (its messages are sent and its reasoning is in the log):
    # re-running the rules is enough, and keeps saving instant
    thought = conn.execute("SELECT 1 FROM agent_log WHERE person_id=? AND day=? AND step='brain'",
                           (person["id"], day.isoformat())).fetchone()
    conn.execute("DELETE FROM agent_log WHERE person_id=? AND day=? AND step<>'brain'", (person["id"], day.isoformat()))
    conn.execute("DELETE FROM alert WHERE person_id=? AND day=? AND status='held'", (person["id"], day.isoformat()))
    conn.execute("DELETE FROM agent_run WHERE person_id=? AND day=?", (person["id"], day.isoformat()))
    run_day(conn, person, day, think=not thought)
