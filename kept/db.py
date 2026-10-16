"""Storage: one SQLite file. Health answers live in their own table so they are never
mixed into anything that leaves the app."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

DATA_DIR = Path(os.environ.get("KEPT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "kept.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS person (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'MT',
  birth_date TEXT,
  status TEXT NOT NULL DEFAULT 'enrolled',      -- enrolled | applying
  last_renewal TEXT,                             -- enrolled: last renewal or approval date
  application_date TEXT,                         -- applying: when she plans to apply
  channel TEXT DEFAULT 'text',
  language TEXT DEFAULT 'en',
  quiet_start INTEGER DEFAULT 21,
  quiet_end INTEGER DEFAULT 8,
  max_alerts_week INTEGER DEFAULT 4,
  seasonal INTEGER DEFAULT 0,
  answers TEXT DEFAULT '{}',                     -- non-sensitive onboarding answers
  is_demo INTEGER DEFAULT 0,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS private_answer (
  person_id INTEGER, key TEXT, value TEXT,
  PRIMARY KEY (person_id, key)
);

CREATE TABLE IF NOT EXISTS activity (
  id INTEGER PRIMARY KEY,
  person_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  usual_day INTEGER,          -- 0 = Monday ... 6 = Sunday, NULL if irregular
  usual_hours REAL,
  pay_rate REAL,
  half_time INTEGER DEFAULT 0,
  org_tax_id TEXT, org_address TEXT, org_phone TEXT, contact_name TEXT,
  keywords TEXT DEFAULT '',
  last_confirmed TEXT,
  active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS document (
  id INTEGER PRIMARY KEY,
  person_id INTEGER NOT NULL,
  kind TEXT,                  -- paystub | gig | volunteer | school | letter | other
  filename TEXT, original_name TEXT, sha256 TEXT,
  ocr_text TEXT, ocr_ok INTEGER DEFAULT 1,
  fields TEXT DEFAULT '{}', confidences TEXT DEFAULT '{}',
  read_by TEXT,               -- ai | rules | manual
  activity_id INTEGER,
  status TEXT DEFAULT 'needs_review',   -- needs_review | confirmed | rejected
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS entry (
  id INTEGER PRIMARY KEY,
  person_id INTEGER NOT NULL,
  activity_id INTEGER NOT NULL,
  start_date TEXT NOT NULL, end_date TEXT NOT NULL,
  hours REAL, income REAL,
  source TEXT,                -- text | photo | form | demo
  read_by TEXT,               -- ai | rules | manual
  document_id INTEGER,
  note TEXT,
  superseded_by INTEGER,      -- a statement that replaced typed entries for the same days
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS letter (
  id INTEGER PRIMARY KEY,
  person_id INTEGER NOT NULL,
  document_id INTEGER,
  kind TEXT,                  -- verification_request | noncompliance | renewal | approval | other
  letter_date TEXT, received_date TEXT, respond_by TEXT, deadline TEXT,
  months TEXT DEFAULT '[]',
  explanation TEXT, read_by TEXT,
  status TEXT DEFAULT 'open', -- open | responded
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS signoff (
  id INTEGER PRIMARY KEY,
  person_id INTEGER, activity_id INTEGER, month TEXT, hours REAL,
  status TEXT DEFAULT 'prepared',   -- prepared | signed
  document_id INTEGER, created_at TEXT,
  UNIQUE (person_id, activity_id, month)
);

CREATE TABLE IF NOT EXISTS alert (
  id INTEGER PRIMARY KEY,
  person_id INTEGER, key TEXT, kind TEXT, priority INTEGER, urgent INTEGER,
  title TEXT, body TEXT, action_url TEXT, action_label TEXT,
  day TEXT, status TEXT,       -- sent | held | dismissed
  reason TEXT
);

CREATE TABLE IF NOT EXISTS agent_log (
  id INTEGER PRIMARY KEY,
  person_id INTEGER, day TEXT, step TEXT, decision TEXT, detail TEXT
);

CREATE TABLE IF NOT EXISTS agent_run (person_id INTEGER, day TEXT, PRIMARY KEY (person_id, day));
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn):
    """Add columns that older databases don't have yet."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(document)")}
    if "for_entry" not in cols:  # the typed job this photo was added to prove
        conn.execute("ALTER TABLE document ADD COLUMN for_entry INTEGER")
        conn.commit()
    lcols = {r["name"] for r in conn.execute("PRAGMA table_info(letter)")}
    if "explanation_es" not in lcols:
        conn.execute("ALTER TABLE letter ADD COLUMN explanation_es TEXT")
        conn.commit()
    if "ai_raw" not in cols:     # what the vision model read, before Kept's rules checked it
        conn.execute("ALTER TABLE document ADD COLUMN ai_raw TEXT")
        conn.commit()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- settings / clock

def get_setting(conn, key, default=None):
    r = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(conn, key, value):
    conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, None if value is None else str(value)))
    conn.commit()


def today(conn) -> date:
    """The app's 'today'. In demo mode the date is simulated so judges can play a month."""
    v = get_setting(conn, "sim_today")
    return date.fromisoformat(v) if v else date.today()


# ---------------------------------------------------------------- person

def get_person(conn, pid=None):
    if pid is None:
        pid = get_setting(conn, "active_person")
        if pid is None:
            return None
    return conn.execute("SELECT * FROM person WHERE id=?", (int(pid),)).fetchone()


def answers(person) -> dict:
    return json.loads(person["answers"] or "{}")


def private_answers(conn, pid) -> dict:
    return {r["key"]: json.loads(r["value"]) for r in
            conn.execute("SELECT key, value FROM private_answer WHERE person_id=?", (pid,))}


def set_private(conn, pid, key, value):
    conn.execute("INSERT INTO private_answer VALUES(?,?,?) ON CONFLICT(person_id,key) DO UPDATE SET value=excluded.value",
                 (pid, key, json.dumps(value)))


def all_exemption_answers(conn, person) -> dict:
    a = answers(person)
    a.update(private_answers(conn, person["id"]))
    if person["birth_date"]:
        a["birth_date"] = person["birth_date"]
    return a


def activities(conn, pid, include_inactive=False):
    q = "SELECT * FROM activity WHERE person_id=?" + ("" if include_inactive else " AND active=1") + " ORDER BY id"
    return conn.execute(q, (pid,)).fetchall()


def delete_person(conn, pid):
    for row in conn.execute("SELECT filename FROM document WHERE person_id=?", (pid,)):
        if row["filename"]:
            try:
                (UPLOAD_DIR / row["filename"]).unlink()
            except FileNotFoundError:
                pass
    for t in ("private_answer", "activity", "document", "entry", "letter", "signoff", "alert", "agent_log", "agent_run"):
        conn.execute(f"DELETE FROM {t} WHERE person_id=?", (pid,))
    conn.execute("DELETE FROM person WHERE id=?", (pid,))
    if get_setting(conn, "active_person") == str(pid):
        conn.execute("DELETE FROM settings WHERE key='active_person'")
    conn.commit()
