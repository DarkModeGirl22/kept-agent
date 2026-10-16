"""Run the agent's daily check for everyone, without opening the app.

Schedule it once a day (Windows Task Scheduler, or cron):  python -m kept.agent.daily
The app also runs it automatically whenever it's opened."""

from datetime import date

from .. import db
from . import watchdog


def main():
    conn = db.connect()
    people = conn.execute("SELECT * FROM person WHERE is_demo=0").fetchall()
    for p in people:
        watchdog.catch_up(conn, p, date.today())
        sent = conn.execute("SELECT COUNT(*) c FROM alert WHERE person_id=? AND day=? AND status='sent'",
                            (p["id"], date.today().isoformat())).fetchone()["c"]
        print(f"{p['name']}: checked, {sent} message(s) today")
    if not people:
        print("No one set up yet.")


if __name__ == "__main__":
    main()
