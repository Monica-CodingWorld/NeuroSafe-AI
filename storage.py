import sqlite3

conn = sqlite3.connect("events.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS events(
time TEXT,
environment TEXT,
decision TEXT,
reason TEXT
)
""")

conn.commit()

def store_event(payload):
    cur.execute(
        "INSERT INTO events VALUES (?,?,?,?)",
        (payload["time"],
         payload["environment"],
         payload["decision"],
         payload["reason"])
    )
    conn.commit()