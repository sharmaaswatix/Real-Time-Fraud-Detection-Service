import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    username TEXT NOT NULL,
    amount REAL NOT NULL,
    input_json TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    threshold REAL NOT NULL,
    flagged INTEGER NOT NULL,
    risk_label TEXT NOT NULL
)"""


@contextmanager
def connect():
    path = os.getenv("FRAUD_DB", "data/fraud.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_prediction(username, tx: dict, proba, threshold, flagged, label) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO predictions (created_at, username, amount, input_json, fraud_probability,"
            " threshold, flagged, risk_label) VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), username, tx["Amount"], json.dumps(tx),
             proba, threshold, int(flagged), label))
        return cur.lastrowid


def list_predictions(limit=50, flagged_only=False) -> list[dict]:
    sql = "SELECT * FROM predictions"
    if flagged_only:
        sql += " WHERE flagged = 1"
    sql += " ORDER BY id DESC LIMIT ?"
    with connect() as c:
        rows = c.execute(sql, (limit,)).fetchall()
    return [{**dict(r), "flagged": bool(r["flagged"])} for r in rows]
