"""SQLite state. History is append-only; nothing is ever deleted."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .config import DEFAULT_DB_PATH
from .models import Hit

SCHEMA = """
CREATE TABLE IF NOT EXISTS hits (
    dedupe_key          TEXT PRIMARY KEY,
    entity_name         TEXT,
    source_url          TEXT NOT NULL,
    source_type         TEXT NOT NULL,
    published_date      TEXT NOT NULL,
    tier                TEXT NOT NULL,
    matched_keywords    TEXT NOT NULL,
    verbatim_excerpt    TEXT NOT NULL,
    already_on_rfi_list TEXT NOT NULL,
    enabler_named       TEXT NOT NULL,
    source_name         TEXT,
    title               TEXT,
    excerpt_provenance  TEXT,
    first_seen_at       TEXT,
    alerted_at          TEXT,
    digested_at         TEXT
);

-- The same announcement is typically carried by four outlets. The hit
-- collapses to one row; every corroborating URL is kept here so no evidence
-- is lost to deduplication.
CREATE TABLE IF NOT EXISTS hit_sources (
    dedupe_key  TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    source_name TEXT,
    source_type TEXT,
    seen_at     TEXT,
    PRIMARY KEY (dedupe_key, source_url)
);

CREATE TABLE IF NOT EXISTS source_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name  TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL,   -- ok | down | error
    item_count   INTEGER DEFAULT 0,
    detail       TEXT
);

CREATE TABLE IF NOT EXISTS discards (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    reason     TEXT NOT NULL,
    payload    TEXT,
    at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hits_tier ON hits(tier);
CREATE INDEX IF NOT EXISTS idx_hits_date ON hits(published_date);
CREATE INDEX IF NOT EXISTS idx_hits_entity ON hits(entity_name);
"""


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_hit(conn: sqlite3.Connection, hit: Hit) -> str:
    """Insert a hit. Returns 'new' or 'duplicate'.

    A duplicate never re-alerts; it only records the extra source URL.
    """
    row = hit.to_row()
    key = row["dedupe_key"]
    existing = conn.execute(
        "SELECT dedupe_key FROM hits WHERE dedupe_key = ?", (key,)
    ).fetchone()

    conn.execute(
        """INSERT OR IGNORE INTO hit_sources
           (dedupe_key, source_url, source_name, source_type, seen_at)
           VALUES (?, ?, ?, ?, ?)""",
        (key, hit.source_url, hit.source_name, hit.source_type, _now()),
    )

    if existing:
        conn.commit()
        return "duplicate"

    conn.execute(
        """INSERT INTO hits (
               dedupe_key, entity_name, source_url, source_type, published_date,
               tier, matched_keywords, verbatim_excerpt, already_on_rfi_list,
               enabler_named, source_name, title, excerpt_provenance, first_seen_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            key, row["entity_name"], row["source_url"], row["source_type"],
            row["published_date"], row["tier"], row["matched_keywords"],
            row["verbatim_excerpt"], row["already_on_rfi_list"],
            row["enabler_named"], row["source_name"], row["title"],
            row["excerpt_provenance"], row["first_seen_at"],
        ),
    )
    conn.commit()
    return "new"


def record_discard(conn: sqlite3.Connection, reason: str, payload: str) -> None:
    conn.execute(
        "INSERT INTO discards (reason, payload, at) VALUES (?,?,?)",
        (reason, payload[:2000], _now()),
    )
    conn.commit()


def start_run(conn: sqlite3.Connection, source_name: str) -> int:
    cur = conn.execute(
        "INSERT INTO source_runs (source_name, started_at, status) VALUES (?,?,?)",
        (source_name, _now(), "running"),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection, run_id: int, status: str, items: int = 0, detail: str = ""
) -> None:
    conn.execute(
        """UPDATE source_runs
           SET finished_at = ?, status = ?, item_count = ?, detail = ?
           WHERE id = ?""",
        (_now(), status, items, detail[:2000], run_id),
    )
    conn.commit()


def pending_alerts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Tier A hits never alerted before. Re-runs must not re-alert."""
    return conn.execute(
        "SELECT * FROM hits WHERE tier = 'A' AND alerted_at IS NULL "
        "ORDER BY published_date DESC"
    ).fetchall()


def pending_digest(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM hits WHERE tier IN ('B','C') AND digested_at IS NULL "
        "ORDER BY tier, published_date DESC"
    ).fetchall()


def mark_alerted(conn: sqlite3.Connection, keys: Iterable[str]) -> None:
    conn.executemany(
        "UPDATE hits SET alerted_at = ? WHERE dedupe_key = ?",
        [(_now(), k) for k in keys],
    )
    conn.commit()


def mark_digested(conn: sqlite3.Connection, keys: Iterable[str]) -> None:
    conn.executemany(
        "UPDATE hits SET digested_at = ? WHERE dedupe_key = ?",
        [(_now(), k) for k in keys],
    )
    conn.commit()


def counts_by_tier(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT tier, COUNT(*) AS n FROM hits GROUP BY tier"
    ).fetchall()
    return {r["tier"]: r["n"] for r in rows}


def find_entity(conn: sqlite3.Connection, needle: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM hits WHERE entity_name LIKE ? ORDER BY published_date",
        (f"%{needle}%",),
    ).fetchall()


def known_entities(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT entity_name FROM hits WHERE entity_name IS NOT NULL"
    ).fetchall()
    return [r["entity_name"] for r in rows]


def failed_sources(conn: sqlite3.Connection, run_ids: list[int]) -> list[sqlite3.Row]:
    if not run_ids:
        return []
    marks = ",".join("?" * len(run_ids))
    return conn.execute(
        f"SELECT * FROM source_runs WHERE id IN ({marks}) AND status != 'ok'",
        run_ids,
    ).fetchall()
