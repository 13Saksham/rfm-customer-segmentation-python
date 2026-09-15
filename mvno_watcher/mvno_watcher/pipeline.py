"""Orchestration: collect -> match -> gate -> persist -> alert."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import alerts, db
from .config import HOST_OPERATORS, KNOWN_ENABLERS
from .entities import detect_enablers, load_rfi_list, on_rfi_list
from .matcher import classify
from .models import DiscardedHit, Hit
from .sources import Item, Source, SourceDown


@dataclass
class RunReport:
    sources_ok: list[str] = field(default_factory=list)
    sources_down: list[tuple[str, str]] = field(default_factory=list)
    items_seen: int = 0
    new_hits: int = 0
    duplicates: int = 0
    discarded: list[tuple[str, str]] = field(default_factory=list)
    alerts_fired: int = 0
    sent: bool = False
    tier_counts: dict[str, int] = field(default_factory=dict)
    rfi_list_configured: bool = True

    @property
    def any_source_down(self) -> bool:
        return bool(self.sources_down)

    def summary(self) -> str:
        lines = [
            f"sources ok       : {len(self.sources_ok)} ({', '.join(self.sources_ok) or '-'})",
            f"sources DOWN     : {len(self.sources_down)}",
        ]
        for name, detail in self.sources_down:
            lines.append(f"  SOURCE DOWN: {name} :: {detail[:160]}")
        lines += [
            f"items examined   : {self.items_seen}",
            f"new hits         : {self.new_hits}",
            f"duplicates       : {self.duplicates}",
            f"discarded (gates): {len(self.discarded)}",
            f"tier counts (db) : {self.tier_counts}",
            f"tier A alerts    : {self.alerts_fired} "
            f"({'posted to Slack' if self.sent else 'printed only, not sent'})",
        ]
        if not self.rfi_list_configured:
            lines.append(
                "  WARNING: config/rfi_list.txt is empty - already_on_RFI_list "
                "recorded as 'unknown' for every hit."
            )
        return "\n".join(lines)


def _hit_from_item(
    item: Item,
    rfi: list[str],
    known_names: list[str],
    provenance: str = "fetched",
    fallback_names: list[str] | None = None,
) -> Optional[Hit]:
    match = classify(
        item.title, item.body, item.source_type, known_names, fallback_names
    )
    if match is None:
        return None

    enablers = match.enablers or detect_enablers(f"{item.title} {item.body}")
    return Hit(
        entity_name=match.entity,
        source_url=item.url,
        source_type=item.source_type,
        published_date=item.published_date or "",
        tier=match.tier,
        matched_keywords=match.keywords,
        verbatim_excerpt=match.excerpt,
        already_on_rfi_list=on_rfi_list(match.entity, rfi),
        enabler_named=", ".join(enablers) if enablers else "no",
        source_name=item.source_name,
        title=item.title,
        excerpt_provenance=provenance,
    )


def process_items(
    conn: sqlite3.Connection,
    items: Iterable[Item],
    report: RunReport,
    rfi: list[str],
    provenance: str = "fetched",
) -> None:
    counterparties = {*KNOWN_ENABLERS, *HOST_OPERATORS}
    # Candidate subjects, with counterparties removed so an enabler that
    # once appeared as an entity cannot outrank a real subject later.
    known = [n for n in {*rfi, *db.known_entities(conn)} if n not in counterparties]
    fallback = sorted(counterparties)
    for item in items:
        report.items_seen += 1
        hit = _hit_from_item(item, rfi, known, provenance, fallback)
        if hit is None:
            continue
        try:
            hit.validate()
        except DiscardedHit as exc:
            # Discarded, not logged as a hit - but the discard itself is
            # recorded so a systematic gate failure is visible.
            report.discarded.append((item.url or "(no url)", str(exc)))
            db.record_discard(conn, str(exc), json.dumps({
                "title": item.title, "url": item.url,
                "source": item.source_name,
            }))
            continue
        outcome = db.upsert_hit(conn, hit)
        if outcome == "new":
            report.new_hits += 1
            if hit.entity_name:
                known.append(hit.entity_name)
        else:
            report.duplicates += 1


def run(
    conn: sqlite3.Connection,
    sources: list[Source],
    since: Optional[str] = None,
    fire_alerts: bool = True,
    send: bool = False,
) -> RunReport:
    """Collect from every source, persist hits, fire Tier A alerts."""
    rfi = load_rfi_list()
    report = RunReport(rfi_list_configured=bool(rfi))

    for source in sources:
        run_id = db.start_run(conn, source.name)
        try:
            items = source.collect(since=since)
        except SourceDown as exc:
            # Fail loudly: never run with partial coverage and report zero.
            db.finish_run(conn, run_id, "down", 0, str(exc))
            report.sources_down.append((source.name, str(exc)))
            alerts.send_source_down(source.name, str(exc), send=send)
            continue
        except Exception as exc:  # unexpected parse/logic failure
            db.finish_run(conn, run_id, "error", 0, f"{type(exc).__name__}: {exc}")
            report.sources_down.append((source.name, f"{type(exc).__name__}: {exc}"))
            alerts.send_source_down(source.name, f"{type(exc).__name__}: {exc}", send=send)
            continue

        db.finish_run(conn, run_id, "ok", len(items))
        report.sources_ok.append(source.name)

        for warning in getattr(source, "last_warnings", []) or []:
            alerts.send_source_down(f"{source.name} (partial)", warning, send=send)

        process_items(conn, items, report, rfi)

    if fire_alerts:
        report.alerts_fired = fire_tier_a(conn, send=send)
        report.sent = send

    report.tier_counts = db.counts_by_tier(conn)
    return report


def fire_tier_a(conn: sqlite3.Connection, send: bool = False) -> int:
    """Alert every unalerted Tier A hit exactly once."""
    rows = db.pending_alerts(conn)
    sent = []
    for row in rows:
        alerts.send_tier_a(row, send=send)
        sent.append(row["dedupe_key"])
    if sent:
        db.mark_alerted(conn, sent)
    return len(sent)


def send_weekly_digest(conn: sqlite3.Connection, send: bool = False) -> int:
    """Monday digest of Tier B and Tier C, once per hit."""
    rows = db.pending_digest(conn)
    alerts.send_digest(rows, send=send)
    if rows:
        db.mark_digested(conn, [r["dedupe_key"] for r in rows])
    return len(rows)


def ingest_records(
    conn: sqlite3.Connection,
    records: list[dict],
    provenance: str = "manual",
    fire_alerts: bool = False,
    send: bool = False,
) -> RunReport:
    """Feed externally-collected records through the same matcher and gates.

    Used for the backfill when direct fetching is unavailable. Nothing
    bypasses classification or the hard gates; only the provenance label
    differs, so unverified excerpts stay distinguishable in the database.
    """
    rfi = load_rfi_list()
    report = RunReport(rfi_list_configured=bool(rfi))
    items = [
        Item(
            title=r.get("title", ""),
            url=r.get("source_url", ""),
            body=r.get("body", ""),
            source_name=r.get("source_name", ""),
            source_type=r.get("source_type", "press"),
            published_date=r.get("published_date"),
        )
        for r in records
    ]
    process_items(conn, items, report, rfi, provenance=provenance)
    if fire_alerts:
        report.alerts_fired = fire_tier_a(conn, send=send)
        report.sent = send
    report.tier_counts = db.counts_by_tier(conn)
    return report
