"""Command line interface.

    python -m mvno_watcher backfill --months 12
    python -m mvno_watcher run --sources PTA PSX press
    python -m mvno_watcher digest
    python -m mvno_watcher report --entity "Zuma"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db, pipeline
from .config import DEFAULT_DB_PATH
from .transport import build_chain, set_chain
from .entities import load_rfi_list
from .sources import all_sources, sources_for
from .sources.base import SourceDown, fetch, months_ago

BACKFILL_TOO_MANY = 50


def _conn(args):
    return db.connect(args.db)


def _apply_transport(args) -> None:
    """Install the fetch chain chosen on the command line."""
    spec = getattr(args, "transports", None)
    if spec:
        set_chain(build_chain(spec))
    fixture_dir = getattr(args, "save_fixtures", None)
    if fixture_dir:
        import os
        os.environ["MVNO_SAVE_FIXTURES"] = fixture_dir


def cmd_init(args) -> int:
    conn = _conn(args)
    print(f"database ready: {args.db}")
    print(f"RFI list entries: {len(load_rfi_list())}")
    conn.close()
    return 0


def cmd_run(args) -> int:
    _apply_transport(args)
    conn = _conn(args)
    sources = sources_for(args.sources, include_extra_press=args.extra_press)
    print(f"running {len(sources)} source(s): {', '.join(s.name for s in sources)}")
    report = pipeline.run(
        conn, sources, since=args.since,
        fire_alerts=not args.no_alerts, send=args.send_slack,
    )
    print(report.summary())
    conn.close()
    return 1 if report.any_source_down else 0


def cmd_backfill(args) -> int:
    """Mandatory pre-live validation: 12 months across all sources."""
    _apply_transport(args)
    conn = _conn(args)
    since = months_ago(args.months)
    sources = sources_for(args.sources, include_extra_press=args.extra_press)
    print(f"BACKFILL: {args.months} months (since {since}) across "
          f"{len(sources)} source(s)\n")

    # A backfill never alerts. A human approves the result first.
    # A backfill never delivers: a human approves the result first.
    report = pipeline.run(conn, sources, since=since, fire_alerts=False,
                          send=False)
    print(report.summary())
    print()
    print(_backfill_verdict(conn, report))
    conn.close()
    return 1 if report.any_source_down else 0


def _backfill_verdict(conn, report) -> str:
    counts = db.counts_by_tier(conn)
    total = sum(counts.values())
    lines = [
        "=" * 62,
        "BACKFILL VALIDATION REPORT",
        "=" * 62,
        f"Tier A : {counts.get('A', 0)}",
        f"Tier B : {counts.get('B', 0)}",
        f"Tier C : {counts.get('C', 0)}",
        f"TOTAL  : {total}",
        "",
    ]

    zuma = db.find_entity(conn, "Zuma")
    if zuma:
        tiers = sorted({r["tier"] for r in zuma})
        lines.append(f"Zuma Resources: PRESENT as Tier {', '.join(tiers)} "
                     f"({len(zuma)} hit(s))")
        for r in zuma:
            lines.append(f"  [{r['tier']}] {r['published_date']} {r['source_type']} "
                         f"{r['source_url']}")
    else:
        lines.append("Zuma Resources: ABSENT - the known-positive control did "
                     "not surface. Treat the backfill as failed and check "
                     "source coverage before trusting a zero.")
    lines.append("")

    if report.any_source_down:
        lines.append("VERDICT: INCONCLUSIVE - one or more sources were down. "
                     "Coverage was partial, so neither a zero nor a count can "
                     "be trusted. Fix the sources and re-run.")
    elif total == 0:
        lines.append("VERDICT: ZERO HITS - the market genuinely has no signal "
                     "yet. The watcher is a 2027 asset. Stop here; do not "
                     "schedule live runs.")
    elif total > BACKFILL_TOO_MANY:
        lines.append(f"VERDICT: TOO LOOSE - {total} hits exceeds the {BACKFILL_TOO_MANY} "
                     "threshold. Tighten the keyword filter (start with the "
                     "WEAK_KEYWORDS set in config.py) and re-run.")
    else:
        lines.append(f"VERDICT: IN RANGE - {total} hits (1-{BACKFILL_TOO_MANY}). "
                     "Await human approval before scheduling live runs.")

    lines.append("")
    lines.append("Live runs remain unscheduled until a human approves this "
                 "result.")
    lines.append("=" * 62)
    return "\n".join(lines)


def cmd_digest(args) -> int:
    conn = _conn(args)
    n = pipeline.send_weekly_digest(conn, send=args.send_slack)
    print(f"digest sent covering {n} Tier B/C hit(s)")
    conn.close()
    return 0


def cmd_report(args) -> int:
    conn = _conn(args)
    if args.entity:
        rows = db.find_entity(conn, args.entity)
    else:
        rows = conn.execute(
            "SELECT * FROM hits ORDER BY tier, published_date DESC"
        ).fetchall()

    if not rows:
        print("no hits")
        conn.close()
        return 0

    if args.json:
        print(json.dumps([dict(r) for r in rows], indent=2, ensure_ascii=False))
        conn.close()
        return 0

    for r in rows:
        print("-" * 62)
        print(f"entity_name         : {r['entity_name'] or '(none - Tier C)'}")
        print(f"source_url          : {r['source_url']}")
        print(f"source_type         : {r['source_type']}")
        print(f"published_date      : {r['published_date']}")
        print(f"tier                : {r['tier']}")
        print(f"matched_keywords    : {r['matched_keywords']}")
        print(f"verbatim_excerpt    : {r['verbatim_excerpt'][:300]}")
        print(f"already_on_RFI_list : {r['already_on_rfi_list']}")
        print(f"enabler_named       : {r['enabler_named']}")
        print(f"excerpt_provenance  : {r['excerpt_provenance']}")
    print("-" * 62)
    print(f"{len(rows)} hit(s); tiers: {db.counts_by_tier(conn)}")
    conn.close()
    return 0


def cmd_ingest(args) -> int:
    conn = _conn(args)
    records = json.loads(Path(args.file).read_text(encoding="utf-8"))
    report = pipeline.ingest_records(
        conn, records, provenance=args.provenance,
        fire_alerts=False, send=False,
    )
    print(report.summary())
    print()
    print(_backfill_verdict(conn, report))
    conn.close()
    return 0


def cmd_doctor(args) -> int:
    """Check every source endpoint without writing anything."""
    _apply_transport(args)
    from .sources.press import FEEDS
    from .sources.psx import ANNOUNCEMENT_URLS
    from .sources.pta import INDEX_URLS as PTA_URLS
    from .sources.secp import INDEX_URLS as SECP_URLS
    from .sources.events import EXHIBITOR_URLS

    checks = (
        [("PTA", u) for u in PTA_URLS]
        + [("SECP", u) for u in SECP_URLS]
        + [("PSX", u) for u in ANNOUNCEMENT_URLS]
        + [(n, u) for n, u in FEEDS.items()]
        + [("ITCN Asia", u) for u in EXHIBITOR_URLS]
    )
    down = 0
    for name, url in checks:
        try:
            body = fetch(url, timeout=20, retries=1)
            print(f"  ok    {name:<24} {url} ({len(body)} bytes)")
        except SourceDown as exc:
            down += 1
            print(f"  DOWN  {name:<24} {url}\n        SOURCE DOWN: {name} :: {exc}")
    print(f"\n{len(checks) - down}/{len(checks)} endpoints reachable")
    return 1 if down else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mvno_watcher",
        description="Pakistan MVNO regulatory watcher - surfaces named "
                    "entities and a source URL to a human. Never contacts "
                    "anyone, never posts.",
    )
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite path")
    sub = p.add_subparsers(dest="command", required=True)

    p.add_argument("--transports", default=None,
                   help="fetch chain, e.g. 'direct' or 'fixture' "
                        "(default: $MVNO_TRANSPORTS or 'direct')")
    p.add_argument("--save-fixtures", default=None, metavar="DIR",
                   help="save every fetched response to DIR for offline replay")

    def common(sp):
        sp.add_argument("--send-slack", action="store_true",
                        help="actually post to Slack. OFF by default: without "
                             "this flag alerts are only printed")
        sp.add_argument("--sources", nargs="*", default=None,
                        help="PTA SECP PSX press event, or an outlet name")
        sp.add_argument("--extra-press", action="store_true",
                        help="include PhoneWorld and TechJuice feeds")

    sp = sub.add_parser("init", help="create the database")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("run", help="one collection cycle")
    common(sp)
    sp.add_argument("--since", default=None, help="ISO date lower bound")
    sp.add_argument("--no-alerts", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("backfill", help="mandatory 12-month pre-live validation")
    common(sp)
    sp.add_argument("--months", type=int, default=12)
    sp.set_defaults(func=cmd_backfill)

    sp = sub.add_parser("digest", help="Monday Tier B/C digest")
    sp.add_argument("--send-slack", action="store_true",
                    help="actually post to Slack (off by default)")
    sp.set_defaults(func=cmd_digest)

    sp = sub.add_parser("report", help="print stored hits")
    sp.add_argument("--entity", default=None)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("ingest", help="feed a JSON record file through the gates")
    sp.add_argument("file")
    sp.add_argument("--provenance", default="manual",
                    choices=["fetched", "search_summary", "manual"])
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("doctor", help="check every source endpoint")
    sp.set_defaults(func=cmd_doctor)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
