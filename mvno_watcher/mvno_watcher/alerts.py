"""Slack delivery.

The alert payload is exactly what the mandate allows: entity_name, tier, a
one-line excerpt, and source_url. No scoring, no recommendation, no drafted
outreach.

Delivery is OPT-IN. Every function here prints the alert and returns without
sending unless the caller passes send=True and a webhook is configured, so
the default behaviour of every command is to show what it would send. The
only outbound destination this module can ever reach is the operator's own
Slack webhook URL; it never contacts a person, a company, or any third party.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


class SlackNotConfigured(RuntimeError):
    pass


def _webhook() -> Optional[str]:
    return os.environ.get("MVNO_SLACK_WEBHOOK_URL") or os.environ.get(
        "SLACK_WEBHOOK_URL"
    )


def _post(text: str, send: bool = False) -> bool:
    """Print the alert; post to Slack only when explicitly told to.

    Delivery is opt-in and off by default. Nothing leaves this process unless
    the caller passes send=True (CLI: --send-slack) AND a webhook is
    configured. A watcher that posts because a default flipped is worse than
    one that stays quiet.
    """
    print(text)
    if not send:
        return False
    url = _webhook()
    if not url:
        print(
            "WARNING: --send-slack given but no webhook configured "
            "(set MVNO_SLACK_WEBHOOK_URL). Nothing was sent.",
            file=sys.stderr,
        )
        return False
    if requests is None:  # pragma: no cover
        raise SlackNotConfigured("requests is required to post to Slack")
    resp = requests.post(url, data=json.dumps({"text": text}),
                         headers={"Content-Type": "application/json"}, timeout=20)
    resp.raise_for_status()
    return True


def format_alert(entity_name: str, tier: str, excerpt: str, source_url: str) -> str:
    return (
        f"*{entity_name}*  [Tier {tier}]\n"
        f"{excerpt}\n"
        f"{source_url}"
    )


def send_tier_a(row, send: bool = False) -> bool:
    """Fire one immediate Tier A alert."""
    return _post(
        format_alert(
            row["entity_name"] or "(unnamed)",
            row["tier"],
            row["verbatim_excerpt"],
            row["source_url"],
        ),
        send,
    )


def send_digest(rows, send: bool = False) -> bool:
    """Monday digest of Tier B and Tier C."""
    if not rows:
        return _post("Pakistan MVNO watcher - weekly digest: no new Tier B/C hits.", send)
    lines = ["Pakistan MVNO watcher - weekly digest"]
    for tier in ("B", "C"):
        tier_rows = [r for r in rows if r["tier"] == tier]
        if not tier_rows:
            continue
        lines.append(f"\nTier {tier} ({len(tier_rows)})")
        for r in tier_rows:
            lines.append(
                f"- {r['entity_name'] or '(no company named)'}: "
                f"{r['verbatim_excerpt'][:160]} {r['source_url']}"
            )
    return _post("\n".join(lines), send)


def send_source_down(source_name: str, detail: str = "", send: bool = False) -> bool:
    """Fail loudly. A silent zero-hit run is worse than no run at all."""
    text = f"SOURCE DOWN: {source_name}"
    if detail:
        text += f"\n{detail[:300]}"
    return _post(text, send)
