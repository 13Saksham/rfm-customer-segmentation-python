"""The Hit record and its hard gates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

from .config import SOURCE_TYPES, TIERS


class DiscardedHit(Exception):
    """Raised when a candidate fails a hard gate and must not be logged."""


# A source_url must be a direct link, never a search page. These patterns
# exist because a search URL is not evidence: it is a promise that evidence
# might be found later.
_SEARCH_URL_PATTERNS = [
    re.compile(r"/search\b", re.I),
    re.compile(r"[?&](q|query|s|keyword|search)=", re.I),
    re.compile(r"^(www\.)?(google|bing|duckduckgo|yandex)\.", re.I),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


@dataclass
class Hit:
    """One matched item. Every mandatory field in the spec is required here."""

    entity_name: Optional[str]
    source_url: str
    source_type: str
    published_date: str          # ISO YYYY-MM-DD
    tier: str
    matched_keywords: list[str]
    verbatim_excerpt: str
    already_on_rfi_list: str     # yes | no | unknown
    enabler_named: str           # no | <enabler name(s)>

    # Provenance / bookkeeping (not part of the alert payload).
    source_name: str = ""
    title: str = ""
    excerpt_provenance: str = "fetched"   # fetched | search_summary | manual
    first_seen_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    # ---- gates ----------------------------------------------------------
    def validate(self) -> "Hit":
        """Apply the hard gates. Raises DiscardedHit on failure."""
        url = (self.source_url or "").strip()
        if not url:
            raise DiscardedHit("no source_url")

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise DiscardedHit(f"unresolvable source_url: {url!r}")

        for pat in _SEARCH_URL_PATTERNS:
            target = parsed.netloc if pat.pattern.startswith("^") else url
            if pat.search(target):
                raise DiscardedHit(f"source_url is a search page: {url!r}")

        if self.source_type not in SOURCE_TYPES:
            raise DiscardedHit(f"bad source_type: {self.source_type!r}")

        if self.tier not in TIERS:
            raise DiscardedHit(f"bad tier: {self.tier!r}")

        if not self.published_date:
            raise DiscardedHit("no published_date")
        try:
            date.fromisoformat(self.published_date)
        except ValueError as exc:
            raise DiscardedHit(f"bad published_date: {self.published_date!r}") from exc

        if not (self.verbatim_excerpt or "").strip():
            raise DiscardedHit("no verbatim_excerpt")

        if not self.matched_keywords:
            raise DiscardedHit("no matched_keywords")

        # Never infer an entity name: an unnamed item can only ever be Tier C.
        if not (self.entity_name or "").strip() and self.tier != "C":
            raise DiscardedHit(
                f"tier {self.tier} requires a named entity (spec: never infer)"
            )

        if self.already_on_rfi_list not in ("yes", "no", "unknown"):
            raise DiscardedHit(
                f"bad already_on_RFI_list: {self.already_on_rfi_list!r}"
            )

        return self

    # ---- identity -------------------------------------------------------
    @property
    def dedupe_key(self) -> str:
        """Dedupe by entity_name + published_date across all sources.

        The same announcement is typically carried by four outlets, so a
        named entity collapses across them. An unnamed (Tier C) item has no
        entity to collapse on, so it falls back to a normalised title, which
        keeps four write-ups of one policy note as one row while keeping two
        genuinely different notes apart.
        """
        if (self.entity_name or "").strip():
            basis = f"entity:{_norm(self.entity_name)}"
        else:
            basis = f"title:{_norm(self.title) or _norm(self.verbatim_excerpt)[:120]}"
        return hashlib.sha256(
            f"{basis}|{self.published_date}".encode("utf-8")
        ).hexdigest()

    def to_row(self) -> dict:
        d = asdict(self)
        d["matched_keywords"] = ", ".join(self.matched_keywords)
        d["dedupe_key"] = self.dedupe_key
        return d

    def alert_line(self) -> str:
        """One line excerpt for Slack. Nothing else goes in the alert."""
        excerpt = re.sub(r"\s+", " ", self.verbatim_excerpt).strip()
        return excerpt if len(excerpt) <= 240 else excerpt[:237] + "..."
