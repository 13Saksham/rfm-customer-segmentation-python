"""SECP - new incorporations, name changes and object changes.

COVERAGE LIMIT, stated plainly: SECP publishes no open feed of daily
incorporations. The registry lives behind the eZfile portal, and bulk
company-data extracts are a paid/registered service. This source therefore
covers what is genuinely public - SECP press releases and notices - and
exposes a hook for an authenticated eZfile export where the operator has
one. A run that only covers press releases reports 'partial', which the
pipeline surfaces rather than hides.
"""

from __future__ import annotations

from typing import Optional

from .base import (
    Item, Source, SourceDown, extract_links, fetch, html_to_text,
    parse_date, within_window,
)

INDEX_URLS = [
    "https://www.secp.gov.pk/media-center/press-releases/",
    "https://www.secp.gov.pk/media-center/notices/",
]

MAX_ARTICLES_PER_INDEX = 30


class SECPSource(Source):
    name = "SECP"
    source_type = "SECP"
    cadence = "weekly"

    #: Set to a local CSV/JSON export from eZfile to cover incorporations
    #: properly. Format: rows with name, date, and objects/description.
    registry_export_path: Optional[str] = None

    def __init__(self, index_urls: Optional[list[str]] = None) -> None:
        self.index_urls = index_urls or INDEX_URLS
        self.last_warnings: list[str] = []

    def collect(self, since: Optional[str] = None) -> list[Item]:
        self.last_warnings = []
        items: list[Item] = []
        failures: list[str] = []
        reached = 0
        seen: set[str] = set()

        for index_url in self.index_urls:
            try:
                html = fetch(index_url)
            except SourceDown as exc:
                failures.append(str(exc))
                continue
            reached += 1

            for url, text in extract_links(html, index_url)[:MAX_ARTICLES_PER_INDEX]:
                if url in seen or "secp.gov.pk" not in url:
                    continue
                seen.add(url)
                published = parse_date(url) or parse_date(text)
                if not within_window(published, since):
                    continue
                if url.lower().endswith(".pdf"):
                    items.append(Item(
                        title=text or url.rsplit("/", 1)[-1], url=url, body=text,
                        source_name=self.name, source_type=self.source_type,
                        published_date=published, extra={"format": "pdf"},
                    ))
                    continue
                try:
                    body = html_to_text(fetch(url))
                except SourceDown:
                    continue
                items.append(Item(
                    title=text or url.rsplit("/", 1)[-1], url=url, body=body,
                    source_name=self.name, source_type=self.source_type,
                    published_date=published or parse_date(body[:400]),
                ))

        # Some indexes read, some not: coverage is reduced, so say so.
        if failures and reached:
            self.last_warnings = failures

        if reached == 0:
            raise SourceDown(f"SECP: no index reachable :: {' | '.join(failures)}")
        return items
