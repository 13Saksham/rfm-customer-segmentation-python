"""PTA - pta.gov.pk. The highest-value source.

Fewer than ten MVNO licences are expected to exist, so a single PTA page can
carry the entire market. Index pages are configurable because the PTA site
is periodically restructured; a 404 on one index raises SourceDown rather
than quietly reducing coverage.
"""

from __future__ import annotations

import re
from typing import Optional

from .base import (
    Item, Source, SourceDown, extract_links, fetch, html_to_text,
    parse_date, within_window,
)

INDEX_URLS = [
    "https://www.pta.gov.pk/en/media-center/press-releases",
    "https://www.pta.gov.pk/en/media-center/news",
    "https://www.pta.gov.pk/category/tenders",
    "https://www.pta.gov.pk/en/licensing",
    "https://www.pta.gov.pk/en/licensing/licensees",
]

# PTA article slugs carry their own publication date, e.g.
# .../pta-opens-...-licensing-in-pakistan-1588693749-2026-06-28
_ARTICLE_RE = re.compile(r"pta\.gov\.pk/(?:category|en)/", re.I)
_DATED_SLUG = re.compile(r"20\d{2}-\d{2}-\d{2}\s*$")

MAX_ARTICLES_PER_INDEX = 40


class PTASource(Source):
    name = "PTA"
    source_type = "PTA"
    cadence = "daily"

    def __init__(self, index_urls: Optional[list[str]] = None) -> None:
        self.index_urls = index_urls or INDEX_URLS
        self.last_warnings: list[str] = []

    def collect(self, since: Optional[str] = None) -> list[Item]:
        self.last_warnings = []
        seen: set[str] = set()
        items: list[Item] = []
        failures: list[str] = []
        reached = 0

        for index_url in self.index_urls:
            try:
                html = fetch(index_url)
            except SourceDown as exc:
                failures.append(str(exc))
                continue
            reached += 1

            candidates = []
            for url, text in extract_links(html, index_url):
                if url in seen or not _ARTICLE_RE.search(url):
                    continue
                if url.rstrip("/") == index_url.rstrip("/"):
                    continue
                seen.add(url)
                candidates.append((url, text))

            for url, link_text in candidates[:MAX_ARTICLES_PER_INDEX]:
                published = parse_date(url) or parse_date(link_text)
                if not within_window(published, since):
                    continue
                if url.lower().endswith(".pdf"):
                    # A PDF link is still evidence: keep the link text as body
                    # rather than guessing at contents we cannot parse.
                    items.append(Item(
                        title=link_text or url.rsplit("/", 1)[-1],
                        url=url, body=link_text,
                        source_name=self.name, source_type=self.source_type,
                        published_date=published,
                        extra={"format": "pdf", "body_is_link_text": True},
                    ))
                    continue
                try:
                    body = html_to_text(fetch(url))
                except SourceDown:
                    continue
                items.append(Item(
                    title=link_text or url.rsplit("/", 1)[-1],
                    url=url, body=body,
                    source_name=self.name, source_type=self.source_type,
                    published_date=published or parse_date(body[:400]),
                ))

        # Every index failing means no PTA coverage at all. Fail loudly.
        # Some indexes read, some not: coverage is reduced, so say so.
        if failures and reached:
            self.last_warnings = failures

        if reached == 0:
            raise SourceDown(f"PTA: all index pages unreachable :: {' | '.join(failures)}")
        return items
