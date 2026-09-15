"""ITCN Asia exhibitor and sponsor lists, year on year.

Monthly cadence. An exhibitor list is a weak signal on its own, which is why
nothing here is promoted above the tier the matcher assigns it.
"""

from __future__ import annotations

from typing import Optional

from .base import (
    Item, Source, SourceDown, extract_links, fetch, html_to_text, parse_date,
)

EXHIBITOR_URLS = [
    "https://itcnasia.com/karachi/exhibitors/",
    "https://itcnasia.com/karachi/",
    "https://itcnasia.com/home/",
]


class EventsSource(Source):
    name = "ITCN Asia"
    source_type = "event"
    cadence = "monthly"

    def __init__(self, urls: Optional[list[str]] = None) -> None:
        self.urls = urls or EXHIBITOR_URLS
        self.last_warnings: list[str] = []

    def collect(self, since: Optional[str] = None) -> list[Item]:
        self.last_warnings = []
        items: list[Item] = []
        reached = 0

        for url in self.urls:
            try:
                html = fetch(url)
            except SourceDown as exc:
                self.last_warnings.append(str(exc))
                continue
            reached += 1

            body = html_to_text(html)
            items.append(Item(
                title=f"ITCN Asia exhibitor listing ({url.rstrip('/').rsplit('/', 1)[-1]})",
                url=url, body=body,
                source_name=self.name, source_type=self.source_type,
                published_date=parse_date(body[:300]),
                extra={"listing": True},
            ))

            # Exhibitor profile pages, where individual companies are named.
            for link, text in extract_links(html, url):
                if "itcnasia.com" in link and "exhibitor" in link.lower() and link != url:
                    items.append(Item(
                        title=text or link.rsplit("/", 1)[-1],
                        url=link, body=text,
                        source_name=self.name, source_type=self.source_type,
                        published_date=None,
                    ))

        if reached == 0:
            raise SourceDown(
                f"ITCN Asia: no exhibitor page reachable :: "
                f"{' | '.join(self.last_warnings)}"
            )
        return items
