"""PSX - dps.psx.com.pk material-information disclosures.

Listed companies must disclose telecom agreements here, which is where the
Zuma / Telna deal surfaced before the press carried it.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

from ..config import ALL_KEYWORDS
from .base import (
    Item, PdfUnreadable, Source, SourceDown, fetch, fetch_pdf_text,
    parse_date, within_window,
)

ANNOUNCEMENT_URLS = [
    "https://dps.psx.com.pk/announcements/companies",
    "https://dps.psx.com.pk/announcements",
]

_DOC_RE = re.compile(r"/download/document/\d+\.pdf", re.I)

# The announcements page lists hundreds of disclosures a day and the body of
# each lives in a PDF. Fetching every one would be wasteful and rude, so the
# row text is pre-filtered on keywords and only promising rows are opened.
_PREFILTER = [k.lower() for k in ALL_KEYWORDS] + [
    "telecom", "telecommunication", "connectivity", "mobile",
]


def _looks_relevant(text: str) -> bool:
    low = (text or "").lower()
    return any(term in low for term in _PREFILTER)


class _RowParser(HTMLParser):
    """Collect each <tr> as (cell texts, hrefs found in the row)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[list[str], list[str]]] = []
        self._in_row = False
        self._cells: list[str] = []
        self._hrefs: list[str] = []
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._in_row, self._cells, self._hrefs, self._buf = True, [], [], []
        elif tag in ("td", "th") and self._in_row:
            self._buf = []
        elif tag == "a" and self._in_row:
            href = dict(attrs).get("href")
            if href:
                self._hrefs.append(href)

    def handle_data(self, data):
        if self._in_row and data.strip():
            self._buf.append(data.strip())

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_row:
            self._cells.append(" ".join(self._buf).strip())
            self._buf = []
        elif tag == "tr" and self._in_row:
            if self._cells or self._hrefs:
                self.rows.append((self._cells, self._hrefs))
            self._in_row = False


class PSXSource(Source):
    name = "PSX"
    source_type = "PSX"
    cadence = "daily"

    def __init__(self, urls: Optional[list[str]] = None) -> None:
        self.urls = urls or ANNOUNCEMENT_URLS
        self.last_warnings: list[str] = []

    def collect(self, since: Optional[str] = None) -> list[Item]:
        self.last_warnings = []
        items: list[Item] = []
        failures: list[str] = []
        reached = 0

        for url in self.urls:
            try:
                html = fetch(url)
            except SourceDown as exc:
                failures.append(str(exc))
                continue
            reached += 1

            parser = _RowParser()
            try:
                parser.feed(html)
            except Exception as exc:
                failures.append(f"{url}: unparseable ({exc})")
                continue

            for cells, hrefs in parser.rows:
                if not cells:
                    continue
                text = " | ".join(c for c in cells if c)
                if not text:
                    continue

                doc = next((h for h in hrefs if _DOC_RE.search(h)), None)
                link = urljoin(url, doc) if doc else None
                if not link:
                    # Without a direct document link there is no resolvable
                    # source_url, and the hard gate would discard it anyway.
                    continue

                published = next(
                    (parse_date(c) for c in cells if parse_date(c)), None
                )
                if not within_window(published, since):
                    continue

                body, extra = text, {"format": "pdf", "body_is_row_text": True}
                if _looks_relevant(text):
                    # A material-information PDF is the primary record; the
                    # listing row is only an index entry.
                    try:
                        body = fetch_pdf_text(link)
                        extra = {"format": "pdf", "body_is_row_text": False}
                    except (PdfUnreadable, SourceDown) as exc:
                        failures.append(f"PDF unreadable: {exc}")

                items.append(Item(
                    title=cells[0] if cells else "PSX disclosure",
                    url=link, body=body,
                    source_name=self.name, source_type=self.source_type,
                    published_date=published, extra=extra,
                ))

        if failures and reached:
            self.last_warnings = failures

        if reached == 0:
            raise SourceDown(f"PSX: no announcement page reachable :: {' | '.join(failures)}")
        return items
