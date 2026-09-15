"""PTA - pta.gov.pk. The highest-value source.

Fewer than ten MVNO licences are expected to exist, so a single PTA page can
carry the entire market. Index pages are configurable because the PTA site
is periodically restructured; a 404 on one index raises SourceDown rather
than quietly reducing coverage.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from ..entities import extract_all_entities
from .base import (
    Item, PdfUnreadable, Source, SourceDown, extract_links, fetch,
    fetch_pdf_text, html_to_text, parse_date, within_window,
)

INDEX_URLS = [
    "https://www.pta.gov.pk/en/media-center/press-releases",
    "https://www.pta.gov.pk/en/media-center/news",
    "https://www.pta.gov.pk/category/tenders",
    "https://www.pta.gov.pk/en/licensing",
    "https://www.pta.gov.pk/en/licensing/licensees",
    # The licensing category page is where a new register gets linked.
    "https://www.pta.gov.pk/category/licensing-721279833-2023-05-30",
]

#: Documents worth polling directly, whether or not an index links them.
#: Extend this with a register URL the moment one is published.
DIRECT_DOCUMENT_URLS = [
    "https://pta.gov.pk/assets/media/2026-01-06-Approved-MVNO-POLICY-FRAMEWORK_Dec-2025-PDF.pdf",
]

# PTA article slugs carry their own publication date, e.g.
# .../pta-opens-...-licensing-in-pakistan-1588693749-2026-06-28
# /assets/media/ must be included: PTA serves its licensee registers and
# policy documents from there as PDFs, and excluding that path silently drops
# the most valuable documents on the site.
_ARTICLE_RE = re.compile(r"pta\.gov\.pk/(?:category|en|assets/media)/", re.I)
_DATED_SLUG = re.compile(r"20\d{2}-\d{2}-\d{2}\s*$")

MAX_ARTICLES_PER_INDEX = 40

# PTA publishes its licence registers as PDFs under a stable path, e.g.
#   /assets/media/2025-01-03-List-of-CVAS-Licensees-02012025.pdf
# An MVNO equivalent is the single highest-value document this watcher can
# read: it names every licensee at once. Registers are always fetched and are
# exempt from the date window, because a register is current state, not news.
_REGISTER_RE = re.compile(
    r"(?:"
    r"licensees?"                  # ...-List-of-CVAS-Licensees-...
    r"|lic[-_](?:list|pak|ajkgb)"  # ldi_lic_list_, sr7_ldi_lic_pak_, ldi-lic-ajkgb-
       # Deliberately not a bare "lic": lic_template_annex-f.pdf is a licence
       # template, not a register, and parsing it row-wise yields nothing.
    r"|[-_]list[-_]"               # fll_list_pak_..., cvas_list_...
    r"|list[-_]of"                 # list-of-new-and-converted-cvas-...
    r")",
    re.I,
)


def _is_register(url: str, link_text: str) -> bool:
    return bool(_REGISTER_RE.search(url) or _REGISTER_RE.search(link_text or ""))


def _register_rows(text: str) -> list[tuple[str, str]]:
    """Split a register into (entity, row text) pairs.

    A register naming eight licensees must yield eight hits, not one:
    collapsing it to a single entity would discard most of the answer.
    """
    rows: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if len(line) < 4:
            continue
        for entity in extract_all_entities(line):
            rows.append((entity, line))
    return rows


class PTASource(Source):
    name = "PTA"
    source_type = "PTA"
    cadence = "daily"

    def __init__(
        self,
        index_urls: Optional[list[str]] = None,
        direct_documents: Optional[list[str]] = None,
    ) -> None:
        self.index_urls = index_urls or INDEX_URLS
        self.direct_documents = (
            DIRECT_DOCUMENT_URLS if direct_documents is None else direct_documents
        )
        self.last_warnings: list[str] = []

    def collect(self, since: Optional[str] = None) -> list[Item]:
        self.last_warnings = []
        seen: set[str] = set()
        items: list[Item] = []
        failures: list[str] = []
        reached = 0

        for doc_url in self.direct_documents:
            if doc_url in seen:
                continue
            seen.add(doc_url)
            try:
                text = fetch_pdf_text(doc_url)
            except (PdfUnreadable, SourceDown) as exc:
                self.last_warnings.append(f"direct document unreadable: {exc}")
                continue
            reached += 1
            items.append(Item(
                title=doc_url.rsplit("/", 1)[-1], url=doc_url, body=text,
                source_name=self.name, source_type=self.source_type,
                published_date=parse_date(doc_url) or parse_date(text[:400]),
                extra={"format": "pdf", "direct": True},
            ))

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
                if not within_window(published, since) and not _is_register(url, link_text):
                    continue
                is_register = _is_register(url, link_text)
                if is_register and not published:
                    published = parse_date(url) or date.today().isoformat()

                if url.lower().endswith(".pdf"):
                    title = link_text or url.rsplit("/", 1)[-1]
                    try:
                        pdf_text = fetch_pdf_text(url)
                    except (PdfUnreadable, SourceDown) as exc:
                        # Never silent: an unreadable PDF is a coverage gap.
                        self.last_warnings.append(f"PDF unreadable: {exc}")
                        items.append(Item(
                            title=title, url=url, body=link_text,
                            source_name=self.name, source_type=self.source_type,
                            published_date=published,
                            extra={"format": "pdf", "body_is_link_text": True},
                        ))
                        continue

                    if is_register:
                        rows = _register_rows(pdf_text)
                        for entity, row in rows:
                            items.append(Item(
                                title=title, url=url,
                                # Row first: the licensee row is the evidence,
                                # so it is what the verbatim excerpt should
                                # quote. The header is appended for the licence
                                # context the row itself may omit.
                                body=f"{row} - from {title}",
                                source_name=self.name,
                                source_type=self.source_type,
                                published_date=published,
                                extra={"format": "pdf", "register": True,
                                       "entity_hint": entity},
                            ))
                        if not rows:
                            self.last_warnings.append(
                                f"register named no companies: {url}"
                            )
                        continue

                    items.append(Item(
                        title=title, url=url, body=pdf_text,
                        source_name=self.name, source_type=self.source_type,
                        published_date=published or parse_date(pdf_text[:400]),
                        extra={"format": "pdf"},
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
