"""Fetching primitives shared by every source.

No third-party HTML/RSS dependency: the standard library is enough and it
keeps the watcher deployable anywhere without a build step.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from typing import Iterable, Optional
from urllib.parse import urljoin
from xml.etree import ElementTree

from ..transport import USER_AGENT, TransportError, get_chain

try:
    import pypdf
except Exception:  # pragma: no cover
    pypdf = None


class SourceDown(Exception):
    """A source could not be read. Never swallowed: it becomes an alert."""


class PdfUnreadable(Exception):
    """A PDF was fetched but its text could not be extracted.

    Not fatal, but never silent: PTA publishes licensee registers and PSX
    publishes material-information disclosures as PDFs, so an unreadable PDF
    is a real coverage gap and is reported as a source warning.
    """


@dataclass
class Item:
    """One candidate document before matching."""

    title: str
    url: str
    body: str
    source_name: str
    source_type: str
    published_date: Optional[str] = None
    extra: dict = field(default_factory=dict)


# --- HTTP ----------------------------------------------------------------

def fetch(url: str, timeout: int = 30, retries: int = 3) -> str:
    """GET a URL through the configured transport chain.

    Raises SourceDown when every transport failed, naming each route tried so
    a coverage gap is attributable rather than mysterious.
    """
    try:
        body = get_chain().get(url, timeout=timeout)
    except TransportError as exc:
        raise SourceDown(str(exc)) from exc

    if os.environ.get("MVNO_SAVE_FIXTURES"):
        # Record the bytes so this run can be replayed offline later.
        from ..transport import FixtureTransport
        FixtureTransport(os.environ["MVNO_SAVE_FIXTURES"]).save(url, body)
    return body


def fetch_pdf_text(url: str, timeout: int = 45) -> str:
    """Fetch a PDF and return its text.

    This matters more than it looks: the PTA publishes its licensee registers
    as PDFs and the PSX serves disclosures the same way, so the single most
    valuable document this watcher can read is a PDF.
    """
    if pypdf is None:
        raise PdfUnreadable("pypdf is not installed; cannot read PDF bodies")
    try:
        data = get_chain().get_bytes(url, timeout=timeout)
    except TransportError as exc:
        raise SourceDown(str(exc)) from exc

    if os.environ.get("MVNO_SAVE_FIXTURES"):
        from ..transport import FixtureTransport
        FixtureTransport(os.environ["MVNO_SAVE_FIXTURES"]).save_bytes(url, data)

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise PdfUnreadable(f"{url}: {type(exc).__name__}: {exc}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise PdfUnreadable(f"{url}: no extractable text (likely a scan)")
    return text


# --- HTML ----------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: Optional[str] = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._href = href
                self._text = []

    def handle_data(self, data):
        if self._href is not None and data.strip():
            self._text.append(data.strip())

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text)))
            self._href, self._text = None, []


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        return []
    out = []
    for href, text in parser.links:
        href = unescape(href.strip())
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        out.append((urljoin(base_url, href), text))
    return out


# --- dates ---------------------------------------------------------------

_SLUG_DATE = re.compile(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})")
_TEXT_DATE = re.compile(
    r"\b(\d{1,2})[-\s]+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s,]+"
    r"(20\d{2})\b",
    re.I,
)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_date(value: str) -> Optional[str]:
    """Best-effort ISO date from a URL slug, RSS pubDate, or visible text."""
    if not value:
        return None
    value = value.strip()

    try:
        return parsedate_to_datetime(value).date().isoformat()
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d",
                "%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value[:30].strip(), fmt).date().isoformat()
        except ValueError:
            continue

    m = _SLUG_DATE.search(value)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass

    m = _TEXT_DATE.search(value)
    if m:
        d, mon, y = m.group(1), m.group(2)[:3].lower(), m.group(3)
        try:
            return date(int(y), _MONTHS[mon], int(d)).isoformat()
        except (ValueError, KeyError):
            pass

    try:
        return datetime.fromisoformat(value[:19]).date().isoformat()
    except ValueError:
        return None


def within_window(iso_date: Optional[str], since: Optional[str]) -> bool:
    if not since:
        return True
    if not iso_date:
        return True   # undated items are kept; a human decides
    return iso_date >= since


def months_ago(months: int) -> str:
    return (date.today() - timedelta(days=int(months * 30.44))).isoformat()


# --- RSS -----------------------------------------------------------------

def parse_rss(xml_text: str, source_name: str, source_type: str) -> list[Item]:
    try:
        root = ElementTree.fromstring(xml_text.encode("utf-8", "ignore"))
    except ElementTree.ParseError as exc:
        raise SourceDown(f"{source_name}: unparseable feed ({exc})") from exc

    items: list[Item] = []
    nodes = root.iter("item")
    for node in nodes:
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        desc = (node.findtext("description") or "")
        pub = node.findtext("pubDate") or node.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
        if not link:
            continue
        items.append(Item(
            title=unescape(title),
            url=link,
            body=html_to_text(unescape(desc)),
            source_name=source_name,
            source_type=source_type,
            published_date=parse_date(pub),
        ))

    if not items:  # Atom
        ns = "{http://www.w3.org/2005/Atom}"
        for node in root.iter(f"{ns}entry"):
            title = (node.findtext(f"{ns}title") or "").strip()
            link_el = node.find(f"{ns}link")
            link = link_el.get("href") if link_el is not None else ""
            summary = node.findtext(f"{ns}summary") or node.findtext(f"{ns}content") or ""
            pub = node.findtext(f"{ns}updated") or node.findtext(f"{ns}published") or ""
            if not link:
                continue
            items.append(Item(
                title=unescape(title), url=link,
                body=html_to_text(unescape(summary)),
                source_name=source_name, source_type=source_type,
                published_date=parse_date(pub),
            ))
    return items


class Source:
    """Base class. Subclasses implement collect()."""

    name: str = "unnamed"
    source_type: str = "press"
    cadence: str = "daily"

    def collect(self, since: Optional[str] = None) -> list[Item]:
        raise NotImplementedError
