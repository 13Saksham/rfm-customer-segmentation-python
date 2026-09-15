"""Conservative entity-name extraction.

The spec is absolute: never infer an entity name. Everything here either
recognises a name the source actually wrote, or returns None so the item
falls to Tier C.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from .config import (
    CONFIG_DIR,
    CORPORATE_DESIGNATORS,
    ENTITY_STOPLIST,
    HOST_OPERATORS,
    KNOWN_ENABLERS,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def load_rfi_list(path: Path | None = None) -> list[str]:
    """Load the 26 known RFI companies. Missing file -> empty list.

    An empty list is not silently treated as 'no match'; callers mark the
    cross-check 'unknown' and warn, per the fail-loudly rule.
    """
    path = path or (CONFIG_DIR / "rfi_list.txt")
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


# A company name is 1-6 capitalised/alphanumeric tokens immediately followed
# by a corporate designator ("Zuma Resources Limited", "Telna North America,
# Inc."). Anything that does not carry a designator is only recognised if it
# is already on a known-name list.
_DESIGNATOR_ALT = "|".join(
    sorted((re.escape(d) for d in CORPORATE_DESIGNATORS), key=len, reverse=True)
)
# Intra-name separators are spaces and tabs only, never \s: \s matches
# newlines, which let a name run across the title/body join and produced the
# welded entity "Jazz Acme Digital Limited" from two different companies on
# two different lines.
_COMPANY_RE = re.compile(
    r"\b((?:[A-Z][\w&.'-]*[ \t]+){0,5}(?:[A-Z][\w&.'-]*)[ \t,]+(?:"
    + _DESIGNATOR_ALT + r"))\b"
)

_TRAILING_JUNK = re.compile(r"[\s,;:.\-]+$")
# A possessive ('Asia's Largest ...') is headline prose, never a legal name.
_POSSESSIVE = re.compile(r"\w[\u2019']s\b")


def _clean(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    return _TRAILING_JUNK.sub("", name)


def _is_stoplisted(name: str) -> bool:
    n = _norm(name)
    if n in ENTITY_STOPLIST:
        return True
    # "Pakistan Telecommunication Authority (PTA)" and similar wrappers.
    return any(stop in n for stop in ENTITY_STOPLIST if len(stop) > 6)


def find_known_names(text: str, known: Iterable[str]) -> list[str]:
    """Return known names that appear literally in the text."""
    found = []
    for name in known:
        if not name:
            continue
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, text, re.I):
            found.append(name)
    return found


def extract_entity(
    text: str,
    known_names: Iterable[str] = (),
    fallback_names: Iterable[str] = (),
) -> Optional[str]:
    """Return the single most likely named company, or None.

    Priority:
      1. a candidate name (RFI list, previously-seen entity) - the subject we
         are actually hunting;
      2. a name carrying a legal suffix, read straight out of the text;
      3. fallback names (enablers, host operators) - only when nothing else
         named a company.

    Step 3 is last on purpose. An enabler or host operator named in a story is
    usually the counterparty: "Zuma Resources Limited signed with Telna" is a
    fact about Zuma. Promoting the enabler would attach the fact to the wrong
    company record, which is the exact failure this watcher exists to avoid.
    """
    text = text or ""

    for name in find_known_names(text, known_names):
        if not _is_stoplisted(name):
            return name

    for match in _COMPANY_RE.finditer(text):
        candidate = _clean(match.group(1))
        if _is_stoplisted(candidate) or _POSSESSIVE.search(candidate):
            continue
        # Reject one-token names that are just the designator itself.
        if len(candidate.split()) < 2:
            continue
        return candidate

    for name in find_known_names(text, fallback_names):
        if not _is_stoplisted(name):
            return name

    return None


def detect_enablers(text: str) -> list[str]:
    """Find MVNE/enabler names. Captured, never used to filter."""
    return find_known_names(text or "", KNOWN_ENABLERS)


def detect_host_operators(text: str) -> list[str]:
    return find_known_names(text or "", HOST_OPERATORS)


def on_rfi_list(entity: Optional[str], rfi: list[str]) -> str:
    """yes | no | unknown. 'unknown' when the RFI list is not configured."""
    if not rfi:
        return "unknown"
    if not entity:
        return "no"
    e = _norm(entity)
    for name in rfi:
        n = _norm(name)
        if n and (n in e or e in n):
            return "yes"
    return "no"


def extract_all_entities(
    text: str,
    known_names: Iterable[str] = (),
) -> list[str]:
    """Every distinct company named in the text, in order of appearance.

    Used for register-style documents - a PTA licensee list names many
    companies in one file, and collapsing that to a single entity would throw
    away most of the answer. Still never infers: each name is one the document
    actually wrote.
    """
    found: list[str] = []
    seen: set[str] = set()

    for name in find_known_names(text or "", known_names):
        if _is_stoplisted(name):
            continue
        if _norm(name) not in seen:
            seen.add(_norm(name))
            found.append(name)

    for match in _COMPANY_RE.finditer(text or ""):
        candidate = _clean(match.group(1))
        if _is_stoplisted(candidate) or _POSSESSIVE.search(candidate):
            continue
        if len(candidate.split()) < 2:
            continue
        if _norm(candidate) not in seen:
            seen.add(_norm(candidate))
            found.append(candidate)

    return found
