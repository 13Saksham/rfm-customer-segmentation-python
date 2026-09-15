"""Keyword matching and tier assignment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .config import (
    ALL_KEYWORDS,
    LEADERSHIP_TERMS,
    LICENCE_TERMS,
    MVNO_TERMS,
    REGISTRATION_TERMS,
    WEAK_KEYWORDS,
    WHOLESALE_TERMS,
)
from .entities import detect_enablers, detect_host_operators, extract_entity

_LATIN = re.compile(r"[A-Za-z]")


def _kw_pattern(kw: str) -> re.Pattern:
    if _LATIN.search(kw):
        return re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", re.I)
    return re.compile(re.escape(kw))


_KEYWORD_PATTERNS = [(kw, _kw_pattern(kw)) for kw in ALL_KEYWORDS]


def match_keywords(text: str) -> list[str]:
    text = text or ""
    return [kw for kw, pat in _KEYWORD_PATTERNS if pat.search(text)]


def _has_any(text: str, terms: list[str]) -> bool:
    low = (text or "").lower()
    return any(t.lower() in low for t in terms)


def find_excerpt(text: str, keywords: list[str]) -> str:
    """Return the verbatim sentence that triggered the match.

    Verbatim means verbatim: the sentence is sliced out of the source text
    unmodified. Nothing is paraphrased or reconstructed.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?۔])\s+", text)
    strong = [k for k in keywords if k not in WEAK_KEYWORDS] or keywords
    for sentence in sentences:
        for kw in strong:
            if _kw_pattern(kw).search(sentence):
                return sentence.strip()
    return sentences[0].strip()


@dataclass
class Match:
    tier: str
    keywords: list[str]
    entity: Optional[str]
    excerpt: str
    enablers: list[str]
    reason: str


def classify(
    title: str,
    body: str,
    source_type: str,
    known_names: list[str] | None = None,
    fallback_names: list[str] | None = None,
) -> Optional[Match]:
    """Classify one candidate item. Returns None if it is not a hit at all.

    Tier A - alert immediately:
      A1 MVNO licence application / grant / applicant list, naming a company
      A2 host-operator wholesale agreement, naming a company
      A3 SECP registration or object change naming MVNO / virtual network
    Tier B - weekly digest:
      B1 eSIM or connectivity partnership by a named company
      B2 PSX disclosure mentioning telecom / eSIM / SIM / roaming / connectivity
      B3 telecom or regulatory-affairs leadership hire at a named candidate
    Tier C - log only: keyword-relevant news with no company named.
    """
    known_names = known_names or []
    fallback_names = fallback_names or []
    text = f"{title}\n{body}".strip()

    keywords = match_keywords(text)
    if not keywords:
        return None

    # A match made only of weak keywords ("SIM", "licence") is noise. This is
    # the primary tightening lever if a backfill returns too many hits.
    if all(k in WEAK_KEYWORDS for k in keywords):
        return None

    entity = extract_entity(text, known_names, fallback_names)
    enablers = detect_enablers(text)
    hosts = detect_host_operators(text)
    # Prefer a sentence from the body; fall back to the title only if the
    # body yields nothing, so the excerpt stays a single real sentence.
    excerpt = find_excerpt(body, keywords) or find_excerpt(title, keywords)

    has_mvno = _has_any(text, MVNO_TERMS)
    has_licence = _has_any(text, LICENCE_TERMS)
    has_wholesale = _has_any(text, WHOLESALE_TERMS)
    has_registration = _has_any(text, REGISTRATION_TERMS)
    has_leadership = _has_any(text, LEADERSHIP_TERMS)

    tier, reason = "C", "keyword-relevant, no company named"

    if entity:
        if has_mvno and has_licence:
            tier, reason = "A", "A1 MVNO licence application/grant naming a company"
        elif hosts and has_wholesale:
            tier, reason = "A", "A2 host-operator wholesale agreement"
        elif source_type == "SECP" and has_mvno and has_registration:
            tier, reason = "A", "A3 SECP registration/object change naming MVNO"
        elif source_type == "PSX":
            tier, reason = "B", "B2 PSX disclosure mentioning telecom/eSIM/SIM/roaming"
        elif _has_any(text, ["eSIM", "e-SIM", "connectivity", "roaming", "ای سم", "رومنگ"]) and _has_any(
            text, ["partner", "agreement", "deal", "collaborat", "tie-up", "معاہدہ"]
        ):
            tier, reason = "B", "B1 eSIM/connectivity partnership by a named company"
        elif has_leadership and entity in known_names:
            tier, reason = "B", "B3 telecom/regulatory leadership hire at a known candidate"
        else:
            tier, reason = "C", "company named but no Tier A/B trigger"

    # Never infer an entity name: unnamed items can only ever be Tier C.
    if not entity:
        tier, reason = "C", "no company named in source"

    return Match(
        tier=tier,
        keywords=keywords,
        entity=entity,
        excerpt=excerpt,
        enablers=enablers,
        reason=reason,
    )
