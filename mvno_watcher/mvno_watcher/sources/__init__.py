"""Source registry."""

from __future__ import annotations

from .base import Item, Source, SourceDown
from .events import EventsSource
from .press import PressFeedSource, press_sources
from .psx import PSXSource
from .pta import PTASource
from .secp import SECPSource

#: Cadence groups from the mandate.
CADENCE = {
    "daily": ("PTA", "PSX", "press"),
    "weekly": ("SECP",),
    "monthly": ("event",),
}


def all_sources(include_extra_press: bool = False) -> list[Source]:
    """Every source, in the mandate's priority order."""
    return [
        PTASource(),
        SECPSource(),
        PSXSource(),
        *press_sources(include_extra=include_extra_press),
        EventsSource(),
    ]


def sources_for(names: list[str] | None = None, include_extra_press: bool = False) -> list[Source]:
    """Select sources by source_type (PTA/SECP/PSX/press/event) or by name."""
    every = all_sources(include_extra_press=include_extra_press)
    if not names:
        return every
    wanted = {n.lower() for n in names}
    return [
        s for s in every
        if s.source_type.lower() in wanted or s.name.lower() in wanted
    ]


__all__ = [
    "Item", "Source", "SourceDown", "PTASource", "SECPSource", "PSXSource",
    "PressFeedSource", "press_sources", "EventsSource",
    "all_sources", "sources_for", "CADENCE",
]
