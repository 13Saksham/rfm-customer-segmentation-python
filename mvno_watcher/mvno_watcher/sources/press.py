"""Pakistani business press.

Each outlet is its own Source so that one dead feed produces a precise
"SOURCE DOWN: ProPakistani" rather than silently thinning coverage.
"""

from __future__ import annotations

from typing import Optional

from .base import Item, Source, SourceDown, fetch, parse_rss, within_window

# Outlet name -> feed URL. Ordered as in the mandate.
FEEDS: dict[str, str] = {
    "Profit (Pakistan Today)": "https://profit.pakistantoday.com.pk/feed/",
    "Business Recorder": "https://www.brecorder.com/feeds/latest-news",
    "ProPakistani": "https://propakistani.pk/feed/",
    "Dawn Business": "https://www.dawn.com/feeds/business",
    "Mettis Global": "https://mettisglobal.news/feed/",
    "Digital Pakistan": "https://digitalpakistan.pk/feed/",
}

# Telecom-trade outlets that carried MVNO coverage; opt in via press_sources().
EXTRA_FEEDS: dict[str, str] = {
    "PhoneWorld": "https://www.phoneworld.com.pk/feed/",
    "TechJuice": "https://www.techjuice.pk/feed/",
}


class PressFeedSource(Source):
    source_type = "press"
    cadence = "daily"

    def __init__(self, name: str, feed_url: str) -> None:
        self.name = name
        self.feed_url = feed_url
        self.last_warnings: list[str] = []

    def collect(self, since: Optional[str] = None) -> list[Item]:
        self.last_warnings = []
        xml = fetch(self.feed_url)          # raises SourceDown
        items = parse_rss(xml, self.name, self.source_type)
        if not items:
            raise SourceDown(f"{self.name}: feed parsed but contained no items")
        return [i for i in items if within_window(i.published_date, since)]


def press_sources(include_extra: bool = False) -> list[PressFeedSource]:
    feeds = dict(FEEDS)
    if include_extra:
        feeds.update(EXTRA_FEEDS)
    return [PressFeedSource(name, url) for name, url in feeds.items()]
