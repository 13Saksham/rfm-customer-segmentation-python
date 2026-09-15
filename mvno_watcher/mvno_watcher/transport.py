"""Fetch transports.

Why a seam at all: a single requests.get() is a weak basis for a daily
watcher of Pakistani regulator sites. pta.gov.pk and secp.gov.pk are slow and
periodically drop TLS, and dps.psx.com.pk serves disclosures as PDFs behind an
unstable listing. Separating "how bytes are obtained" from "what we do with
them" lets the fetch strategy change per deployment without touching the
matcher, and lets the whole pipeline be exercised offline.

Two transports ship here, both local in nature:

  direct    requests straight to the origin. The normal path.
  fixture   replay saved responses from disk. No network. Used by the tests
            and to reproduce a past run from the exact bytes it saw.

Order comes from MVNO_TRANSPORTS (default "direct").

EGRESS POLICY - READ THIS BEFORE ADDING A TRANSPORT
Where outbound access is governed by an allowlist, a denied host answers
403/407 at the proxy. That is an organisation policy decision. This module
treats it as terminal: it is reported, never retried and never routed around.
If the sources are blocked where you are running, the fix is to run the
watcher somewhere they are permitted, or to have the hosts allowlisted - not
to find another way out of the network.

Deployments that legitimately need a different route (a public web archive
for historical backfill, a rendering reader for JavaScript listings, or a
commercial fetch API where a host refuses datacentre IPs) can register one
with register_transport() in their own environment, where that access is
theirs to authorise. None is bundled here.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

USER_AGENT = (
    "Mozilla/5.0 (compatible; pakistan-mvno-watcher/1.0; "
    "regulatory monitoring; +https://pta.gov.pk)"
)

#: Proxy/policy denials. Terminal by design.
POLICY_STATUSES = {403, 407}


class TransportError(Exception):
    """One transport failed. Not fatal alone: the chain may continue."""

    def __init__(self, transport: str, detail: str, policy_denied: bool = False):
        super().__init__(f"{transport}: {detail}")
        self.transport = transport
        self.detail = detail
        self.policy_denied = policy_denied


class Transport:
    name = "base"

    def get(self, url: str, timeout: int = 30) -> str:
        raise NotImplementedError


class DirectTransport(Transport):
    name = "direct"

    def __init__(self, retries: int = 3) -> None:
        self.retries = retries

    def get(self, url: str, timeout: int = 30) -> str:
        if requests is None:  # pragma: no cover
            raise TransportError(self.name, "the 'requests' package is not installed")
        last: Optional[TransportError] = None
        for attempt in range(self.retries):
            try:
                resp = requests.get(
                    url, timeout=timeout, headers={"User-Agent": USER_AGENT}
                )
            except Exception as exc:
                last = TransportError(self.name, f"{type(exc).__name__}: {exc}")
            else:
                if resp.status_code in POLICY_STATUSES:
                    # An organisation policy denial. Do not retry, do not
                    # route around: surface it and stop.
                    raise TransportError(
                        self.name,
                        f"HTTP {resp.status_code} - egress policy denial for "
                        f"{urlparse(url).netloc}; host is not permitted from "
                        f"this environment",
                        policy_denied=True,
                    )
                if resp.status_code >= 400:
                    last = TransportError(self.name, f"HTTP {resp.status_code}")
                else:
                    resp.encoding = resp.encoding or "utf-8"
                    return resp.text
            if attempt < self.retries - 1:
                time.sleep(2 ** (attempt + 1))
        raise last or TransportError(self.name, "unknown failure")


class FixtureTransport(Transport):
    """Replay saved responses from disk. Purely local; touches no network."""

    name = "fixture"

    def __init__(self, directory: Optional[str] = None) -> None:
        self.directory = Path(
            directory or os.environ.get("MVNO_FIXTURE_DIR", "tests/fixtures")
        )

    @staticmethod
    def key_for(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20] + ".html"

    def path_for(self, url: str) -> Path:
        return self.directory / self.key_for(url)

    def save(self, url: str, body: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(url)
        path.write_text(body, encoding="utf-8")
        with (self.directory / "index.txt").open("a", encoding="utf-8") as fh:
            fh.write(f"{self.key_for(url)}\t{url}\n")
        return path

    def get(self, url: str, timeout: int = 30) -> str:
        path = self.path_for(url)
        if not path.exists():
            raise TransportError(self.name, f"no fixture saved for {url}")
        return path.read_text(encoding="utf-8")


BACKENDS: dict[str, Callable[[], Transport]] = {
    "direct": DirectTransport,
    "fixture": FixtureTransport,
}

DEFAULT_CHAIN = "direct"


def register_transport(name: str, factory: Callable[[], Transport]) -> None:
    """Register a deployment-specific transport.

    The extension point exists so that a route which is legitimate in your
    environment - and authorised by whoever runs it - can be added without
    editing this package. Nothing is registered by default.
    """
    BACKENDS[name.strip().lower()] = factory


class Chain:
    """Try each transport in order; report every route that failed."""

    def __init__(self, transports: list[Transport]) -> None:
        self.transports = transports
        self.last_route: Optional[str] = None

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.transports]

    def get(self, url: str, timeout: int = 30) -> str:
        errors: list[str] = []
        for transport in self.transports:
            try:
                body = transport.get(url, timeout=timeout)
            except TransportError as exc:
                errors.append(str(exc))
                continue
            except Exception as exc:
                errors.append(f"{transport.name}: {type(exc).__name__}: {exc}")
                continue
            if body and body.strip():
                self.last_route = transport.name
                return body
            errors.append(f"{transport.name}: empty response")
        raise TransportError(
            "chain",
            f"{url} unreachable via [{', '.join(self.names)}] :: "
            f"{' | '.join(errors)}",
        )


def build_chain(spec: Optional[str] = None) -> Chain:
    spec = spec or os.environ.get("MVNO_TRANSPORTS", DEFAULT_CHAIN)
    transports: list[Transport] = []
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in BACKENDS:
            raise ValueError(
                f"unknown transport {name!r}; registered: {sorted(BACKENDS)}"
            )
        transports.append(BACKENDS[name]())
    return Chain(transports or [DirectTransport()])


_CHAIN: Optional[Chain] = None


def get_chain() -> Chain:
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = build_chain()
    return _CHAIN


def set_chain(chain: Optional[Chain]) -> None:
    """Override the process-wide chain (tests, CLI --transports)."""
    global _CHAIN
    _CHAIN = chain
