# Pakistan MVNO regulatory watcher

Detects companies entering the Pakistan MVNO market from official and press
sources, and surfaces **a named entity plus a source URL to a human**.

It is not a lead-gen bot. It never contacts anyone, never posts, does no
scoring, drafts no outreach, and writes to no CRM. Slack delivery to the
operator's own webhook is the only outbound path, and it is off by default.

---

## Status: validated, NOT live

The mandatory 12-month backfill has **not** been completed against live
sources, because every source is blocked by the egress policy of the
environment this was built in (see [Egress](#egress-reality)). Live runs are
therefore unscheduled, as the mandate requires.

Two runs were completed:

| Run | Result |
|---|---|
| `backfill` against live sources | **INCONCLUSIVE** — 0/18 endpoints reachable. Reported as `SOURCE DOWN` per source; the tool explicitly refuses to read this as "no signal". |
| `ingest` of research-derived records | 8 hits — **Tier A 2, Tier B 2, Tier C 4**. Zuma Resources present as **Tier A**. |

**No MVNO applicant or licensee has been named publicly** as of 15 Sep 2026.
Applications opened 24 June 2026; PTA has announced no grants, publishes no
applicant list, and no outlet has named an applicant. Zuma Resources is the
only company on public record stating an intention to apply.

The second run is provisional: its records carry
`excerpt_provenance = search_summary`, meaning entity, URL and date are
attributable but the excerpt is not yet a verified verbatim source sentence.
Re-run `backfill` where the sources are reachable before treating any of it
as confirmed.

### What the backfill found

The market premise has changed. Three prior attempts found no private-sector
signal; there is now a public regulatory trail:

- **2026-01-07** — PTA notified the MVNO Policy Framework after Cabinet
  approval; 15-year licences, no spectrum assignment. *(Tier C, no company named)*
- **2026-04-09** — PTA consultation draft, feedback closed 22 April 2026. *(Tier C)*
- **2026-06-28** — PTA **opened MVNO licence applications**; USD 140,000
  upfront, nationwide. *(Tier C)*
- **2026-03-12 / 2026-05-04** — **Zuma Resources Limited** (PSX: ZUMA)
  agreement with **Telna** for multi-IMSI eSIM/SIM connectivity and
  international roaming; states it plans to obtain an MVNO licence from PTA.
  *(Tier A; `enabler_named = Telna`)*
- **2026-03-17** — **Effortel**, an MVNE, announced entry into Pakistan to
  support new MVNO launches. *(Tier A; `enabler_named = Effortel`)*

The two enabler findings are the disqualifier signal the mandate asks to
capture rather than filter: an enabler-supplied platform may remove the
PortaBilling need entirely.

---

## Install and run

```bash
pip install -r requirements.txt

python -m mvno_watcher init
python -m mvno_watcher doctor                  # check every endpoint
python -m mvno_watcher backfill --months 12    # mandatory pre-live validation
python -m mvno_watcher run --sources PTA PSX press
python -m mvno_watcher digest                  # Monday Tier B/C digest
python -m mvno_watcher report --entity Zuma
```

Nothing is delivered to Slack unless you pass `--send-slack` **and** set
`MVNO_SLACK_WEBHOOK_URL`. Without the flag every command prints the alert it
would have sent.

```bash
export MVNO_SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'
python -m mvno_watcher run --send-slack
```

### Cadence

PTA/PSX/press daily, SECP weekly, events monthly. Tier A fires immediately;
Tier B and C go to a Monday digest. **Do not install these until a human has
approved a backfill run made against reachable sources.**

```cron
0  6 * * *  cd /srv/mvno && python -m mvno_watcher run --sources PTA PSX press --send-slack
0  7 * * 1  cd /srv/mvno && python -m mvno_watcher run --sources SECP --send-slack
0  8 1 * *  cd /srv/mvno && python -m mvno_watcher run --sources event --send-slack
30 8 * * 1  cd /srv/mvno && python -m mvno_watcher digest --send-slack
```

---

## Match logic

**Tier A — alert immediately**

| Rule | Trigger |
|---|---|
| A1 | MVNO term + licence/application/grant term + a named company |
| A2 | Host operator (Jazz/Ufone/Zong/Telenor/PTML/SCOM) + an agreement + a named company |
| A3 | SECP registration or object change naming MVNO / virtual network operator |

**A2 is the earliest signal, not A1.** PTA requires a prospective MVNO to
*"first sign an agreement with at least one Mobile Network Operator (MNO)
before applying for the permission from PTA"*
([Revised Framework for MVNO Services](https://www.pta.gov.pk/en/media-center/single-media/revised-framework-for-mvno-services-in-pakistan)).
The host-operator deal therefore happens **before** any PTA filing exists, so
a Jazz/Ufone/Zong/Telenor agreement is the first public trace of an entrant.

Because of that, a host-operator deal is its own **admission route**,
independent of the keyword list: *"Acme Digital Limited signed an agreement
with Jazz to launch mobile services"* contains no listed keyword, yet is
precisely the target signal. It requires a named host operator **and** deal
language **and** mobile context, so a network-equipment deal between an
operator and a vendor does not qualify.

**Tier B — weekly digest**: eSIM or connectivity partnership by a named
company; any PSX disclosure mentioning telecom/eSIM/SIM/roaming/connectivity;
telecom or regulatory-affairs leadership hire at a known candidate.

**Tier C — log only**: keyword-relevant news with **no company named**.

A1 deliberately fires on a company publicly stating it will apply, not only
on a granted licence — the mandate's purpose is to catch entrants *before*
they are publicly known, and Tier A volume is inherently tiny. Tighten by
narrowing `LICENCE_TERMS` in `config.py` if that proves too broad.

Keywords, including Urdu equivalents, live in `config.py`. `WEAK_KEYWORDS`
(`SIM`, `IMSI`, `لائسنس`, `پی ٹی اے`) cannot produce a hit on their own —
this is the first tightening lever if a backfill exceeds 50 hits.

---

## Hard gates

Enforced in `models.Hit.validate()`; a failure means **discarded, not logged**
(the discard itself is recorded in the `discards` table so a systematic
failure is visible):

- **No resolvable `source_url` → discarded.** No exceptions. Search-engine and
  `?s=`/`/search` URLs are rejected: a search URL is not evidence.
- **An entity name is never inferred.** No company named → Tier C, always. A
  Tier A/B hit without an entity cannot be stored.
- **Verbatim excerpt required**, sliced unmodified from the source text.
- **Deduplicated** by `entity_name + published_date` across all sources, so
  four outlets covering one announcement collapse to one hit. Every
  corroborating URL is kept in `hit_sources` — dedupe loses no evidence.
- **Fail loudly.** An unreachable source raises `SOURCE DOWN: <name>`; a
  partially-read source reports reduced coverage. A backfill with any source
  down returns **INCONCLUSIVE**, never "0 hits = no signal".

### Wrong-company protection

A prior enrichment run attached correct facts to the wrong company record, so
attribution is tested directly. Two bugs were caught and fixed during the
build, both regression-tested:

1. `Telecom`/`Technologies`/`Holdings` were treated as legal suffixes, so the
   headline *"…One of Asia's Largest New Telecom Opportunities"* produced the
   entity **"Asia's Largest New Telecom"**. Designators are now legal suffixes
   only, and possessive fragments are rejected.
2. Recognising enabler names promoted the counterparty over the subject, so
   *"Telna shall provide Zuma Resources Limited…"* filed under **Telna**.
   Enablers and host operators are now a last-resort fallback only; the
   enabler is still captured in `enabler_named`.

---

## Output fields

Every hit stores all mandated fields: `entity_name`, `source_url`,
`source_type` (PTA|SECP|PSX|press|event), `published_date`, `tier`,
`matched_keywords`, `verbatim_excerpt`, `already_on_RFI_list`,
`enabler_named`. Plus `source_name`, `title`, `excerpt_provenance`,
`first_seen_at`, `alerted_at`, `digested_at`.

The Slack payload carries only `entity_name`, tier, a one-line excerpt and
`source_url`. Nothing else.

State is SQLite (`data/watcher.sqlite3`), append-only. Re-runs never re-alert:
`alerted_at`/`digested_at` gate delivery.

---

## Egress reality

The transport layer (`transport.py`) separates *how bytes are obtained* from
*what is done with them*, because `pta.gov.pk` and `secp.gov.pk` are slow,
periodically drop TLS, and geo-block foreign IPs. Two transports ship, both
local: `direct` (requests) and `fixture` (offline replay of saved responses).

```bash
python -m mvno_watcher run --save-fixtures tests/fixtures   # record
python -m mvno_watcher --transports fixture run             # replay, no network
```

**On blocked hosts.** Where outbound access is governed by an allowlist, a
denied host answers 403/407 at the proxy. That is an organisation policy
decision: this tool reports it, never retries it, and does not route around
it. In the environment this was built in, all 18 source endpoints — plus every
archive, mirror and reader service tested — are denied, so no alternative
route exists there. The fix is to run the watcher where the sources are
permitted, or to have the hosts allowlisted.

A deployment that legitimately needs another route (a public web archive for
historical backfill, a rendering reader for JavaScript listings, a commercial
fetch API for hosts refusing datacentre IPs) can register one with
`transport.register_transport()` in its own environment, where that access is
theirs to authorise. None is bundled.

### Meanwhile: the ingest path

`ingest` feeds externally-collected records through the **same** matcher and
the same hard gates, differing only in the `excerpt_provenance` label. This is
how the provisional backfill above was produced.

```bash
python -m mvno_watcher ingest data/backfill_seed.json --provenance search_summary
```

---

## Known limitations

1. **SECP incorporations are not fully covered.** SECP publishes no open feed
   of daily incorporations; the registry sits behind the eZfile portal and
   bulk extracts are a paid service. The source covers SECP press releases and
   notices, and exposes `SECPSource.registry_export_path` for an authenticated
   eZfile export. This matters: an object change is how Zuma appeared
   (formerly Bilal Fibres Ltd), so this is the weakest link in coverage.
2. **`config/rfi_list.txt` is empty.** The 26 RFI companies are not invented:
   until the file is populated every hit records
   `already_on_RFI_list = "unknown"` and each run prints a warning. One name
   per line enables the cross-check.
3. **Scanned PDFs are not read.** PDF text is extracted with `pypdf`, so
   text-based PTA registers and PSX disclosures are parsed. An image-only
   scan yields no text; that is reported as a source warning
   (`PDF unreadable`), never silently skipped. There is no OCR.
4. **Provisional excerpts.** Records ingested with
   `excerpt_provenance = search_summary` are not verified verbatim sentences.

## The licensee register

The document that actually answers *"who holds an MVNO licence"* is a PTA
register, published as a PDF under `/assets/media/`. PTA publishes these for
every licence class — FLL, CVAS, LDI — under inconsistent filenames:

```
2025-01-03-List-of-CVAS-Licensees-02012025.pdf
2025-04-16-Updated-FLL-Licensees-List-for-Pakistan-As-on-14Apr25.pdf
fll_list_pak_09-02-2023.pdf
cvas_list_05112021.pdf
ldi_lic_list_14062022.pdf
sr7_ldi_lic_pak_22-02-2024.pdf
ldi-lic-ajkgb-190717.pdf
```

**No MVNO register exists yet** — its absence alongside these is itself
evidence that no MVNO licence has been granted. The watcher treats registers
specially:

- any link matching `List-of-…-Licensee` is recognised as a register;
- it is **always fetched and is exempt from the date window** — a register is
  current state, not news;
- its PDF text is extracted and **exploded row by row**, so a register naming
  eight licensees produces eight named Tier A hits rather than one.

Three bugs were caught here and are regression-tested: `/assets/media/` was
excluded by the crawler's link filter, which silently dropped every PTA PDF
including registers and the policy framework; register rows collapsed to a
single entity, discarding most of the answer; and the filename pattern
recognised only two of the seven real register shapes above.

## Tests

```bash
python -m unittest discover -s tests   # 57 tests
```

Covers the hard gates, tier assignment, Urdu keyword matching, entity
false-positives and attribution, dedupe, re-alert suppression, fail-loudly
behaviour, and a full offline scrape→parse→match→gate→persist run against
fixtures shaped like the live pages, including PDF extraction and
row-by-row parsing of a licensee register.

## Out of scope

LinkedIn (any automated access), contact or email collection, outreach
drafting, scoring, CRM writes.
