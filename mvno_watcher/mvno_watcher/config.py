"""Static configuration: keywords, source registry, known-entity lists."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_DB_PATH = os.environ.get(
    "MVNO_WATCHER_DB", str(REPO_ROOT / "data" / "watcher.sqlite3")
)

# --- Keywords -------------------------------------------------------------
# Every keyword is matched case-insensitively on word boundaries where the
# script is Latin; Urdu terms are matched as plain substrings because Urdu
# has no equivalent word-boundary semantics in the `re` module.

CORE_KEYWORDS = [
    "MVNO",
    "mobile virtual network operator",
    "virtual network operator",
    "virtual network",
    "MVNE",
    "mobile virtual network enabler",
    "PTA licence",
    "PTA license",
    "telecom licence",
    "telecom license",
    "host operator",
    "host mobile network",
    "wholesale agreement",
    "wholesale arrangement",
    "eSIM",
    "e-SIM",
    "SIM",
    "roaming agreement",
    "international roaming",
    "multi-IMSI",
    "multi IMSI",
    "IMSI",
    "connectivity agreement",
    "connectivity partnership",
]

# Urdu equivalents, for sources that publish in Urdu.
URDU_KEYWORDS = [
    "موبائل ورچوئل نیٹ ورک آپریٹر",  # mobile virtual network operator
    "ورچوئل نیٹ ورک",                  # virtual network
    "ای سم",                            # eSIM
    "ٹیلی کام لائسنس",                 # telecom licence
    "لائسنس",                           # licence
    "رومنگ",                            # roaming
    "تھوک معاہدہ",                      # wholesale agreement
    "پی ٹی اے",                         # PTA
]

ALL_KEYWORDS = CORE_KEYWORDS + URDU_KEYWORDS

# Keywords that, on their own, are too generic to justify a Tier A/B match.
# "SIM" matches "SIM card price" all day long; it only counts alongside a
# stronger term or an explicit telecom context.
WEAK_KEYWORDS = {"SIM", "IMSI", "لائسنس", "پی ٹی اے"}

# --- Tier A trigger vocabulary -------------------------------------------

MVNO_TERMS = [
    "MVNO",
    "mobile virtual network operator",
    "virtual network operator",
    "موبائل ورچوئل نیٹ ورک آپریٹر",
]

LICENCE_TERMS = [
    "licence",
    "license",
    "licensing",
    "licensee",
    "applicant",
    "application",
    "apply",
    "grant",
    "granted",
    "award",
    "bid",
    "tender",
    "لائسنس",
]

WHOLESALE_TERMS = [
    "wholesale",
    "host operator",
    "host mobile network",
    "network capacity",
    "commercial agreement",
    "capacity agreement",
    "تھوک معاہدہ",
]

REGISTRATION_TERMS = [
    "incorporat",       # incorporated / incorporation
    "registered",
    "registration",
    "change of name",
    "name change",
    "change in objects",
    "object clause",
    "memorandum of association",
]

LEADERSHIP_TERMS = [
    "appointed",
    "appointment",
    "joins as",
    "named as",
    "head of regulatory",
    "regulatory affairs",
    "chief technology officer",
    "chief executive",
    "director telecom",
]

# --- Host operators (Tier A: wholesale agreement counterparties) ----------

HOST_OPERATORS = [
    "Jazz",
    "Mobilink",
    "Ufone",
    "Pak Telecom Mobile",
    "PTML",
    "Zong",
    "China Mobile Pakistan",
    "CMPak",
    "Telenor Pakistan",
    "SCOM",
    "Special Communications Organization",
]

# --- Enablers / MVNEs (DISQUALIFIER: capture, never filter) ---------------

KNOWN_ENABLERS = [
    "Telna",
    "Effortel",
    "Plintron",
    "Transatel",
    "Tata Communications MOVE",
    "Gigs",
    "floLIVE",
    "flolive",
    "Mobilise",
    "Cellact",
    "Comviva",
    "Xius",
    "MATRIXX",
    "Optiva",
    "Amdocs",
    "Tecnotree",
    "Nexign",
    "PortaOne",
    "Airalo",
    "Truphone",
    "1GLOBAL",
    "Sim Local",
    "Monty Mobile",
    "Telefonica Kite",
    "Ericsson",
    "Huawei",
    "ZTE",
]

# --- Entity-extraction stoplist ------------------------------------------
# Institutions, regulators and generic bodies that carry corporate-looking
# designators but are never the entity we are hunting.

ENTITY_STOPLIST = {
    "pakistan telecommunication authority",
    "pta",
    "securities and exchange commission of pakistan",
    "securities & exchange commission of pakistan",
    "secp",
    "pakistan stock exchange",
    "pakistan stock exchange limited",
    "psx",
    "federal cabinet",
    "ministry of information technology",
    "ministry of it and telecommunication",
    "universal service fund",
    "universal service fund company",
    "research and development fund",
    "state bank of pakistan",
    "federal board of revenue",
    "government of pakistan",
    "national assembly",
    "senate of pakistan",
    "supreme court of pakistan",
    "gsm association",
    "gsma",
    "itcn asia",
    "ecommerce gateway pakistan",
}

# Corporate designators used to recognise a company name in free text.
CORPORATE_DESIGNATORS = [
    "(SMC-PRIVATE) LIMITED",
    "(SMC-Private) Limited",
    "(Private) Limited",
    "(Pvt) Limited",
    "(Pvt.) Ltd",
    "(Pvt) Ltd",
    "Private Limited",
    "Limited",
    "Ltd.",
    "Ltd",
    "Incorporated",
    "Inc.",
    "Inc",
    "Corporation",
    "Corp.",
    "PLC",
    "LLC",
    "LLP",
    "GmbH",
    "N.V.",
]

SOURCE_TYPES = ("PTA", "SECP", "PSX", "press", "event")

TIERS = ("A", "B", "C")
