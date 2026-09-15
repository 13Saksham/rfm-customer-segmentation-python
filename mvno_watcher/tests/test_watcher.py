"""Tests for the hard gates, tier logic, dedupe and fail-loudly behaviour."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mvno_watcher import db, pipeline
from mvno_watcher.entities import detect_enablers, extract_entity, on_rfi_list
from mvno_watcher.matcher import classify, match_keywords
from mvno_watcher.models import DiscardedHit, Hit
from mvno_watcher.sources import Item, Source, SourceDown


def make_hit(**kw) -> Hit:
    base = dict(
        entity_name="Zuma Resources Limited",
        source_url="https://dps.psx.com.pk/download/document/276579.pdf",
        source_type="PSX",
        published_date="2026-05-04",
        tier="A",
        matched_keywords=["MVNO"],
        verbatim_excerpt="Zuma plans to obtain an MVNO licence from the PTA.",
        already_on_rfi_list="unknown",
        enabler_named="Telna",
    )
    base.update(kw)
    return Hit(**base)


class TestHardGates(unittest.TestCase):
    def test_valid_hit_passes(self):
        hit = make_hit()
        self.assertIs(hit.validate(), hit)

    def test_missing_url_is_discarded(self):
        with self.assertRaises(DiscardedHit):
            make_hit(source_url="").validate()

    def test_unresolvable_url_is_discarded(self):
        with self.assertRaises(DiscardedHit):
            make_hit(source_url="not-a-url").validate()

    def test_search_page_url_is_discarded(self):
        for url in (
            "https://www.google.com/search?q=mvno+pakistan",
            "https://propakistani.pk/?s=mvno",
            "https://www.pta.gov.pk/search/mvno",
        ):
            with self.subTest(url=url), self.assertRaises(DiscardedHit):
                make_hit(source_url=url).validate()

    def test_missing_excerpt_is_discarded(self):
        with self.assertRaises(DiscardedHit):
            make_hit(verbatim_excerpt="   ").validate()

    def test_missing_date_is_discarded(self):
        with self.assertRaises(DiscardedHit):
            make_hit(published_date="").validate()

    def test_bad_date_is_discarded(self):
        with self.assertRaises(DiscardedHit):
            make_hit(published_date="04/05/2026").validate()

    def test_tier_a_without_entity_is_discarded(self):
        """Never infer an entity name: unnamed cannot be Tier A."""
        with self.assertRaises(DiscardedHit):
            make_hit(entity_name=None, tier="A").validate()

    def test_tier_c_without_entity_is_allowed(self):
        make_hit(entity_name=None, tier="C").validate()


class TestEntities(unittest.TestCase):
    def test_designator_name_extracted(self):
        self.assertEqual(
            extract_entity("The board of Zuma Resources Limited approved the deal."),
            "Zuma Resources Limited",
        )

    def test_regulator_is_not_an_entity(self):
        self.assertIsNone(
            extract_entity("The Pakistan Telecommunication Authority issued a framework.")
        )

    def test_no_company_returns_none(self):
        self.assertIsNone(extract_entity("Applications are invited for MVNO licences."))

    def test_known_name_wins(self):
        self.assertEqual(
            extract_entity("Effortel enters Pakistan.", known_names=["Effortel"]),
            "Effortel",
        )

    def test_enablers_are_captured(self):
        found = detect_enablers("Zuma signed with Telna; Effortel also entered.")
        self.assertIn("Telna", found)
        self.assertIn("Effortel", found)

    def test_rfi_unknown_when_list_empty(self):
        self.assertEqual(on_rfi_list("Zuma Resources Limited", []), "unknown")

    def test_rfi_match_is_substring_tolerant(self):
        self.assertEqual(
            on_rfi_list("Zuma Resources Limited", ["Zuma Resources"]), "yes"
        )
        self.assertEqual(on_rfi_list("Acme Limited", ["Zuma Resources"]), "no")


class TestMatcher(unittest.TestCase):
    def test_urdu_keywords_match(self):
        kws = match_keywords("پی ٹی اے نے موبائل ورچوئل نیٹ ورک آپریٹر لائسنس جاری کیا")
        self.assertIn("موبائل ورچوئل نیٹ ورک آپریٹر", kws)

    def test_tier_a_mvno_licence_named_company(self):
        m = classify(
            "Zuma to seek MVNO licence",
            "Zuma Resources Limited is planning to obtain a Mobile Virtual "
            "Network Operator (MVNO) license from the PTA.",
            "PSX",
        )
        self.assertEqual(m.tier, "A")
        self.assertEqual(m.entity, "Zuma Resources Limited")

    def test_tier_a_host_operator_wholesale(self):
        m = classify(
            "Wholesale deal signed",
            "Acme Digital Limited signed a wholesale agreement with Jazz to "
            "host its virtual network.",
            "press",
        )
        self.assertEqual(m.tier, "A")

    def test_tier_a_secp_object_change(self):
        m = classify(
            "Object change filed",
            "Bilal Fibres Limited registered a change in objects to include "
            "mobile virtual network operator services.",
            "SECP",
        )
        self.assertEqual(m.tier, "A")

    def test_tier_b_psx_disclosure(self):
        m = classify(
            "Telecom disclosure",
            "Orion Holdings Limited disclosed an eSIM and roaming arrangement.",
            "PSX",
        )
        self.assertEqual(m.tier, "B")

    def test_tier_c_when_no_company_named(self):
        m = classify(
            "PTA opens MVNO licensing",
            "The Pakistan Telecommunication Authority invited applications for "
            "the grant of MVNO licences at a fee of $140,000.",
            "PTA",
        )
        self.assertEqual(m.tier, "C")
        self.assertIsNone(m.entity)

    def test_weak_only_match_is_not_a_hit(self):
        self.assertIsNone(classify("SIM prices rise", "The price of a SIM went up.", "press"))

    def test_irrelevant_item_is_not_a_hit(self):
        self.assertIsNone(classify("Cement output", "Cement despatches rose.", "press"))

    def test_excerpt_is_verbatim_from_source(self):
        body = ("Unrelated opening sentence. Zuma Resources Limited will apply "
                "for an MVNO licence. Trailing sentence.")
        m = classify("t", body, "press")
        self.assertIn(m.excerpt, body)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")

    def test_dedupe_same_entity_and_date_across_outlets(self):
        a = make_hit(source_url="https://propakistani.pk/a", source_type="press")
        b = make_hit(source_url="https://www.brecorder.com/b", source_type="press")
        self.assertEqual(db.upsert_hit(self.conn, a), "new")
        self.assertEqual(db.upsert_hit(self.conn, b), "duplicate")
        self.assertEqual(db.counts_by_tier(self.conn), {"A": 1})
        rows = self.conn.execute("SELECT * FROM hit_sources").fetchall()
        self.assertEqual(len(rows), 2, "both outlet URLs must be retained")

    def test_rerun_does_not_realert(self):
        db.upsert_hit(self.conn, make_hit())
        self.assertEqual(pipeline.fire_tier_a(self.conn), 1)
        self.assertEqual(pipeline.fire_tier_a(self.conn), 0)

    def test_history_is_never_deleted_on_reupsert(self):
        db.upsert_hit(self.conn, make_hit())
        db.upsert_hit(self.conn, make_hit())
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) c FROM hits").fetchone()["c"], 1
        )


class _DeadSource(Source):
    name = "DeadFeed"
    source_type = "press"

    def collect(self, since=None):
        raise SourceDown("connection refused")


class _LiveSource(Source):
    name = "LiveFeed"
    source_type = "press"

    def collect(self, since=None):
        return [Item(
            title="Zuma signs telecom deal",
            url="https://propakistani.pk/2026/05/04/zuma-telna/",
            body="Zuma Resources Limited will apply for an MVNO licence from the PTA.",
            source_name="ProPakistani", source_type="press",
            published_date="2026-05-04",
        )]


class TestFailLoudly(unittest.TestCase):
    def test_source_down_is_reported_not_swallowed(self):
        conn = db.connect(":memory:")
        report = pipeline.run(conn, [_DeadSource(), _LiveSource()])
        self.assertTrue(report.any_source_down)
        self.assertEqual(report.sources_down[0][0], "DeadFeed")
        self.assertIn("LiveFeed", report.sources_ok)
        # The live source still produced its hit; coverage loss is explicit.
        self.assertEqual(report.new_hits, 1)

    def test_all_sources_down_does_not_look_like_zero_signal(self):
        conn = db.connect(":memory:")
        report = pipeline.run(conn, [_DeadSource()])
        self.assertTrue(report.any_source_down)
        self.assertEqual(report.new_hits, 0)
        self.assertIn("SOURCE DOWN", report.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestEntityFalsePositives(unittest.TestCase):
    """Regression: a headline fragment must never become a company record.

    'Effortel Enters Pakistan as MVNO Regulations Open One of Asia's Largest
    New Telecom Opportunities' previously yielded the entity "Asia's Largest
    New Telecom" - exactly the wrong-company-record failure the hard gates
    exist to prevent.
    """

    HEADLINE = ("Effortel Enters Pakistan as MVNO Regulations Open One of "
                "Asia's Largest New Telecom Opportunities")

    def test_headline_fragment_is_not_an_entity(self):
        self.assertIsNone(extract_entity(self.HEADLINE))

    def test_named_enabler_is_recognised_from_known_list(self):
        self.assertEqual(
            extract_entity(self.HEADLINE, known_names=["Effortel"]), "Effortel"
        )

    def test_common_nouns_are_not_designators(self):
        for text in (
            "One of Asia's Largest New Telecom Opportunities",
            "Pakistan's Leading Digital Technologies",
            "The Future of Mobile Communications",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_entity(text))

    def test_real_legal_suffixes_still_extract(self):
        for text, expected in (
            ("Zuma Resources Limited approved the deal.", "Zuma Resources Limited"),
            ("Telna North America, Inc. will provide access.", "Telna North America, Inc"),
            ("Bilal Fibres Ltd changed its objects.", "Bilal Fibres Ltd"),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_entity(text), expected)


class TestEntityAttribution(unittest.TestCase):
    """An enabler is a counterparty, not the subject of the story."""

    STORY = ("Under the agreement Telna shall provide Zuma Resources Limited "
             "access to its multi-IMSI global connectivity infrastructure.")

    def test_subject_beats_enabler(self):
        self.assertEqual(
            extract_entity(self.STORY, fallback_names=["Telna"]),
            "Zuma Resources Limited",
        )

    def test_enabler_used_only_when_nothing_else_named(self):
        self.assertEqual(
            extract_entity("Effortel entered the Pakistan market.",
                           fallback_names=["Effortel"]),
            "Effortel",
        )

    def test_enabler_still_captured_separately(self):
        self.assertEqual(detect_enablers(self.STORY), ["Telna"])

    def test_host_operator_does_not_steal_subject(self):
        self.assertEqual(
            extract_entity(
                "Acme Digital Limited signed a wholesale agreement with Jazz.",
                fallback_names=["Jazz"],
            ),
            "Acme Digital Limited",
        )


class TestEndToEndOffline(unittest.TestCase):
    """Full scrape -> parse -> match -> gate -> persist path, no network.

    Exercises the real HTML/RSS parsers against fixtures shaped like the live
    pages, so a layout-parsing regression fails here rather than in a silent
    zero-hit production run.
    """

    FIXTURES = Path(__file__).resolve().parent / "fixtures"

    @classmethod
    def setUpClass(cls):
        if not cls.FIXTURES.exists():
            raise unittest.SkipTest("fixtures not generated")

    def setUp(self):
        from mvno_watcher.transport import Chain, FixtureTransport, set_chain
        set_chain(Chain([FixtureTransport(str(self.FIXTURES))]))
        self.conn = db.connect(":memory:")

    def tearDown(self):
        from mvno_watcher.transport import set_chain
        set_chain(None)

    def test_pta_grant_becomes_tier_a_with_correct_entity(self):
        from mvno_watcher.sources.pta import PTASource
        src = PTASource(index_urls=[
            "https://www.pta.gov.pk/en/media-center/press-releases"
        ])
        pipeline.run(self.conn, [src], fire_alerts=False)
        # Scoped to the grant article: the same index also carries the
        # licensee register, which legitimately produces its own hits.
        rows = self.conn.execute(
            "SELECT * FROM hits WHERE tier='A' AND source_url LIKE ?",
            ("%pta-grants-mvno-licence-to-acme-digital%",),
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entity_name"], "Acme Digital (Pvt) Limited")
        self.assertEqual(rows[0]["source_type"], "PTA")
        self.assertEqual(rows[0]["published_date"], "2026-08-12")
        # The excerpt must be a genuine sentence, not headline + body fused.
        self.assertTrue(rows[0]["verbatim_excerpt"].endswith("."))
        self.assertIn("granted a Mobile Virtual Network Operator",
                      rows[0]["verbatim_excerpt"])

    def test_unnamed_pta_notice_is_tier_c(self):
        from mvno_watcher.sources.pta import PTASource
        src = PTASource(index_urls=[
            "https://www.pta.gov.pk/en/media-center/press-releases"
        ])
        pipeline.run(self.conn, [src], fire_alerts=False)
        rows = self.conn.execute(
            "SELECT * FROM hits WHERE tier='C'"
        ).fetchall()
        self.assertTrue(rows)
        self.assertTrue(all(r["entity_name"] is None for r in rows))

    def test_rss_feed_yields_zuma_with_telna_as_enabler(self):
        from mvno_watcher.sources.press import PressFeedSource
        src = PressFeedSource("ProPakistani", "https://propakistani.pk/feed/")
        pipeline.run(self.conn, [src], fire_alerts=False)
        rows = db.find_entity(self.conn, "Zuma")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tier"], "A")
        self.assertEqual(rows[0]["enabler_named"], "Telna")
        self.assertEqual(rows[0]["published_date"], "2026-05-04")

    def test_irrelevant_feed_item_is_not_stored(self):
        from mvno_watcher.sources.press import PressFeedSource
        src = PressFeedSource("ProPakistani", "https://propakistani.pk/feed/")
        pipeline.run(self.conn, [src], fire_alerts=False)
        self.assertEqual(db.find_entity(self.conn, "Cement"), [])

    def test_partial_index_failure_is_reported(self):
        from mvno_watcher.sources.pta import PTASource
        src = PTASource(index_urls=[
            "https://www.pta.gov.pk/en/media-center/press-releases",
            "https://www.pta.gov.pk/en/media-center/does-not-exist",
        ])
        pipeline.run(self.conn, [src], fire_alerts=False)
        self.assertTrue(src.last_warnings,
                        "a partially-read source must report reduced coverage")


class TestLicenseeRegister(unittest.TestCase):
    """A PTA licensee register must yield one named hit per licensee.

    This is the document that actually answers "who has a licence", and PTA
    publishes it as a PDF under /assets/media/. Collapsing it to a single
    entity, or excluding that path from the crawl, would discard the answer -
    both were real bugs caught here.
    """

    FIXTURES = Path(__file__).resolve().parent / "fixtures"
    REGISTER_URL = ("https://www.pta.gov.pk/assets/media/"
                    "2026-09-01-List-of-MVNO-Licensees-01092026.pdf")

    @classmethod
    def setUpClass(cls):
        if not (cls.FIXTURES / "index.txt").exists():
            raise unittest.SkipTest("fixtures not generated")

    def setUp(self):
        from mvno_watcher.transport import Chain, FixtureTransport, set_chain
        set_chain(Chain([FixtureTransport(str(self.FIXTURES))]))
        self.conn = db.connect(":memory:")

    def tearDown(self):
        from mvno_watcher.transport import set_chain
        set_chain(None)

    def _run(self):
        from mvno_watcher.sources.pta import PTASource
        src = PTASource(index_urls=[
            "https://www.pta.gov.pk/en/media-center/press-releases"
        ])
        pipeline.run(self.conn, [src], fire_alerts=False)
        return src

    def test_pdf_text_is_extracted(self):
        from mvno_watcher.sources.base import fetch_pdf_text
        text = fetch_pdf_text(self.REGISTER_URL)
        self.assertIn("Acme Digital (Pvt) Limited", text)
        self.assertIn("List of MVNO Licensees", text)

    def test_every_licensee_becomes_its_own_hit(self):
        self._run()
        rows = self.conn.execute(
            "SELECT entity_name FROM hits WHERE source_url = ? ORDER BY entity_name",
            (self.REGISTER_URL,),
        ).fetchall()
        self.assertEqual(
            [r["entity_name"] for r in rows],
            ["Acme Digital (Pvt) Limited", "Orion Connect Services Limited",
             "Zuma Resources Limited"],
        )

    def test_register_rows_are_tier_a(self):
        self._run()
        rows = self.conn.execute(
            "SELECT tier FROM hits WHERE source_url = ?", (self.REGISTER_URL,)
        ).fetchall()
        self.assertTrue(rows)
        self.assertTrue(all(r["tier"] == "A" for r in rows))

    def test_excerpt_quotes_the_licensee_row(self):
        self._run()
        row = self.conn.execute(
            "SELECT verbatim_excerpt FROM hits WHERE source_url = ? "
            "AND entity_name = ?",
            (self.REGISTER_URL, "Orion Connect Services Limited"),
        ).fetchone()
        self.assertIn("Orion Connect Services Limited", row["verbatim_excerpt"])

    def test_assets_media_pdfs_are_crawled(self):
        """Regression: /assets/media/ was excluded by the link filter."""
        from mvno_watcher.sources.pta import _ARTICLE_RE
        self.assertTrue(_ARTICLE_RE.search(self.REGISTER_URL))

    def test_register_is_exempt_from_the_date_window(self):
        """A register is current state, not news: a since-filter must not drop it."""
        from mvno_watcher.sources.pta import PTASource
        src = PTASource(index_urls=[
            "https://www.pta.gov.pk/en/media-center/press-releases"
        ])
        pipeline.run(self.conn, [src], since="2030-01-01", fire_alerts=False)
        rows = self.conn.execute(
            "SELECT COUNT(*) c FROM hits WHERE source_url = ?",
            (self.REGISTER_URL,),
        ).fetchone()
        self.assertEqual(rows["c"], 3)
