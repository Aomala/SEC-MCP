"""Tests for sec_mcp.xbrl_mappings — industry classification and concept maps."""

import pytest

from sec_mcp.xbrl_mappings import (
    CONCEPT_MAP,
    REVENUE_MAP,
    ConceptEntry,
    IndustryClass,
    detect_industry_class,
    get_revenue_concepts,
)


class TestDetectIndustryClass:
    def test_bank_sic_6020(self):
        assert detect_industry_class(6020) == IndustryClass.BANK

    def test_insurance_sic_6310(self):
        assert detect_industry_class(6310) == IndustryClass.INSURANCE

    def test_reit_sic_6500(self):
        assert detect_industry_class(6500) == IndustryClass.REIT

    def test_utility_sic_4900(self):
        assert detect_industry_class(4900) == IndustryClass.UTILITY

    def test_standard_sic_7372(self):
        assert detect_industry_class(7372) == IndustryClass.STANDARD

    def test_none_sic_returns_standard(self):
        assert detect_industry_class(None) == IndustryClass.STANDARD

    def test_string_sic_code(self):
        assert detect_industry_class("6020") == IndustryClass.BANK

    def test_invalid_sic_returns_standard(self):
        assert detect_industry_class("not_a_number") == IndustryClass.STANDARD

    def test_ticker_override_coin(self):
        assert detect_industry_class(None, ticker="COIN") == IndustryClass.CRYPTO

    def test_ticker_override_takes_priority(self):
        # Even with a bank SIC, COIN ticker should override to crypto
        assert detect_industry_class(6020, ticker="COIN") == IndustryClass.CRYPTO


class TestRevenueConcepts:
    def test_revenue_concepts_exist_for_standard(self):
        concepts = get_revenue_concepts(IndustryClass.STANDARD)
        assert len(concepts) > 0
        assert all(isinstance(c, ConceptEntry) for c in concepts)

    def test_revenue_concepts_exist_for_bank(self):
        concepts = get_revenue_concepts(IndustryClass.BANK)
        assert len(concepts) > 0

    def test_revenue_concepts_exist_for_insurance(self):
        concepts = get_revenue_concepts(IndustryClass.INSURANCE)
        assert len(concepts) > 0

    def test_revenue_concepts_exist_for_reit(self):
        concepts = get_revenue_concepts(IndustryClass.REIT)
        assert len(concepts) > 0

    def test_revenue_concepts_exist_for_utility(self):
        concepts = get_revenue_concepts(IndustryClass.UTILITY)
        assert len(concepts) > 0

    def test_revenue_concepts_exist_for_crypto(self):
        concepts = get_revenue_concepts(IndustryClass.CRYPTO)
        assert len(concepts) > 0

    def test_all_industry_classes_in_revenue_map(self):
        for cls in IndustryClass:
            assert cls in REVENUE_MAP, f"{cls} missing from REVENUE_MAP"


class TestConceptMap:
    def test_has_revenue_key(self):
        # Revenue is handled via REVENUE_MAP, not CONCEPT_MAP
        # but net_income, total_assets etc. should be in CONCEPT_MAP
        assert "net_income" in CONCEPT_MAP

    def test_has_total_assets(self):
        assert "total_assets" in CONCEPT_MAP

    def test_has_operating_income(self):
        assert "operating_income" in CONCEPT_MAP

    def test_has_eps_diluted(self):
        assert "eps_diluted" in CONCEPT_MAP

    def test_has_cash_and_equivalents(self):
        assert "cash_and_equivalents" in CONCEPT_MAP

    def test_has_stockholders_equity(self):
        assert "stockholders_equity" in CONCEPT_MAP

    def test_concept_entries_are_lists(self):
        for key, entries in CONCEPT_MAP.items():
            assert isinstance(entries, list), f"CONCEPT_MAP[{key}] is not a list"
            assert len(entries) > 0, f"CONCEPT_MAP[{key}] is empty"

    def test_concept_entries_are_named_tuples(self):
        for key, entries in CONCEPT_MAP.items():
            for entry in entries:
                assert isinstance(entry, ConceptEntry), (
                    f"CONCEPT_MAP[{key}] contains non-ConceptEntry: {entry}"
                )
                assert isinstance(entry.xbrl_concept, str)
                assert isinstance(entry.display_name, str)
