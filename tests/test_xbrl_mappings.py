"""Tests for sec_mcp.xbrl_mappings — industry classification and concept maps."""

import pytest

from sec_mcp.xbrl_mappings import (
    BANK_NONINTEREST_SUBDRIVERS,
    BANK_REVENUE_DRIVERS,
    CONCEPT_MAP,
    OPERATING_INCOME,
    OPERATING_INCOME_FINANCIAL,
    REVENUE_MAP,
    ConceptEntry,
    IndustryClass,
    detect_industry_class,
    get_bank_revenue_drivers,
    get_operating_income_concepts,
    get_revenue_concepts,
)

_PRETAX = "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"


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


class TestBankRevenueDrivers:
    def test_getter_returns_mapping(self):
        assert get_bank_revenue_drivers() is BANK_REVENUE_DRIVERS

    def test_tier1_drivers_present(self):
        # The reconciling backbone must exist.
        assert "net_interest_income" in BANK_REVENUE_DRIVERS
        assert "noninterest_income" in BANK_REVENUE_DRIVERS

    def test_entries_well_formed(self):
        for driver, entries in BANK_REVENUE_DRIVERS.items():
            assert isinstance(entries, list) and entries, f"{driver} empty"
            for e in entries:
                assert isinstance(e, ConceptEntry)
                assert isinstance(e.xbrl_concept, str) and e.xbrl_concept
                assert isinstance(e.display_name, str) and e.display_name

    def test_pick_first_not_aggregate(self):
        # Drivers are resolved pick-first; no entry should be flagged aggregate
        # (aggregate=True belongs to REVENUE_BANK's summed total, not here — a
        # stray aggregate flag would change resolution semantics).
        for driver, entries in BANK_REVENUE_DRIVERS.items():
            for e in entries:
                assert e.aggregate is False, f"{driver}/{e.xbrl_concept} is aggregate"

    def test_subdrivers_are_known_drivers(self):
        # Every noninterest sub-driver must be a real driver key (used to derive
        # the "Other fee income" residual plug).
        for key in BANK_NONINTEREST_SUBDRIVERS:
            assert key in BANK_REVENUE_DRIVERS, f"sub-driver {key} not a driver"

    def test_subdrivers_exclude_tier1(self):
        assert "net_interest_income" not in BANK_NONINTEREST_SUBDRIVERS
        assert "noninterest_income" not in BANK_NONINTEREST_SUBDRIVERS

    def test_no_duplicate_concepts_within_driver(self):
        for driver, entries in BANK_REVENUE_DRIVERS.items():
            tags = [e.xbrl_concept for e in entries]
            assert len(tags) == len(set(tags)), f"{driver} has duplicate tags"


class TestOperatingIncomeSplit:
    def test_standard_list_has_no_pretax_fallback(self):
        # Standard filers must NOT fall back to pretax income (it mislabels
        # pretax as operating income for JNJ/LLY/GE/O/XOM).
        tags = [e.xbrl_concept for e in OPERATING_INCOME]
        assert _PRETAX not in tags

    def test_financials_have_pretax_as_fallback(self):
        # Financials get pretax as a LAST-RESORT fallback (not first): a real
        # OperatingIncomeLoss (UNH) still wins; only filers with no op-income tag
        # (banks, BRK-B) fall through to the pretax proxy.
        tags = [e.xbrl_concept for e in OPERATING_INCOME_FINANCIAL]
        assert _PRETAX in tags
        assert tags[0] == "OperatingIncomeLoss"          # real op-income wins first
        assert tags.index("OperatingIncomeLoss") < tags.index(_PRETAX)

    def test_getter_routes_by_industry(self):
        for ind in (IndustryClass.BANK, IndustryClass.INSURANCE):
            assert get_operating_income_concepts(ind) is OPERATING_INCOME_FINANCIAL
        for ind in (IndustryClass.STANDARD, IndustryClass.REIT,
                    IndustryClass.UTILITY, IndustryClass.CRYPTO):
            assert get_operating_income_concepts(ind) is OPERATING_INCOME

    def test_real_operating_income_still_first_for_standard(self):
        # A filer that DOES tag real operating income (AAPL) is unaffected.
        assert OPERATING_INCOME[0].xbrl_concept == "OperatingIncomeLoss"
