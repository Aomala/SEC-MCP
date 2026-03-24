"""Production-grade XBRL concept → canonical metric mappings.

Three-layer architecture:
  Layer 1 — Canonical schema  (metric key → display name)
  Layer 2 — Synonym mapping   (each canonical maps to ordered XBRL tags)
  Layer 3 — LLM disambiguation (called only when ambiguous — see financials.py)

Industry-specific mappings:
  - Standard corporates (tech, consumer, industrial, healthcare, SaaS)
  - Banks & broker-dealers (MS, GS, JPM)
  - Insurance (premiums + investment income + loss ratios)
  - REITs (lease revenue, FFO, NOI, occupancy)
  - Utilities (electric + gas)
  - Crypto / fintech (COIN, MSTR, MARA — standard GAAP + crypto-aware)
  - Custom extension filers (ms_NetRevenues, gs_TotalNetRevenues, etc.)
  - SaaS (deferred revenue, RPO, subscription revenue)
  - Healthcare / pharma (drug revenue, milestone payments)
  - Energy / oil & gas (production revenue, proved reserves)
  - Retail (same-store sales, comparable store)
  - Fintech / payments (transaction volume, take rate)

Non-GAAP metrics that are NOT in XBRL but exist in filing text:
  - ARR (annual recurring revenue) — SaaS
  - Same-store sales / comps — Retail
  - FFO (funds from operations) — REIT
  - CASM / RASM — Airlines
  - Production volume / realized price — Energy
  These require filing-text NLP extraction, not XBRL parsing.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import NamedTuple


# ═══════════════════════════════════════════════════════════════════════════
#  Industry classification
# ═══════════════════════════════════════════════════════════════════════════

class IndustryClass(str, Enum):
    STANDARD = "standard"
    BANK = "bank"
    INSURANCE = "insurance"
    REIT = "reit"
    UTILITY = "utility"
    CRYPTO = "crypto"


# SIC-based classification
_SIC_RANGES: list[tuple[tuple[int, int], IndustryClass]] = [
    ((6020, 6029), IndustryClass.BANK),
    ((6030, 6039), IndustryClass.BANK),
    ((6040, 6062), IndustryClass.BANK),
    ((6140, 6159), IndustryClass.BANK),
    # 6199 is NOT mapped to bank — it's generic "finance services"
    # (crypto exchanges, payment processors, etc.)
    ((6200, 6299), IndustryClass.BANK),   # security & commodity brokers
    ((6310, 6399), IndustryClass.INSURANCE),
    ((6411, 6411), IndustryClass.INSURANCE),
    ((6500, 6799), IndustryClass.REIT),   # Real estate (6512, 6552, 6726, 6798 REITs)
    ((4900, 4991), IndustryClass.UTILITY),
]

# Ticker-based overrides (more reliable than SIC for edge cases)
_TICKER_OVERRIDES: dict[str, IndustryClass] = {
    # ── Crypto / digital assets ──
    "COIN": IndustryClass.CRYPTO,
    "MSTR": IndustryClass.CRYPTO,
    "MARA": IndustryClass.CRYPTO,
    "RIOT": IndustryClass.CRYPTO,
    "CLSK": IndustryClass.CRYPTO,
    "HUT": IndustryClass.CRYPTO,
    "BITF": IndustryClass.CRYPTO,
    "CIFR": IndustryClass.CRYPTO,
    "BTBT": IndustryClass.CRYPTO,
    "WULF": IndustryClass.CRYPTO,
    "IREN": IndustryClass.CRYPTO,
    "CORZ": IndustryClass.CRYPTO,
    "BTDR": IndustryClass.CRYPTO,
    "GREE": IndustryClass.CRYPTO,
    "ARBK": IndustryClass.CRYPTO,
    # ── REITs ──
    "SPG": IndustryClass.REIT,      # Simon Property Group
    "O": IndustryClass.REIT,        # Realty Income
    "PLD": IndustryClass.REIT,      # Prologis
    "AMT": IndustryClass.REIT,      # American Tower
    "EQIX": IndustryClass.REIT,     # Equinix
    "DLR": IndustryClass.REIT,      # Digital Realty
    "PSA": IndustryClass.REIT,      # Public Storage
    "WELL": IndustryClass.REIT,     # Welltower
    "AVB": IndustryClass.REIT,      # AvalonBay
    "EQR": IndustryClass.REIT,      # Equity Residential
    "VTR": IndustryClass.REIT,      # Ventas
    "ARE": IndustryClass.REIT,      # Alexandria Real Estate
    "MAA": IndustryClass.REIT,      # Mid-America Apartment
    "UDR": IndustryClass.REIT,      # UDR Inc
    "ESS": IndustryClass.REIT,      # Essex Property Trust
    "CPT": IndustryClass.REIT,      # Camden Property Trust
    "INVH": IndustryClass.REIT,     # Invitation Homes
    "SUI": IndustryClass.REIT,      # Sun Communities
    "ELS": IndustryClass.REIT,      # Equity LifeStyle
    "CUBE": IndustryClass.REIT,     # CubeSmart
    "EXR": IndustryClass.REIT,      # Extra Space Storage
    "LSI": IndustryClass.REIT,      # Life Storage
    "REG": IndustryClass.REIT,      # Regency Centers
    "FRT": IndustryClass.REIT,      # Federal Realty
    "KIM": IndustryClass.REIT,      # Kimco Realty
    "BXP": IndustryClass.REIT,      # Boston Properties
    "SLG": IndustryClass.REIT,      # SL Green Realty
    "VNO": IndustryClass.REIT,      # Vornado Realty
    "HIW": IndustryClass.REIT,      # Highwoods Properties
    "CCI": IndustryClass.REIT,      # Crown Castle
    "SBAC": IndustryClass.REIT,     # SBA Communications
    "MPW": IndustryClass.REIT,      # Medical Properties Trust
    "OHI": IndustryClass.REIT,      # Omega Healthcare
    "NNN": IndustryClass.REIT,      # NNN REIT
    "WPC": IndustryClass.REIT,      # W. P. Carey
    "STAG": IndustryClass.REIT,     # STAG Industrial
    # ── Insurance ──
    "BRK-A": IndustryClass.INSURANCE,
    "BRK-B": IndustryClass.INSURANCE,
    "BRK.A": IndustryClass.INSURANCE,
    "BRK.B": IndustryClass.INSURANCE,
    "ALL": IndustryClass.INSURANCE,     # Allstate
    "PGR": IndustryClass.INSURANCE,     # Progressive
    "MET": IndustryClass.INSURANCE,     # MetLife
    "AIG": IndustryClass.INSURANCE,     # AIG
    "PRU": IndustryClass.INSURANCE,     # Prudential
    "TRV": IndustryClass.INSURANCE,     # Travelers
    "AFL": IndustryClass.INSURANCE,     # Aflac
    "HIG": IndustryClass.INSURANCE,     # Hartford Financial
    "CB": IndustryClass.INSURANCE,      # Chubb
    "CINF": IndustryClass.INSURANCE,    # Cincinnati Financial
    "GL": IndustryClass.INSURANCE,      # Globe Life
    "LNC": IndustryClass.INSURANCE,     # Lincoln National
    "UNM": IndustryClass.INSURANCE,     # Unum Group
    "RGA": IndustryClass.INSURANCE,     # Reinsurance Group
    "EG": IndustryClass.INSURANCE,      # Everest Group
    "RNR": IndustryClass.INSURANCE,     # RenaissanceRe
    "ACGL": IndustryClass.INSURANCE,    # Arch Capital
    "WRB": IndustryClass.INSURANCE,     # Berkley
    "ERIE": IndustryClass.INSURANCE,    # Erie Indemnity
    "KNSL": IndustryClass.INSURANCE,    # Kinsale Capital
    # ── Utilities ──
    "NEE": IndustryClass.UTILITY,     # NextEra Energy
    "DUK": IndustryClass.UTILITY,     # Duke Energy
    "SO": IndustryClass.UTILITY,      # Southern Company
    "AEP": IndustryClass.UTILITY,     # American Electric Power
    "D": IndustryClass.UTILITY,       # Dominion Energy
    "SRE": IndustryClass.UTILITY,     # Sempra Energy
    "EXC": IndustryClass.UTILITY,     # Exelon
    "XEL": IndustryClass.UTILITY,     # Xcel Energy
    "ED": IndustryClass.UTILITY,      # Consolidated Edison
    "WEC": IndustryClass.UTILITY,     # WEC Energy
    "ES": IndustryClass.UTILITY,      # Eversource
    "AEE": IndustryClass.UTILITY,     # Ameren
    "DTE": IndustryClass.UTILITY,     # DTE Energy
    "ETR": IndustryClass.UTILITY,     # Entergy
    "CMS": IndustryClass.UTILITY,     # CMS Energy
    "FE": IndustryClass.UTILITY,      # FirstEnergy
    "PPL": IndustryClass.UTILITY,     # PPL Corp
    "CEG": IndustryClass.UTILITY,     # Constellation Energy
    "PCG": IndustryClass.UTILITY,     # PG&E
    "EIX": IndustryClass.UTILITY,     # Edison International
    "AWK": IndustryClass.UTILITY,     # American Water Works
    "NI": IndustryClass.UTILITY,      # NiSource
    "EVRG": IndustryClass.UTILITY,    # Evergy
    "ATO": IndustryClass.UTILITY,     # Atmos Energy
    "CNP": IndustryClass.UTILITY,     # CenterPoint Energy
    "PNW": IndustryClass.UTILITY,     # Pinnacle West
    "LNT": IndustryClass.UTILITY,     # Alliant Energy
    "OGE": IndustryClass.UTILITY,     # OGE Energy
    # ── Banks (explicit overrides for edge cases) ──
    "JPM": IndustryClass.BANK,
    "BAC": IndustryClass.BANK,
    "WFC": IndustryClass.BANK,
    "C": IndustryClass.BANK,
    "GS": IndustryClass.BANK,
    "MS": IndustryClass.BANK,
    "USB": IndustryClass.BANK,
    "PNC": IndustryClass.BANK,
    "TFC": IndustryClass.BANK,
    "SCHW": IndustryClass.BANK,
    "BK": IndustryClass.BANK,
    "STT": IndustryClass.BANK,
    "FITB": IndustryClass.BANK,
    "KEY": IndustryClass.BANK,
    "HBAN": IndustryClass.BANK,
    "CFG": IndustryClass.BANK,
    "MTB": IndustryClass.BANK,
    "RF": IndustryClass.BANK,
    "ZION": IndustryClass.BANK,
    "CMA": IndustryClass.BANK,
    "ALLY": IndustryClass.BANK,
    "COF": IndustryClass.BANK,
    "DFS": IndustryClass.BANK,
    "SYF": IndustryClass.BANK,
}


def detect_industry_class(
    sic_code: str | int | None,
    ticker: str | None = None,
) -> IndustryClass:
    """Map SIC code (and optional ticker override) to IndustryClass."""
    # Ticker override takes priority
    if ticker:
        override = _TICKER_OVERRIDES.get(ticker.upper().strip())
        if override is not None:
            return override

    if sic_code is None:
        return IndustryClass.STANDARD
    try:
        sic = int(str(sic_code).strip())
    except (ValueError, TypeError):
        return IndustryClass.STANDARD
    for (lo, hi), cls in _SIC_RANGES:
        if lo <= sic <= hi:
            return cls
    return IndustryClass.STANDARD


# ═══════════════════════════════════════════════════════════════════════════
#  Concept entry
# ═══════════════════════════════════════════════════════════════════════════

class ConceptEntry(NamedTuple):
    xbrl_concept: str       # tag name (without us-gaap: prefix)
    display_name: str       # human label
    aggregate: bool = False # True = add to running total instead of replace


# ═══════════════════════════════════════════════════════════════════════════
#  REVENUE — Standard / Industrial / Tech / SaaS
#  Canonical: net_revenue
#  Labels: Revenue, Net Revenue, Total Revenue, Net Sales, Sales,
#          Operating Revenue, Consolidated Revenue, Net Turnover,
#          Contract Revenue, Customer Revenue
# ═══════════════════════════════════════════════════════════════════════════

REVENUE_STANDARD: list[ConceptEntry] = [
    ConceptEntry("Revenues", "Total Revenue"),
    ConceptEntry("Revenue", "Revenue"),
    ConceptEntry("SalesRevenueNet", "Net Sales Revenue"),
    ConceptEntry("RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenue from Contract with Customer"),
    ConceptEntry("RevenueFromContractWithCustomerIncludingAssessedTax",
                 "Revenue from Contract with Customer (incl. tax)"),
    ConceptEntry("TotalRevenues", "Total Revenues"),
    ConceptEntry("TotalRevenue", "Total Revenue (alt)"),
    ConceptEntry("OperatingRevenue", "Operating Revenue"),
    ConceptEntry("SalesRevenueServicesNet", "Net Services Revenue"),
    ConceptEntry("SalesRevenueGoodsNet", "Net Goods Revenue"),
    ConceptEntry("NetRevenues", "Net Revenues"),
    ConceptEntry("TotalRevenuesAndOtherIncome", "Total Revenues and Other Income"),
    # IFRS fallbacks (Alibaba, ASML, SAP, etc.)
    ConceptEntry("RevenueFromContractsWithCustomers", "IFRS Revenue"),
    ConceptEntry("Revenue", "IFRS Revenue (alt)"),
    ConceptEntry("Turnover", "IFRS Turnover"),
    ConceptEntry("GrossRevenue", "Gross Revenue"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  REVENUE — Banks / Broker-Dealers
#  Canonical: net_revenue
#  Labels: Net Revenues, Revenue Net of Interest Expense
# ═══════════════════════════════════════════════════════════════════════════

REVENUE_BANK: list[ConceptEntry] = [
    # Top-level totals first
    ConceptEntry("NetRevenues", "Net Revenues"),
    ConceptEntry("TotalNetRevenues", "Total Net Revenues"),
    ConceptEntry("RevenuesNetOfInterestExpense", "Revenue Net of Interest Expense"),
    ConceptEntry("RevenueNetOfInterestExpense", "Revenue Net of Interest Expense (alt)"),
    ConceptEntry("RevenuesNetOfInterestExpenseAfterProvisionForCreditLosses",
                 "Revenue Net of Int Exp After Provision"),
    ConceptEntry("Revenues", "Total Revenue"),
    # Fallback components — aggregated if no total found
    ConceptEntry("InterestIncomeOperating", "Interest Income (Operating)", aggregate=True),
    ConceptEntry("InterestAndDividendIncomeOperating",
                 "Interest & Dividend Income", aggregate=True),
    ConceptEntry("InterestIncome", "Interest Income", aggregate=True),
    ConceptEntry("InterestIncomeAfterProvisionForCreditLosses",
                 "Interest Income After Provision", aggregate=True),
    ConceptEntry("NetInterestIncome", "Net Interest Income", aggregate=True),
    ConceptEntry("InterestIncomeExpenseNet", "Net Interest Income (alt)", aggregate=True),
    ConceptEntry("NoninterestIncome", "Noninterest Income", aggregate=True),
    ConceptEntry("FeeAndCommissionIncome", "Fee & Commission Income", aggregate=True),
    ConceptEntry("InvestmentBankingRevenue", "Investment Banking Revenue", aggregate=True),
    ConceptEntry("TradingGainsLosses", "Trading Gains/Losses", aggregate=True),
    ConceptEntry("TradingRevenue", "Trading Revenue", aggregate=True),
    ConceptEntry("AssetManagementFees", "Asset Management Fees", aggregate=True),
    ConceptEntry("InvestmentManagementFees", "Investment Management Fees", aggregate=True),
    ConceptEntry("WealthManagementRevenue", "Wealth Management Revenue", aggregate=True),
    ConceptEntry("BrokerageCommissionsRevenue", "Brokerage Commissions", aggregate=True),
    ConceptEntry("FeesAndCommissions", "Fees and Commissions", aggregate=True),
    ConceptEntry("InsuranceServicesRevenue", "Insurance Services Revenue", aggregate=True),
    ConceptEntry("GainLossOnInvestments", "Gain/Loss on Investments", aggregate=True),
    ConceptEntry("OtherNoninterestIncome", "Other Noninterest Income", aggregate=True),
]

# ═══════════════════════════════════════════════════════════════════════════
#  REVENUE — Insurance
#  Canonical: premiums_earned + investment income
# ═══════════════════════════════════════════════════════════════════════════

REVENUE_INSURANCE: list[ConceptEntry] = [
    ConceptEntry("Revenues", "Total Revenue"),
    ConceptEntry("NetRevenues", "Net Revenues"),
    ConceptEntry("InsuranceRevenue", "Insurance Revenue"),
    ConceptEntry("PremiumsEarnedNet", "Net Premiums Earned", aggregate=True),
    ConceptEntry("PremiumsEarned", "Premiums Earned", aggregate=True),
    ConceptEntry("PremiumsWrittenNet", "Net Premiums Written", aggregate=True),
    ConceptEntry("ReinsurancePremiumsEarned", "Reinsurance Premiums", aggregate=True),
    ConceptEntry("PolicyChargesAndFeeIncome", "Policy Charges & Fees", aggregate=True),
    ConceptEntry("NetInvestmentIncome", "Net Investment Income", aggregate=True),
    ConceptEntry("GainLossOnInvestments", "Gain/Loss on Investments", aggregate=True),
    ConceptEntry("FeesAndCommissions", "Fees and Commissions", aggregate=True),
    ConceptEntry("OtherIncome", "Other Income", aggregate=True),
]

# ═══════════════════════════════════════════════════════════════════════════
#  REVENUE — REITs
# ═══════════════════════════════════════════════════════════════════════════

REVENUE_REIT: list[ConceptEntry] = [
    ConceptEntry("Revenues", "Total Revenue"),
    ConceptEntry("RealEstateRevenueNet", "Net Real Estate Revenue"),
    ConceptEntry("RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenue from Contract with Customer"),
    ConceptEntry("OperatingLeasesIncomeStatementLeaseRevenue",
                 "Operating Lease Revenue", aggregate=True),
    ConceptEntry("LeaseIncome", "Lease Income", aggregate=True),
    ConceptEntry("RentalRevenue", "Rental Revenue", aggregate=True),
    ConceptEntry("TenantReimbursements", "Tenant Reimbursements", aggregate=True),
    ConceptEntry("ManagementFeesRevenue", "Management Fees Revenue", aggregate=True),
    ConceptEntry("OtherIncome", "Other Income", aggregate=True),
]

# ═══════════════════════════════════════════════════════════════════════════
#  REVENUE — Utilities
# ═══════════════════════════════════════════════════════════════════════════

REVENUE_UTILITY: list[ConceptEntry] = [
    ConceptEntry("Revenues", "Total Revenue"),
    ConceptEntry("RegulatedAndUnregulatedOperatingRevenue",
                 "Regulated & Unregulated Revenue"),
    ConceptEntry("RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenue from Contract with Customer"),
    ConceptEntry("ElectricUtilityRevenue", "Electric Utility Revenue", aggregate=True),
    ConceptEntry("GasUtilityRevenue", "Gas Utility Revenue", aggregate=True),
    ConceptEntry("RegulatedElectricRevenue", "Regulated Electric Revenue", aggregate=True),
    ConceptEntry("RegulatedGasRevenue", "Regulated Gas Revenue", aggregate=True),
    ConceptEntry("UnregulatedRevenue", "Unregulated Revenue", aggregate=True),
]

# ═══════════════════════════════════════════════════════════════════════════
#  REVENUE — Crypto / Fintech
#  Uses standard GAAP tags but with crypto-aware ordering
# ═══════════════════════════════════════════════════════════════════════════

REVENUE_CRYPTO: list[ConceptEntry] = [
    ConceptEntry("Revenues", "Total Revenue"),
    ConceptEntry("Revenue", "Revenue"),
    ConceptEntry("RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenue from Contract with Customer"),
    ConceptEntry("NetRevenues", "Net Revenues"),
    ConceptEntry("TotalRevenues", "Total Revenues"),
    ConceptEntry("SalesRevenueNet", "Net Sales Revenue"),
    ConceptEntry("OperatingRevenue", "Operating Revenue"),
]

REVENUE_MAP: dict[IndustryClass, list[ConceptEntry]] = {
    IndustryClass.STANDARD: REVENUE_STANDARD,
    IndustryClass.BANK: REVENUE_BANK,
    IndustryClass.INSURANCE: REVENUE_INSURANCE,
    IndustryClass.REIT: REVENUE_REIT,
    IndustryClass.UTILITY: REVENUE_UTILITY,
    IndustryClass.CRYPTO: REVENUE_CRYPTO,
}


def get_revenue_concepts(industry: IndustryClass) -> list[ConceptEntry]:
    return REVENUE_MAP.get(industry, REVENUE_STANDARD)


# ═══════════════════════════════════════════════════════════════════════════
#  COST OF REVENUE
#  Canonical: cost_of_revenue
#  Labels: Cost of Revenue, COGS, Cost of Goods Sold, Cost of Sales,
#          Cost of Services, Cost of Products Sold,
#          Cost of Subscription Revenue, Traffic Acquisition Costs
# ═══════════════════════════════════════════════════════════════════════════

COST_OF_REVENUE: list[ConceptEntry] = [
    ConceptEntry("CostOfRevenue", "Cost of Revenue"),
    ConceptEntry("CostOfGoodsAndServicesSold", "Cost of Goods & Services Sold"),
    ConceptEntry("CostOfGoodsSold", "Cost of Goods Sold"),
    ConceptEntry("CostOfServices", "Cost of Services"),
    ConceptEntry("CostOfProductsSold", "Cost of Products Sold"),
    # IFRS
    ConceptEntry("CostOfSales", "IFRS Cost of Sales"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  GROSS PROFIT
# ═══════════════════════════════════════════════════════════════════════════

GROSS_PROFIT: list[ConceptEntry] = [
    ConceptEntry("GrossProfit", "Gross Profit"),
    ConceptEntry("GrossProfitLoss", "Gross Profit (Loss)"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  OPERATING EXPENSES (total)
#  Canonical: operating_expenses
#  Labels: Operating Expenses, Total OpEx
# ═══════════════════════════════════════════════════════════════════════════

OPERATING_EXPENSES: list[ConceptEntry] = [
    ConceptEntry("OperatingExpenses", "Operating Expenses"),
    ConceptEntry("CostsAndExpenses", "Costs and Expenses"),
    ConceptEntry("OperatingCostsAndExpenses", "Operating Costs and Expenses"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  SG&A
#  Canonical: sga_expense
#  Labels: SG&A, Selling Expense, General and Administrative,
#          Sales and Marketing
# ═══════════════════════════════════════════════════════════════════════════

SGA_EXPENSE: list[ConceptEntry] = [
    ConceptEntry("SellingGeneralAndAdministrativeExpense", "SG&A Expense"),
    ConceptEntry("GeneralAndAdministrativeExpense", "G&A Expense"),
    ConceptEntry("SellingAndMarketingExpense", "Selling & Marketing"),
    ConceptEntry("SellingExpense", "Selling Expense"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  R&D
#  Canonical: rd_expense
#  Labels: R&D, Research and Development, Engineering Expense
# ═══════════════════════════════════════════════════════════════════════════

RD_EXPENSE: list[ConceptEntry] = [
    ConceptEntry("ResearchAndDevelopmentExpense", "R&D Expense"),
    ConceptEntry("ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
                 "R&D Expense (excl. acquired)"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  NET INCOME
#  Canonical: net_income
#  Labels: Net Income, Profit/Loss, Net Income Attributable to Parent
# ═══════════════════════════════════════════════════════════════════════════

NET_INCOME: list[ConceptEntry] = [
    ConceptEntry("NetIncomeLoss", "Net Income (Loss)"),
    ConceptEntry("ProfitLoss", "Profit (Loss)"),
    ConceptEntry("NetIncome", "Net Income"),
    ConceptEntry("NetIncomeAttributableToParent", "Net Income Attributable to Parent"),
    ConceptEntry("NetIncomeLossAttributableToParent",
                 "Net Income (Loss) Attributable to Parent"),
    ConceptEntry("NetIncomeLossAvailableToCommonStockholdersBasic",
                 "Net Income Available to Common (Basic)"),
    ConceptEntry("IncomeLossFromContinuingOperations",
                 "Income from Continuing Operations"),
    ConceptEntry("ComprehensiveIncomeNetOfTax",
                 "Comprehensive Income Net of Tax"),
    # IFRS
    ConceptEntry("ProfitLossAttributableToOwnersOfParent",
                 "IFRS Profit Attributable to Parent"),
    ConceptEntry("Profit", "IFRS Profit"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  OPERATING INCOME
#  Canonical: operating_income
#  Labels: Operating Income, Income from Operations
# ═══════════════════════════════════════════════════════════════════════════

OPERATING_INCOME: list[ConceptEntry] = [
    ConceptEntry("OperatingIncomeLoss", "Operating Income (Loss)"),
    ConceptEntry("IncomeLossFromOperations", "Income from Operations"),
    ConceptEntry("OperatingProfitLoss", "Operating Profit (Loss)"),
    # IFRS
    ConceptEntry("ProfitLossFromOperatingActivities", "IFRS Operating Profit"),
    ConceptEntry("OperatingProfit", "IFRS Operating Profit (alt)"),
    ConceptEntry("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                 "Income Before Tax"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  OTHER INCOME / EXPENSE
#  Canonical: other_income_expense
# ═══════════════════════════════════════════════════════════════════════════

OTHER_INCOME_EXPENSE: list[ConceptEntry] = [
    ConceptEntry("NonoperatingIncomeExpense", "Nonoperating Income (Expense)"),
    ConceptEntry("OtherNonoperatingIncomeExpense", "Other Nonoperating Income (Expense)"),
    ConceptEntry("OtherIncomeExpenseNet", "Other Income (Expense), Net"),
    ConceptEntry("OtherOperatingIncomeExpenseNet", "Other Operating Income (Expense), Net"),
    ConceptEntry("OtherIncome", "Other Income"),
    ConceptEntry("OtherExpenses", "Other Expenses"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  INCOME BEFORE TAX
#  Canonical: income_before_tax
# ═══════════════════════════════════════════════════════════════════════════

INCOME_BEFORE_TAX: list[ConceptEntry] = [
    ConceptEntry("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                 "Income Before Tax"),
    ConceptEntry("IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
                 "Income Before Tax (alt)"),
    ConceptEntry("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
                 "Income Before Tax (detailed)"),
    ConceptEntry("IncomeBeforeIncomeTaxes", "Income Before Income Taxes"),
    # IFRS
    ConceptEntry("ProfitBeforeTax", "IFRS Profit Before Tax"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  INTEREST EXPENSE
#  Canonical: interest_expense
# ═══════════════════════════════════════════════════════════════════════════

INTEREST_EXPENSE: list[ConceptEntry] = [
    ConceptEntry("InterestExpense", "Interest Expense"),
    ConceptEntry("InterestExpenseDebt", "Interest Expense on Debt"),
    ConceptEntry("InterestExpenseBorrowings", "Interest Expense on Borrowings"),
    ConceptEntry("InterestIncomeExpenseNonoperatingNet",
                 "Net Nonoperating Interest"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  INCOME TAX EXPENSE
#  Canonical: income_tax_expense
# ═══════════════════════════════════════════════════════════════════════════

INCOME_TAX_EXPENSE: list[ConceptEntry] = [
    ConceptEntry("IncomeTaxExpenseBenefit", "Income Tax Expense (Benefit)"),
    ConceptEntry("IncomeTaxesPaid", "Income Taxes Paid"),
    ConceptEntry("CurrentIncomeTaxExpenseBenefit", "Current Income Tax Expense"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  SHARE-BASED COMPENSATION
#  Canonical: share_based_compensation
# ═══════════════════════════════════════════════════════════════════════════

SHARE_BASED_COMPENSATION: list[ConceptEntry] = [
    ConceptEntry("ShareBasedCompensation", "Stock-Based Compensation"),
    ConceptEntry("AllocatedShareBasedCompensationExpense",
                 "Allocated Share-Based Compensation"),
    ConceptEntry("ShareBasedCompensationExpense", "Share-Based Compensation Expense"),
    ConceptEntry("EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized",
                 "SBC Cost Not Yet Recognized"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  PROVISION FOR CREDIT LOSSES (Banking)
#  Canonical: provision_for_credit_losses
#  Labels: Provision for Loan Losses, Credit Loss Expense
# ═══════════════════════════════════════════════════════════════════════════

PROVISION_FOR_CREDIT_LOSSES: list[ConceptEntry] = [
    ConceptEntry("ProvisionForLoanLeaseAndOtherLosses",
                 "Provision for Loan & Lease Losses"),
    ConceptEntry("ProvisionForLoanLossesExpensed",
                 "Provision for Loan Losses"),
    ConceptEntry("ProvisionForCreditLosses", "Provision for Credit Losses"),
    ConceptEntry("CreditLossExpenseReversal", "Credit Loss Expense (Reversal)"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  EBITDA building blocks
# ═══════════════════════════════════════════════════════════════════════════

DEPRECIATION_AMORTIZATION: list[ConceptEntry] = [
    ConceptEntry("DepreciationDepletionAndAmortization", "D&A"),
    ConceptEntry("DepreciationAndAmortization", "D&A (alt)"),
    ConceptEntry("Depreciation", "Depreciation"),
    ConceptEntry("AmortizationOfIntangibleAssets", "Amortization of Intangibles"),
]

EBITDA_COMPONENTS: list[ConceptEntry] = [
    ConceptEntry("OperatingIncomeLoss", "Operating Income"),
    ConceptEntry("DepreciationDepletionAndAmortization",
                 "Depreciation & Amortization", aggregate=True),
    ConceptEntry("DepreciationAndAmortization",
                 "Depreciation & Amortization (alt)", aggregate=True),
    ConceptEntry("Depreciation", "Depreciation", aggregate=True),
    ConceptEntry("AmortizationOfIntangibleAssets",
                 "Amortization of Intangibles", aggregate=True),
]


# ═══════════════════════════════════════════════════════════════════════════
#  BALANCE SHEET — Core
# ═══════════════════════════════════════════════════════════════════════════

TOTAL_ASSETS: list[ConceptEntry] = [
    ConceptEntry("Assets", "Total Assets"),
    ConceptEntry("AssetsNet", "Net Assets"),
    ConceptEntry("TotalAssets", "Total Assets (alt tag)"),
    # IFRS
    ConceptEntry("NoncurrentAssets", "Noncurrent Assets"),
    # Insurance / bank
    ConceptEntry("InvestmentsAndCash", "Investments and Cash"),
]

CURRENT_ASSETS: list[ConceptEntry] = [
    ConceptEntry("AssetsCurrent", "Current Assets"),
    ConceptEntry("CurrentAssets", "IFRS Current Assets"),
    ConceptEntry("TotalCurrentAssets", "Total Current Assets"),
]

TOTAL_LIABILITIES: list[ConceptEntry] = [
    ConceptEntry("Liabilities", "Total Liabilities"),
    ConceptEntry("LiabilitiesAndStockholdersEquity", "Liabilities + Equity (for derivation)"),
    ConceptEntry("TotalLiabilities", "Total Liabilities (alt tag)"),
    # IFRS
    ConceptEntry("NoncurrentLiabilities", "Noncurrent Liabilities"),
]

CURRENT_LIABILITIES: list[ConceptEntry] = [
    ConceptEntry("LiabilitiesCurrent", "Current Liabilities"),
    ConceptEntry("CurrentLiabilities", "IFRS Current Liabilities"),
    ConceptEntry("TotalCurrentLiabilities", "Total Current Liabilities"),
]

STOCKHOLDERS_EQUITY: list[ConceptEntry] = [
    ConceptEntry("StockholdersEquity", "Stockholders' Equity"),
    ConceptEntry("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                 "Total Equity (incl. NCI)"),
    ConceptEntry("Equity", "Equity"),
    ConceptEntry("CommonStockholdersEquity", "Common Stockholders' Equity"),
    ConceptEntry("MembersEquity", "Members' Equity (partnerships)"),
    ConceptEntry("PartnersCapital", "Partners' Capital"),
    ConceptEntry("TotalEquity", "Total Equity"),
    ConceptEntry("TotalStockholdersEquity", "Total Stockholders' Equity"),
    # IFRS
    ConceptEntry("EquityAttributableToOwnersOfParent", "IFRS Equity Attributable to Parent"),
    ConceptEntry("EquityAttributableToParent", "IFRS Equity to Parent (alt)"),
]

TOTAL_EQUITY: list[ConceptEntry] = [
    ConceptEntry("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                 "Total Equity (incl. NCI)"),
    ConceptEntry("Equity", "Equity"),
    ConceptEntry("StockholdersEquity", "Stockholders' Equity"),
    ConceptEntry("LiabilitiesAndStockholdersEquity", "Liabilities & Stockholders' Equity"),
    # IFRS
    ConceptEntry("TotalEquity", "IFRS Total Equity"),
]

LONG_TERM_DEBT: list[ConceptEntry] = [
    ConceptEntry("LongTermDebt", "Long-Term Debt"),
    ConceptEntry("LongTermDebtNoncurrent", "Long-Term Debt (Noncurrent)"),
    ConceptEntry("LongTermDebtAndCapitalLeaseObligations",
                 "Long-Term Debt & Capital Lease"),
    # IFRS
    ConceptEntry("NoncurrentPortionOfNoncurrentBorrowings", "IFRS Long-Term Borrowings"),
    ConceptEntry("LongTermBorrowings", "IFRS Long-Term Borrowings (alt)"),
]

SHORT_TERM_DEBT: list[ConceptEntry] = [
    ConceptEntry("ShortTermBorrowings", "Short-Term Borrowings"),
    ConceptEntry("DebtCurrent", "Current Debt"),
    ConceptEntry("CommercialPaper", "Commercial Paper"),
    # IFRS
    ConceptEntry("CurrentPortionOfNoncurrentBorrowings", "IFRS Current Borrowings"),
    ConceptEntry("CurrentBorrowings", "IFRS Current Borrowings (alt)"),
]

CASH_AND_EQUIVALENTS: list[ConceptEntry] = [
    ConceptEntry("CashAndCashEquivalentsAtCarryingValue",
                 "Cash and Cash Equivalents"),
    ConceptEntry("CashCashEquivalentsAndShortTermInvestments",
                 "Cash, Equivalents & Short-Term Investments"),
    ConceptEntry("Cash", "Cash"),
    # IFRS
    ConceptEntry("CashAndCashEquivalents", "IFRS Cash and Equivalents"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  BALANCE SHEET — Extended
# ═══════════════════════════════════════════════════════════════════════════

ACCOUNTS_RECEIVABLE: list[ConceptEntry] = [
    ConceptEntry("AccountsReceivableNetCurrent", "Accounts Receivable, Net (Current)"),
    ConceptEntry("AccountsReceivableNet", "Accounts Receivable, Net"),
    ConceptEntry("ReceivablesNetCurrent", "Receivables, Net (Current)"),
    ConceptEntry("TradeAndOtherCurrentReceivables", "Trade & Other Receivables (Current)"),
    # IFRS
    ConceptEntry("TradeAndOtherReceivables", "IFRS Trade Receivables"),
]

INVENTORY: list[ConceptEntry] = [
    ConceptEntry("InventoryNet", "Inventory, Net"),
    ConceptEntry("InventoryFinishedGoods", "Finished Goods Inventory"),
    ConceptEntry("InventoryWorkInProcess", "Work-in-Process Inventory"),
    ConceptEntry("InventoryRawMaterials", "Raw Materials Inventory"),
    ConceptEntry("Inventories", "Inventories"),
    # IFRS
    ConceptEntry("CurrentInventories", "IFRS Current Inventories"),
]

GOODWILL: list[ConceptEntry] = [
    ConceptEntry("Goodwill", "Goodwill"),
    ConceptEntry("GoodwillAndIntangibleAssetsNet", "Goodwill & Intangible Assets, Net"),
]

INTANGIBLE_ASSETS: list[ConceptEntry] = [
    ConceptEntry("IntangibleAssetsNetExcludingGoodwill",
                 "Intangible Assets, Net (excl. Goodwill)"),
    ConceptEntry("IntangibleAssetsNetIncludingGoodwill",
                 "Intangible Assets, Net (incl. Goodwill)"),
    ConceptEntry("FiniteLivedIntangibleAssetsNet", "Finite-Lived Intangible Assets, Net"),
    ConceptEntry("IndefiniteLivedIntangibleAssetsExcludingGoodwill",
                 "Indefinite-Lived Intangible Assets"),
]

PROPERTY_PLANT_EQUIPMENT: list[ConceptEntry] = [
    ConceptEntry("PropertyPlantAndEquipmentNet", "PP&E, Net"),
    ConceptEntry("PropertyPlantAndEquipmentGross", "PP&E, Gross"),
    ConceptEntry("PropertyPlantAndEquipmentOther", "PP&E, Other"),
    # IFRS
    ConceptEntry("PropertyPlantAndEquipment", "IFRS PP&E"),
]

ACCOUNTS_PAYABLE: list[ConceptEntry] = [
    ConceptEntry("AccountsPayableCurrent", "Accounts Payable (Current)"),
    ConceptEntry("AccountsPayable", "Accounts Payable"),
    ConceptEntry("AccountsPayableAndAccruedLiabilitiesCurrent",
                 "Accounts Payable & Accrued Liabilities (Current)"),
    # IFRS
    ConceptEntry("TradeAndOtherCurrentPayables", "IFRS Trade Payables (Current)"),
]

ACCRUED_LIABILITIES: list[ConceptEntry] = [
    ConceptEntry("AccruedLiabilitiesCurrent", "Accrued Liabilities (Current)"),
    ConceptEntry("AccruedLiabilities", "Accrued Liabilities"),
    ConceptEntry("OtherAccruedLiabilitiesCurrent", "Other Accrued Liabilities (Current)"),
    ConceptEntry("AccruedIncomeTaxesCurrent", "Accrued Income Taxes (Current)"),
    ConceptEntry("EmployeeRelatedLiabilitiesCurrent", "Employee Related Liabilities (Current)"),
]

DEFERRED_REVENUE_BS: list[ConceptEntry] = [
    ConceptEntry("DeferredRevenueCurrent", "Deferred Revenue (Current)"),
    ConceptEntry("DeferredRevenueNoncurrent", "Deferred Revenue (Noncurrent)"),
    ConceptEntry("DeferredRevenue", "Deferred Revenue"),
    ConceptEntry("ContractWithCustomerLiabilityCurrent",
                 "Contract Liability (Current)"),
    ConceptEntry("ContractWithCustomerLiabilityNoncurrent",
                 "Contract Liability (Noncurrent)"),
    ConceptEntry("ContractWithCustomerLiability", "Contract Liability"),
    ConceptEntry("UnearnedRevenue", "Unearned Revenue"),
]

RETAINED_EARNINGS: list[ConceptEntry] = [
    ConceptEntry("RetainedEarningsAccumulatedDeficit",
                 "Retained Earnings (Accumulated Deficit)"),
    ConceptEntry("RetainedEarnings", "Retained Earnings"),
    ConceptEntry("AccumulatedUndistributedEarnings", "Accumulated Undistributed Earnings"),
]

ACCUMULATED_OTHER_COMPREHENSIVE_INCOME: list[ConceptEntry] = [
    ConceptEntry("AccumulatedOtherComprehensiveIncomeLossNetOfTax",
                 "AOCI, Net of Tax"),
    ConceptEntry("AccumulatedOtherComprehensiveIncomeLoss",
                 "Accumulated Other Comprehensive Income (Loss)"),
    ConceptEntry("AccumulatedOtherComprehensiveIncomeLossForeignCurrencyTranslationAdjustmentNetOfTax",
                 "AOCI — Foreign Currency Translation"),
    ConceptEntry("AccumulatedOtherComprehensiveIncomeLossAvailableForSaleSecuritiesAdjustmentNetOfTax",
                 "AOCI — AFS Securities"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  CASH FLOW — Core
# ═══════════════════════════════════════════════════════════════════════════

OPERATING_CASH_FLOW: list[ConceptEntry] = [
    ConceptEntry("NetCashProvidedByUsedInOperatingActivities",
                 "Operating Cash Flow"),
    ConceptEntry("NetCashProvidedByOperatingActivities",
                 "Operating Cash Flow (alt)"),
    ConceptEntry("CashFlowsFromUsedInOperatingActivities",
                 "Cash Flows from Operations"),
    # IFRS
    ConceptEntry("CashFlowsFromUsedInOperations", "IFRS Operating Cash Flow"),
]

CAPITAL_EXPENDITURES: list[ConceptEntry] = [
    ConceptEntry("PaymentsToAcquirePropertyPlantAndEquipment",
                 "Capital Expenditures"),
    ConceptEntry("AdditionsToPropertyPlantAndEquipment",
                 "Capital Expenditures (alt)"),
    ConceptEntry("CapitalExpenditures", "Capital Expenditures (generic)"),
    ConceptEntry("PaymentsToAcquireProductiveAssets",
                 "Payments to Acquire Assets"),
    # IFRS
    ConceptEntry("PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
                 "IFRS CapEx"),
]

INVESTING_CASH_FLOW: list[ConceptEntry] = [
    ConceptEntry("NetCashProvidedByUsedInInvestingActivities",
                 "Investing Cash Flow"),
    # IFRS
    ConceptEntry("CashFlowsFromUsedInInvestingActivities",
                 "IFRS Investing Cash Flow"),
]

FINANCING_CASH_FLOW: list[ConceptEntry] = [
    ConceptEntry("NetCashProvidedByUsedInFinancingActivities",
                 "Financing Cash Flow"),
    # IFRS
    ConceptEntry("CashFlowsFromUsedInFinancingActivities",
                 "IFRS Financing Cash Flow"),
]

DIVIDENDS_PAID: list[ConceptEntry] = [
    ConceptEntry("PaymentsOfDividends", "Dividends Paid"),
    ConceptEntry("PaymentsOfDividendsCommonStock", "Dividends Paid (Common)"),
    ConceptEntry("PaymentsOfOrdinaryDividends", "Ordinary Dividends Paid"),
    # IFRS
    ConceptEntry("DividendsPaidClassifiedAsFinancingActivities", "IFRS Dividends Paid"),
]

SHARES_REPURCHASED: list[ConceptEntry] = [
    ConceptEntry("PaymentsForRepurchaseOfCommonStock", "Share Repurchases"),
    ConceptEntry("PaymentsForRepurchaseOfEquity", "Equity Repurchases"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  CASH FLOW — Extended
# ═══════════════════════════════════════════════════════════════════════════

SBC_CASH_FLOW: list[ConceptEntry] = [
    ConceptEntry("ShareBasedCompensation", "Stock-Based Compensation (CF add-back)"),
    ConceptEntry("AllocatedShareBasedCompensationExpense",
                 "Allocated SBC (CF add-back)"),
]

ACQUISITIONS: list[ConceptEntry] = [
    ConceptEntry("PaymentsToAcquireBusinessesNetOfCashAcquired",
                 "Acquisitions, Net of Cash"),
    ConceptEntry("PaymentsToAcquireBusinessesGross", "Acquisitions, Gross"),
    ConceptEntry("BusinessCombinationConsiderationTransferred1",
                 "Business Combination Consideration"),
    ConceptEntry("PaymentsToAcquireInterestInSubsidiariesAndAffiliates",
                 "Payments for Subsidiaries/Affiliates"),
]

PROCEEDS_FROM_DEBT: list[ConceptEntry] = [
    ConceptEntry("ProceedsFromIssuanceOfLongTermDebt",
                 "Proceeds from Long-Term Debt Issuance"),
    ConceptEntry("ProceedsFromIssuanceOfDebt", "Proceeds from Debt Issuance"),
    ConceptEntry("ProceedsFromLongTermLinesOfCredit",
                 "Proceeds from Lines of Credit"),
    ConceptEntry("ProceedsFromIssuanceOfSeniorLongTermDebt",
                 "Proceeds from Senior Debt"),
    ConceptEntry("ProceedsFromShortTermDebt", "Proceeds from Short-Term Debt"),
    ConceptEntry("ProceedsFromBorrowings", "IFRS Proceeds from Borrowings"),
]

REPAYMENT_OF_DEBT: list[ConceptEntry] = [
    ConceptEntry("RepaymentsOfLongTermDebt", "Repayment of Long-Term Debt"),
    ConceptEntry("RepaymentsOfDebt", "Repayment of Debt"),
    ConceptEntry("RepaymentsOfLongTermLinesOfCredit", "Repayment of Lines of Credit"),
    ConceptEntry("RepaymentsOfShortTermDebt", "Repayment of Short-Term Debt"),
    ConceptEntry("RepaymentOfDebt", "Repayment of Debt (alt)"),
    ConceptEntry("RepaymentsOfBorrowings", "IFRS Repayments of Borrowings"),
]

FREE_CASH_FLOW_COMPONENTS: list[ConceptEntry] = [
    # FCF = Operating CF - CapEx; these are the building blocks
    ConceptEntry("NetCashProvidedByUsedInOperatingActivities",
                 "Operating Cash Flow"),
    ConceptEntry("PaymentsToAcquirePropertyPlantAndEquipment",
                 "Capital Expenditures"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  PER-SHARE
# ═══════════════════════════════════════════════════════════════════════════

EPS_BASIC: list[ConceptEntry] = [
    ConceptEntry("EarningsPerShareBasic", "EPS (Basic)"),
    # IFRS
    ConceptEntry("BasicEarningsLossPerShare", "IFRS EPS (Basic)"),
]

EPS_DILUTED: list[ConceptEntry] = [
    ConceptEntry("EarningsPerShareDiluted", "EPS (Diluted)"),
    # IFRS
    ConceptEntry("DilutedEarningsLossPerShare", "IFRS EPS (Diluted)"),
]

SHARES_OUTSTANDING: list[ConceptEntry] = [
    ConceptEntry("CommonStockSharesOutstanding", "Common Shares Outstanding"),
    ConceptEntry("WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                 "Weighted Avg Shares"),
    ConceptEntry("EntityCommonStockSharesOutstanding",
                 "Entity Common Shares Outstanding"),
]

BOOK_VALUE_PER_SHARE: list[ConceptEntry] = [
    ConceptEntry("BookValuePerShareOfCommonStock", "Book Value Per Share"),
    ConceptEntry("TangibleBookValuePerShare", "Tangible Book Value Per Share"),
    ConceptEntry("BookValuePerShare", "Book Value Per Share (alt)"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDUSTRY-SPECIFIC: REIT
#  FFO, NOI, Same-Store NOI, Occupancy
# ═══════════════════════════════════════════════════════════════════════════

REIT_FFO: list[ConceptEntry] = [
    # NAREIT-defined FFO tags (custom extensions — many REITs file these)
    ConceptEntry("FundsFromOperations", "Funds From Operations"),
    ConceptEntry("NareitFundsFromOperationsPerShare", "NAREIT FFO Per Share"),
    ConceptEntry("FundsFromOperationsPerShare", "FFO Per Share"),
    ConceptEntry("FundsFromOperationsPerShareDiluted", "FFO Per Share (Diluted)"),
    ConceptEntry("AdjustedFundsFromOperations", "Adjusted FFO"),
    ConceptEntry("AdjustedFundsFromOperationsPerShare", "AFFO Per Share"),
    ConceptEntry("AdjustedFundsFromOperationsPerShareDiluted", "AFFO Per Share (Diluted)"),
    ConceptEntry("NareitFFO", "NAREIT FFO"),
    ConceptEntry("CoreFundsFromOperations", "Core FFO"),
    ConceptEntry("CoreFFOPerDilutedShare", "Core FFO Per Diluted Share"),
    # Building blocks for manual FFO calc: Net Income + D&A - Gains on sale
    ConceptEntry("GainLossOnSaleOfProperties", "Gain/Loss on Sale of Properties"),
    ConceptEntry("DepreciationOfRealEstateAssets", "Depreciation of Real Estate"),
    ConceptEntry("RealEstateDepreciation", "Real Estate Depreciation"),
]

REIT_NOI: list[ConceptEntry] = [
    ConceptEntry("NetOperatingIncome", "Net Operating Income"),
    ConceptEntry("NOI", "NOI"),
    ConceptEntry("PropertyNetOperatingIncome", "Property NOI"),
    ConceptEntry("SameStoreNetOperatingIncome", "Same-Store NOI"),
    ConceptEntry("SameStoreNOI", "Same-Store NOI (alt)"),
    ConceptEntry("SameStorePropertyNetOperatingIncome", "Same-Store Property NOI"),
    ConceptEntry("ComparableStoreNetOperatingIncome", "Comparable Store NOI"),
]

REIT_OCCUPANCY: list[ConceptEntry] = [
    ConceptEntry("OccupancyRate", "Occupancy Rate"),
    ConceptEntry("PercentageOccupied", "Percentage Occupied"),
    ConceptEntry("OccupancyPercentage", "Occupancy Percentage"),
    ConceptEntry("PhysicalOccupancy", "Physical Occupancy"),
    ConceptEntry("EconomicOccupancy", "Economic Occupancy"),
    ConceptEntry("SameStoreOccupancy", "Same-Store Occupancy"),
    ConceptEntry("PortfolioOccupancy", "Portfolio Occupancy"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDUSTRY-SPECIFIC: INSURANCE
#  Loss Ratios, Combined Ratio, Claims
# ═══════════════════════════════════════════════════════════════════════════

INSURANCE_LOSSES: list[ConceptEntry] = [
    ConceptEntry("PolicyholderBenefitsAndClaimsIncurredNet",
                 "Net Policyholder Benefits & Claims"),
    ConceptEntry("PolicyholderBenefitsAndClaimsIncurredGross",
                 "Gross Policyholder Benefits & Claims"),
    ConceptEntry("InsuranceLossesAndSettlementExpense",
                 "Insurance Losses & Settlement Expense"),
    ConceptEntry("LossesIncurredNet", "Net Losses Incurred"),
    ConceptEntry("LossesAndLossAdjustmentExpense", "Losses & LAE"),
    ConceptEntry("BenefitsLossesAndExpenses", "Benefits, Losses & Expenses"),
    ConceptEntry("IncurredClaimsPropertyCasualtyAndLiability",
                 "Incurred Claims (P&C)"),
    ConceptEntry("LiabilityForFuturePolicyBenefits", "Liability for Future Benefits"),
    ConceptEntry("LossAndLossAdjustmentExpenseReserve", "Loss & LAE Reserve"),
]

INSURANCE_COMBINED_RATIO: list[ConceptEntry] = [
    # Combined ratio = Loss ratio + Expense ratio (usually custom extension)
    ConceptEntry("CombinedRatio", "Combined Ratio"),
    ConceptEntry("CombinedRatioPercentage", "Combined Ratio (%)"),
    ConceptEntry("LossRatio", "Loss Ratio"),
    ConceptEntry("LossRatioPercentage", "Loss Ratio (%)"),
    ConceptEntry("ExpenseRatio", "Expense Ratio"),
    ConceptEntry("UnderwritingExpenseRatio", "Underwriting Expense Ratio"),
    ConceptEntry("PolicyAcquisitionCostRatio", "Policy Acquisition Cost Ratio"),
    ConceptEntry("PolicyAcquisitionExpense", "Policy Acquisition Expense"),
    ConceptEntry("DeferredPolicyAcquisitionCostAmortizationExpense",
                 "Deferred Acquisition Cost Amortization"),
]

INSURANCE_UNDERWRITING: list[ConceptEntry] = [
    ConceptEntry("UnderwritingIncomeLoss", "Underwriting Income (Loss)"),
    ConceptEntry("NetUnderwritingIncome", "Net Underwriting Income"),
    ConceptEntry("UnderwritingExpenses", "Underwriting Expenses"),
    ConceptEntry("PremiumsWrittenGross", "Gross Premiums Written"),
    ConceptEntry("PremiumsWrittenNet", "Net Premiums Written"),
    ConceptEntry("PremiumsCeded", "Premiums Ceded"),
    ConceptEntry("InsuranceCommissionsAndFees", "Insurance Commissions & Fees"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDUSTRY-SPECIFIC: BANKING — Extended
#  Net Charge-Offs, Efficiency Ratio, Capital Ratios, NIM
# ═══════════════════════════════════════════════════════════════════════════

BANK_CHARGE_OFFS: list[ConceptEntry] = [
    ConceptEntry("ChargeOffsNet", "Net Charge-Offs"),
    ConceptEntry("AllowanceForLoanLossesChargedOff", "Loan Losses Charged Off"),
    ConceptEntry("NetChargeOffs", "Net Charge-Offs (alt)"),
    ConceptEntry("ChargeOffsGross", "Gross Charge-Offs"),
    ConceptEntry("Recoveries", "Recoveries"),
    ConceptEntry("AllowanceForCreditLosses", "Allowance for Credit Losses"),
    ConceptEntry("AllowanceForLoanAndLeaseLosses", "Allowance for Loan & Lease Losses"),
    ConceptEntry("FinancingReceivableAllowanceForCreditLosses",
                 "Financing Receivable ACL"),
]

BANK_EFFICIENCY: list[ConceptEntry] = [
    ConceptEntry("EfficiencyRatio", "Efficiency Ratio"),
    ConceptEntry("NoninterestExpense", "Noninterest Expense"),
    ConceptEntry("NoninterestExpenseToAverageAssetsRatio",
                 "Noninterest Expense / Avg Assets"),
    ConceptEntry("OccupancyNetExpense", "Occupancy Expense"),
    ConceptEntry("TechnologyAndCommunicationsExpense", "Technology & Communications"),
    ConceptEntry("CompensationAndBenefitsExpense", "Compensation & Benefits"),
    ConceptEntry("ProfessionalFees", "Professional Fees"),
]

BANK_CAPITAL_RATIOS: list[ConceptEntry] = [
    ConceptEntry("Tier1CapitalRatio", "Tier 1 Capital Ratio"),
    ConceptEntry("Tier1Capital", "Tier 1 Capital"),
    ConceptEntry("CommonEquityTier1CapitalRatio", "CET1 Ratio"),
    ConceptEntry("CommonEquityTier1Capital", "CET1 Capital"),
    ConceptEntry("TotalCapitalRatio", "Total Capital Ratio"),
    ConceptEntry("TotalRiskBasedCapitalRatio", "Total Risk-Based Capital Ratio"),
    ConceptEntry("TotalRiskWeightedAssets", "Total Risk-Weighted Assets"),
    ConceptEntry("RiskWeightedAssets", "Risk-Weighted Assets"),
    ConceptEntry("LeverageRatio", "Leverage Ratio"),
    ConceptEntry("SupplementaryLeverageRatio", "Supplementary Leverage Ratio"),
    ConceptEntry("TangibleCommonEquityRatio", "Tangible Common Equity Ratio"),
    ConceptEntry("TangibleCommonEquity", "Tangible Common Equity"),
]

BANK_NET_INTEREST_MARGIN: list[ConceptEntry] = [
    ConceptEntry("NetInterestMargin", "Net Interest Margin"),
    ConceptEntry("NetInterestIncomeExpenseRatio", "Net Interest Income Ratio"),
    ConceptEntry("AverageInterestEarningAssets", "Avg Interest-Earning Assets"),
    ConceptEntry("NetInterestSpread", "Net Interest Spread"),
    ConceptEntry("NetInterestRateSpread", "Net Interest Rate Spread"),
]

BANK_DEPOSITS: list[ConceptEntry] = [
    ConceptEntry("Deposits", "Total Deposits"),
    ConceptEntry("InterestBearingDeposits", "Interest-Bearing Deposits"),
    ConceptEntry("NoninterestBearingDeposits", "Noninterest-Bearing Deposits"),
    ConceptEntry("TimeDepositsAtOrAboveFDICInsuranceLimit",
                 "Time Deposits Above FDIC Limit"),
    ConceptEntry("SavingsDeposits", "Savings Deposits"),
    ConceptEntry("DemandDeposits", "Demand Deposits"),
]

BANK_LOANS: list[ConceptEntry] = [
    ConceptEntry("LoansAndLeasesReceivableNetOfDeferredIncome",
                 "Net Loans & Leases"),
    ConceptEntry("LoansAndLeasesReceivableGrossCarryingAmount",
                 "Gross Loans & Leases"),
    ConceptEntry("FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
                 "Financing Receivable, Net of ACL"),
    ConceptEntry("LoansReceivableNet", "Loans Receivable, Net"),
    ConceptEntry("NonperformingLoans", "Nonperforming Loans"),
    ConceptEntry("NonperformingAssets", "Nonperforming Assets"),
    ConceptEntry("NonaccrualLoans", "Nonaccrual Loans"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDUSTRY-SPECIFIC: FINTECH / PAYMENTS
#  Transaction volume, processing, take rate
# ═══════════════════════════════════════════════════════════════════════════

FINTECH_TRANSACTION_VOLUME: list[ConceptEntry] = [
    ConceptEntry("PaymentVolume", "Payment Volume"),
    ConceptEntry("TotalPaymentVolume", "Total Payment Volume"),
    ConceptEntry("GrossPaymentVolume", "Gross Payment Volume"),
    ConceptEntry("TransactionVolume", "Transaction Volume"),
    ConceptEntry("TotalTransactions", "Total Transactions"),
    ConceptEntry("NumberOfTransactions", "Number of Transactions"),
    ConceptEntry("GrossMerchandiseVolume", "Gross Merchandise Volume"),
    ConceptEntry("ProcessedVolume", "Processed Volume"),
]

FINTECH_PROCESSING: list[ConceptEntry] = [
    ConceptEntry("TransactionAndProcessingFees", "Transaction & Processing Fees"),
    ConceptEntry("PaymentProcessingRevenue", "Payment Processing Revenue"),
    ConceptEntry("TransactionRevenue", "Transaction Revenue"),
    ConceptEntry("TransactionBasedRevenue", "Transaction-Based Revenue"),
    ConceptEntry("MerchantServicesRevenue", "Merchant Services Revenue"),
    ConceptEntry("ProcessingFees", "Processing Fees"),
    ConceptEntry("InterchangeFees", "Interchange Fees"),
]

FINTECH_TAKE_RATE: list[ConceptEntry] = [
    ConceptEntry("TakeRate", "Take Rate"),
    ConceptEntry("TransactionMargin", "Transaction Margin"),
    ConceptEntry("NetTransactionRevenue", "Net Transaction Revenue"),
    ConceptEntry("PaymentProcessingMargin", "Payment Processing Margin"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDUSTRY-SPECIFIC: SaaS
#  Deferred Revenue, RPO, Subscription Revenue
# ═══════════════════════════════════════════════════════════════════════════

SAAS_DEFERRED_REVENUE: list[ConceptEntry] = [
    ConceptEntry("DeferredRevenueCurrent", "Deferred Revenue (Current)"),
    ConceptEntry("DeferredRevenueNoncurrent", "Deferred Revenue (Noncurrent)"),
    ConceptEntry("DeferredRevenue", "Deferred Revenue"),
    ConceptEntry("ContractWithCustomerLiability", "Contract Liability"),
    ConceptEntry("ContractWithCustomerLiabilityCurrent",
                 "Contract Liability (Current)"),
    ConceptEntry("ContractWithCustomerLiabilityNoncurrent",
                 "Contract Liability (Noncurrent)"),
    ConceptEntry("UnearnedRevenue", "Unearned Revenue"),
]

SAAS_RPO: list[ConceptEntry] = [
    ConceptEntry("RemainingPerformanceObligation", "Remaining Performance Obligation"),
    ConceptEntry("RevenueRemainingPerformanceObligation",
                 "Revenue RPO"),
    ConceptEntry("RevenueRemainingPerformanceObligationExpectedTimingOfSatisfactionPeriod1",
                 "RPO Expected Timing"),
    ConceptEntry("ContractWithCustomerLiabilityRevenueRecognized",
                 "Revenue Recognized from Contract Liability"),
    ConceptEntry("RemainingPerformanceObligationCurrent", "RPO (Current)"),
    ConceptEntry("RemainingPerformanceObligationNoncurrent", "RPO (Noncurrent)"),
]

SAAS_SUBSCRIPTION_REVENUE: list[ConceptEntry] = [
    ConceptEntry("SubscriptionRevenue", "Subscription Revenue"),
    ConceptEntry("SaaSRevenue", "SaaS Revenue"),
    ConceptEntry("CloudServicesRevenue", "Cloud Services Revenue"),
    ConceptEntry("SubscriptionAndSupportRevenue", "Subscription & Support Revenue"),
    ConceptEntry("RecurringRevenue", "Recurring Revenue"),
    ConceptEntry("LicenseRevenue", "License Revenue"),
    ConceptEntry("MaintenanceRevenue", "Maintenance Revenue"),
    ConceptEntry("ProfessionalServicesRevenue", "Professional Services Revenue"),
]

SAAS_COST_OF_SUBSCRIPTION: list[ConceptEntry] = [
    ConceptEntry("CostOfSubscriptionRevenue", "Cost of Subscription Revenue"),
    ConceptEntry("CostOfCloudServices", "Cost of Cloud Services"),
    ConceptEntry("CostOfSaaSRevenue", "Cost of SaaS Revenue"),
    ConceptEntry("CostOfSubscriptionAndSupportRevenue",
                 "Cost of Subscription & Support Revenue"),
    ConceptEntry("CostOfServices", "Cost of Services"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDUSTRY-SPECIFIC: HEALTHCARE / PHARMA
#  Drug Revenue, R&D Pipeline, Milestone Payments
# ═══════════════════════════════════════════════════════════════════════════

HEALTHCARE_DRUG_REVENUE: list[ConceptEntry] = [
    ConceptEntry("ProductRevenue", "Product Revenue"),
    ConceptEntry("ProductRevenueNet", "Product Revenue, Net"),
    ConceptEntry("ProductSalesNet", "Product Sales, Net"),
    ConceptEntry("PharmaceuticalRevenue", "Pharmaceutical Revenue"),
    ConceptEntry("DrugRevenue", "Drug Revenue"),
    ConceptEntry("VaccineRevenue", "Vaccine Revenue"),
    ConceptEntry("BiologicsRevenue", "Biologics Revenue"),
    ConceptEntry("OncologyRevenue", "Oncology Revenue"),
    ConceptEntry("CollaborationRevenue", "Collaboration Revenue"),
    ConceptEntry("LicenseAndCollaborationRevenue", "License & Collaboration Revenue"),
    ConceptEntry("RoyaltyRevenue", "Royalty Revenue"),
    ConceptEntry("MilestoneRevenue", "Milestone Revenue"),
]

HEALTHCARE_RD_PIPELINE: list[ConceptEntry] = [
    ConceptEntry("ResearchAndDevelopmentExpense", "R&D Expense"),
    ConceptEntry("AcquiredInProcessResearchAndDevelopment",
                 "Acquired In-Process R&D"),
    ConceptEntry("UpfrontAndMilestonePaymentsReceived",
                 "Upfront & Milestone Payments Received"),
    ConceptEntry("CollaborationArrangementTransactionAmount",
                 "Collaboration Arrangement Amount"),
    ConceptEntry("LicenseAgreementTermsUpfrontPayment",
                 "License Upfront Payment"),
]

HEALTHCARE_MILESTONE_PAYMENTS: list[ConceptEntry] = [
    ConceptEntry("MilestonePaymentsReceived", "Milestone Payments Received"),
    ConceptEntry("MilestonePaymentsMade", "Milestone Payments Made"),
    ConceptEntry("ContingentConsiderationPayments", "Contingent Consideration Payments"),
    ConceptEntry("UpfrontPayments", "Upfront Payments"),
    ConceptEntry("LicensingMilestoneRevenue", "Licensing Milestone Revenue"),
    ConceptEntry("RegulatoryMilestonePayments", "Regulatory Milestone Payments"),
    ConceptEntry("CommercialMilestonePayments", "Commercial Milestone Payments"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDUSTRY-SPECIFIC: ENERGY / OIL & GAS
#  Production Revenue, Proved Reserves, Average Realized Price
# ═══════════════════════════════════════════════════════════════════════════

ENERGY_PRODUCTION_REVENUE: list[ConceptEntry] = [
    ConceptEntry("RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenue from Contract with Customer"),
    ConceptEntry("OilAndGasRevenue", "Oil & Gas Revenue"),
    ConceptEntry("OilRevenue", "Oil Revenue"),
    ConceptEntry("NaturalGasRevenue", "Natural Gas Revenue"),
    ConceptEntry("NaturalGasLiquidsRevenue", "NGL Revenue"),
    ConceptEntry("CrudeOilRevenue", "Crude Oil Revenue"),
    ConceptEntry("RefinedProductsRevenue", "Refined Products Revenue"),
    ConceptEntry("MidstreamRevenue", "Midstream Revenue"),
    ConceptEntry("UpstreamRevenue", "Upstream Revenue"),
    ConceptEntry("DownstreamRevenue", "Downstream Revenue"),
    ConceptEntry("ProductionRevenue", "Production Revenue"),
    ConceptEntry("GatheringAndProcessingRevenue", "Gathering & Processing Revenue"),
]

ENERGY_PROVED_RESERVES: list[ConceptEntry] = [
    ConceptEntry("ProvedDevelopedReservesOil", "Proved Developed Reserves (Oil)"),
    ConceptEntry("ProvedDevelopedReservesGas", "Proved Developed Reserves (Gas)"),
    ConceptEntry("ProvedUndevelopedReservesOil", "Proved Undeveloped Reserves (Oil)"),
    ConceptEntry("ProvedUndevelopedReservesGas", "Proved Undeveloped Reserves (Gas)"),
    ConceptEntry("ProvedReservesOil", "Proved Reserves (Oil)"),
    ConceptEntry("ProvedReservesGas", "Proved Reserves (Gas)"),
    ConceptEntry("ProvedReservesNaturalGasLiquids", "Proved Reserves (NGL)"),
    ConceptEntry("EstimatedProvedReservesBarrelOfOilEquivalent",
                 "Proved Reserves (BOE)"),
    ConceptEntry("ProvedReservesBeginningBalance", "Proved Reserves (Beginning)"),
    ConceptEntry("ProvedReservesEndingBalance", "Proved Reserves (Ending)"),
]

ENERGY_REALIZED_PRICE: list[ConceptEntry] = [
    ConceptEntry("AverageRealizedPricePerBarrelOfOil", "Avg Realized Price (Oil)"),
    ConceptEntry("AverageRealizedPricePerMcfOfGas", "Avg Realized Price (Gas)"),
    ConceptEntry("AverageRealizedPricePerBoe", "Avg Realized Price (BOE)"),
    ConceptEntry("OilProductionPerDay", "Oil Production Per Day"),
    ConceptEntry("GasProductionPerDay", "Gas Production Per Day"),
    ConceptEntry("TotalProductionPerDay", "Total Production Per Day"),
    ConceptEntry("AverageDailyProduction", "Average Daily Production"),
    ConceptEntry("TotalProduction", "Total Production"),
    ConceptEntry("OilAndCondensateProduction", "Oil & Condensate Production"),
    ConceptEntry("NaturalGasProduction", "Natural Gas Production"),
]

ENERGY_EXPLORATION: list[ConceptEntry] = [
    ConceptEntry("ExplorationExpense", "Exploration Expense"),
    ConceptEntry("ExplorationCosts", "Exploration Costs"),
    ConceptEntry("DryHoleCosts", "Dry Hole Costs"),
    ConceptEntry("OilAndGasPropertyFullCostMethodNet",
                 "O&G Property (Full Cost), Net"),
    ConceptEntry("OilAndGasPropertySuccessfulEffortMethodNet",
                 "O&G Property (Successful Efforts), Net"),
    ConceptEntry("DevelopmentCosts", "Development Costs"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDUSTRY-SPECIFIC: RETAIL
#  Same-Store Sales, Comparable Store
# ═══════════════════════════════════════════════════════════════════════════

RETAIL_SAME_STORE: list[ConceptEntry] = [
    ConceptEntry("SameStoreSales", "Same-Store Sales"),
    ConceptEntry("SameStoreSalesGrowth", "Same-Store Sales Growth"),
    ConceptEntry("ComparableStoreSales", "Comparable Store Sales"),
    ConceptEntry("ComparableStoreSalesGrowth", "Comparable Store Sales Growth"),
    ConceptEntry("IdenticalStoreSales", "Identical Store Sales"),
    ConceptEntry("ComparableSalesGrowth", "Comparable Sales Growth"),
    ConceptEntry("SameRestaurantSales", "Same-Restaurant Sales"),
    ConceptEntry("DomesticComparableStoreSales", "Domestic Comparable Store Sales"),
    ConceptEntry("InternationalComparableStoreSales", "International Comparable Store Sales"),
]

RETAIL_STORE_COUNT: list[ConceptEntry] = [
    ConceptEntry("NumberOfStores", "Number of Stores"),
    ConceptEntry("NumberOfStoresOpened", "Stores Opened"),
    ConceptEntry("NumberOfStoresClosed", "Stores Closed"),
    ConceptEntry("NumberOfRestaurants", "Number of Restaurants"),
    ConceptEntry("TotalUnits", "Total Units"),
    ConceptEntry("NetNewStores", "Net New Stores"),
    ConceptEntry("CompanyOperatedStores", "Company-Operated Stores"),
    ConceptEntry("FranchisedStores", "Franchised Stores"),
]

RETAIL_ECOMMERCE: list[ConceptEntry] = [
    ConceptEntry("DigitalRevenue", "Digital Revenue"),
    ConceptEntry("ECommerceRevenue", "E-Commerce Revenue"),
    ConceptEntry("OnlineSalesRevenue", "Online Sales Revenue"),
    ConceptEntry("DirectToConsumerRevenue", "Direct-to-Consumer Revenue"),
    ConceptEntry("DigitalSalesGrowth", "Digital Sales Growth"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER MAP (revenue handled separately via REVENUE_MAP)
# ═══════════════════════════════════════════════════════════════════════════

CONCEPT_MAP: dict[str, list[ConceptEntry]] = {
    # ── Income Statement — Core ──
    "net_income": NET_INCOME,
    "gross_profit": GROSS_PROFIT,
    "operating_income": OPERATING_INCOME,
    "cost_of_revenue": COST_OF_REVENUE,
    "operating_expenses": OPERATING_EXPENSES,
    "sga_expense": SGA_EXPENSE,
    "rd_expense": RD_EXPENSE,
    "interest_expense": INTEREST_EXPENSE,
    "income_tax_expense": INCOME_TAX_EXPENSE,
    "depreciation_amortization": DEPRECIATION_AMORTIZATION,
    "provision_for_credit_losses": PROVISION_FOR_CREDIT_LOSSES,
    # ── Income Statement — Extended ──
    "other_income_expense": OTHER_INCOME_EXPENSE,
    "income_before_tax": INCOME_BEFORE_TAX,
    "share_based_compensation": SHARE_BASED_COMPENSATION,
    # ── Balance Sheet — Core ──
    "total_assets": TOTAL_ASSETS,
    "current_assets": CURRENT_ASSETS,
    "total_liabilities": TOTAL_LIABILITIES,
    "current_liabilities": CURRENT_LIABILITIES,
    "stockholders_equity": STOCKHOLDERS_EQUITY,
    "total_equity": TOTAL_EQUITY,
    "long_term_debt": LONG_TERM_DEBT,
    "short_term_debt": SHORT_TERM_DEBT,
    "cash_and_equivalents": CASH_AND_EQUIVALENTS,
    # ── Balance Sheet — Extended ──
    "accounts_receivable": ACCOUNTS_RECEIVABLE,
    "inventory": INVENTORY,
    "goodwill": GOODWILL,
    "intangible_assets": INTANGIBLE_ASSETS,
    "property_plant_equipment": PROPERTY_PLANT_EQUIPMENT,
    "accounts_payable": ACCOUNTS_PAYABLE,
    "accrued_liabilities": ACCRUED_LIABILITIES,
    "deferred_revenue": DEFERRED_REVENUE_BS,
    "retained_earnings": RETAINED_EARNINGS,
    "accumulated_other_comprehensive_income": ACCUMULATED_OTHER_COMPREHENSIVE_INCOME,
    # ── Cash Flow — Core ──
    "operating_cash_flow": OPERATING_CASH_FLOW,
    "capital_expenditures": CAPITAL_EXPENDITURES,
    "investing_cash_flow": INVESTING_CASH_FLOW,
    "financing_cash_flow": FINANCING_CASH_FLOW,
    "dividends_paid": DIVIDENDS_PAID,
    "shares_repurchased": SHARES_REPURCHASED,
    # ── Cash Flow — Extended ──
    "sbc_cash_flow": SBC_CASH_FLOW,
    "acquisitions": ACQUISITIONS,
    "proceeds_from_debt": PROCEEDS_FROM_DEBT,
    "repayment_of_debt": REPAYMENT_OF_DEBT,
    "free_cash_flow_components": FREE_CASH_FLOW_COMPONENTS,
    # ── Per-Share ──
    "eps_basic": EPS_BASIC,
    "eps_diluted": EPS_DILUTED,
    "shares_outstanding": SHARES_OUTSTANDING,
    "book_value_per_share": BOOK_VALUE_PER_SHARE,
    # ── REIT ──
    "reit_ffo": REIT_FFO,
    "reit_noi": REIT_NOI,
    "reit_occupancy": REIT_OCCUPANCY,
    # ── Insurance ──
    "insurance_losses": INSURANCE_LOSSES,
    "insurance_combined_ratio": INSURANCE_COMBINED_RATIO,
    "insurance_underwriting": INSURANCE_UNDERWRITING,
    # ── Banking — Extended ──
    "bank_charge_offs": BANK_CHARGE_OFFS,
    "bank_efficiency": BANK_EFFICIENCY,
    "bank_capital_ratios": BANK_CAPITAL_RATIOS,
    "bank_net_interest_margin": BANK_NET_INTEREST_MARGIN,
    "bank_deposits": BANK_DEPOSITS,
    "bank_loans": BANK_LOANS,
    # ── Fintech / Payments ──
    "fintech_transaction_volume": FINTECH_TRANSACTION_VOLUME,
    "fintech_processing": FINTECH_PROCESSING,
    "fintech_take_rate": FINTECH_TAKE_RATE,
    # ── SaaS ──
    "saas_deferred_revenue": SAAS_DEFERRED_REVENUE,
    "saas_rpo": SAAS_RPO,
    "saas_subscription_revenue": SAAS_SUBSCRIPTION_REVENUE,
    "saas_cost_of_subscription": SAAS_COST_OF_SUBSCRIPTION,
    # ── Healthcare / Pharma ──
    "healthcare_drug_revenue": HEALTHCARE_DRUG_REVENUE,
    "healthcare_rd_pipeline": HEALTHCARE_RD_PIPELINE,
    "healthcare_milestone_payments": HEALTHCARE_MILESTONE_PAYMENTS,
    # ── Energy / Oil & Gas ──
    "energy_production_revenue": ENERGY_PRODUCTION_REVENUE,
    "energy_proved_reserves": ENERGY_PROVED_RESERVES,
    "energy_realized_price": ENERGY_REALIZED_PRICE,
    "energy_exploration": ENERGY_EXPLORATION,
    # ── Retail ──
    "retail_same_store": RETAIL_SAME_STORE,
    "retail_store_count": RETAIL_STORE_COUNT,
    "retail_ecommerce": RETAIL_ECOMMERCE,
}


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM / EXTENSION PATTERN MATCHING
# ═══════════════════════════════════════════════════════════════════════════

# Negative lookahead prevents matching guidance/forecast concepts:
#   e.g., "EstimatedRevenues", "GuidanceRevenue", "ProjectedTotalRevenue"
_FORECAST_EXCLUSION = r"(?!.*(?:Guidance|Forecast|Estimated|Expected|Projected|Budget|Plan|Target|Outlook))"

_CUSTOM_NET_REVENUE_RE = re.compile(
    rf"{_FORECAST_EXCLUSION}(?:NetRevenue|TotalNetRevenue|_NetRevenues$)",
    re.IGNORECASE,
)
_CUSTOM_REVENUE_RE = re.compile(
    rf"{_FORECAST_EXCLUSION}(?:_Revenues$|_Revenue$|_TotalRevenue$|_TotalRevenues$)",
    re.IGNORECASE,
)


def is_custom_net_revenue(concept: str) -> bool:
    return bool(_CUSTOM_NET_REVENUE_RE.search(concept))


def is_custom_revenue(concept: str) -> bool:
    return bool(_CUSTOM_REVENUE_RE.search(concept))


# ═══════════════════════════════════════════════════════════════════════════
#  ROW QUALITY FILTERS
# ═══════════════════════════════════════════════════════════════════════════

DIMENSION_COLUMNS = [
    "dimension", "dimensions", "segment", "segments", "member",
    "axis", "Axis", "Dimension", "Dimensions",
]
ABSTRACT_COLUMNS = ["abstract", "is_abstract", "isAbstract", "Abstract"]
LEVEL_COLUMNS = ["level", "Level", "depth", "Depth"]
MAX_ACCEPTABLE_LEVEL = 2
