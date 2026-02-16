"""Production-grade XBRL concept → canonical metric mappings.

Three-layer architecture:
  Layer 1 — Canonical schema  (metric key → display name)
  Layer 2 — Synonym mapping   (each canonical maps to ordered XBRL tags)
  Layer 3 — LLM disambiguation (called only when ambiguous — see financials.py)

Industry-specific mappings:
  - Standard corporates (tech, consumer, industrial, healthcare, SaaS)
  - Banks & broker-dealers (MS, GS, JPM)
  - Insurance (premiums + investment income)
  - REITs (lease revenue)
  - Utilities (electric + gas)
  - Crypto / fintech (COIN, MSTR, MARA — standard GAAP + crypto-aware)
  - Custom extension filers (ms_NetRevenues, gs_TotalNetRevenues, etc.)

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
    # Crypto / digital assets
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
#  BALANCE SHEET
# ═══════════════════════════════════════════════════════════════════════════

TOTAL_ASSETS: list[ConceptEntry] = [
    ConceptEntry("Assets", "Total Assets"),
    # IFRS — same concept name, but listed for clarity
    ConceptEntry("NoncurrentAssets", "Noncurrent Assets"),
]

CURRENT_ASSETS: list[ConceptEntry] = [
    ConceptEntry("AssetsCurrent", "Current Assets"),
    ConceptEntry("CurrentAssets", "IFRS Current Assets"),
]

TOTAL_LIABILITIES: list[ConceptEntry] = [
    ConceptEntry("Liabilities", "Total Liabilities"),
    ConceptEntry("NoncurrentLiabilities", "Noncurrent Liabilities"),
]

CURRENT_LIABILITIES: list[ConceptEntry] = [
    ConceptEntry("LiabilitiesCurrent", "Current Liabilities"),
    ConceptEntry("CurrentLiabilities", "IFRS Current Liabilities"),
]

STOCKHOLDERS_EQUITY: list[ConceptEntry] = [
    ConceptEntry("StockholdersEquity", "Stockholders' Equity"),
    ConceptEntry("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                 "Total Equity (incl. NCI)"),
    ConceptEntry("Equity", "Equity"),
    ConceptEntry("CommonStockholdersEquity", "Common Stockholders' Equity"),
    # IFRS
    ConceptEntry("EquityAttributableToOwnersOfParent", "IFRS Equity Attributable to Parent"),
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
#  CASH FLOW
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

# ═══════════════════════════════════════════════════════════════════════════
#  MASTER MAP (revenue handled separately via REVENUE_MAP)
# ═══════════════════════════════════════════════════════════════════════════

CONCEPT_MAP: dict[str, list[ConceptEntry]] = {
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
    "total_assets": TOTAL_ASSETS,
    "current_assets": CURRENT_ASSETS,
    "total_liabilities": TOTAL_LIABILITIES,
    "current_liabilities": CURRENT_LIABILITIES,
    "stockholders_equity": STOCKHOLDERS_EQUITY,
    "long_term_debt": LONG_TERM_DEBT,
    "short_term_debt": SHORT_TERM_DEBT,
    "cash_and_equivalents": CASH_AND_EQUIVALENTS,
    "operating_cash_flow": OPERATING_CASH_FLOW,
    "capital_expenditures": CAPITAL_EXPENDITURES,
    "investing_cash_flow": INVESTING_CASH_FLOW,
    "financing_cash_flow": FINANCING_CASH_FLOW,
    "dividends_paid": DIVIDENDS_PAID,
    "shares_repurchased": SHARES_REPURCHASED,
    "eps_basic": EPS_BASIC,
    "eps_diluted": EPS_DILUTED,
    "shares_outstanding": SHARES_OUTSTANDING,
}


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM / EXTENSION PATTERN MATCHING
# ═══════════════════════════════════════════════════════════════════════════

_CUSTOM_NET_REVENUE_RE = re.compile(
    r"(?:NetRevenue|TotalNetRevenue|_NetRevenues$)", re.IGNORECASE
)
_CUSTOM_REVENUE_RE = re.compile(
    r"(?:_Revenues$|_Revenue$|_TotalRevenue$|_TotalRevenues$)", re.IGNORECASE
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
