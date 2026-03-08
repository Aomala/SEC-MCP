"""Dynamic peer discovery engine — finds comparable companies by industry, SIC, or custom lists.

Combines three strategies:
  1. Curated PEER_MAP (50+ industry groups covering Tech, Finance, Energy, etc.)
  2. Same SIC code lookup via SEC tickers list
  3. User-supplied custom peer lists
  
Ranks peers by relevance score and deduplicates.
Integrates with existing extract_financials_batch() for peer comparisons.
"""

from __future__ import annotations

import logging
from typing import Any

# Import existing financials extraction for batch comparisons
from sec_mcp.financials import extract_financials_batch

# Module logger for warnings and errors
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Peer Map — Curated Industry Groups
# ═══════════════════════════════════════════════════════════════════════════

# PEER_MAP defines established peer groups across ~50 industry segments
# Each key is a ticker, value is a list of up to 5 comparable company tickers
# This map is sourced from chat_app.py and represents industry consensus groupings
# Covers: Big Tech, Semiconductors, Enterprise Software, Cybersecurity, Cloud/SaaS,
# E-commerce/Retail, Auto, Streaming/Media, Banks, Payments, Pharma, Healthcare,
# Energy, Telecom, Consumer Staples, Restaurants, Airlines, REITs, Industrials,
# Mining/Materials, Crypto, and Music/Audio sectors

PEER_MAP: dict[str, list[str]] = {
    # ─────────────────────────────────────────────────────────────────────
    # Big Tech — Consumer and enterprise technology platforms
    # ─────────────────────────────────────────────────────────────────────
    "AAPL": ["MSFT", "GOOG", "AMZN", "META"],  # Apple: competitors in devices, services, cloud
    "MSFT": ["AAPL", "GOOG", "AMZN", "ORCL"],  # Microsoft: cloud, productivity, gaming
    "GOOG": ["META", "MSFT", "AMZN", "SNAP"],  # Google: search, ads, cloud, hardware
    "GOOGL": ["META", "MSFT", "AMZN", "SNAP"],  # Google (voting shares): same as GOOG
    "META": ["GOOG", "SNAP", "PINS", "NFLX"],  # Meta: social, ads, VR, streaming
    "AMZN": ["MSFT", "AAPL", "GOOG", "WMT"],  # Amazon: e-commerce, cloud, advertising
    
    # ─────────────────────────────────────────────────────────────────────
    # Semiconductors — Chip design and manufacturing
    # ─────────────────────────────────────────────────────────────────────
    "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],  # Nvidia: GPUs, AI chips, data center
    "AMD": ["NVDA", "INTC", "QCOM", "AVGO"],  # AMD: processors, GPUs, data center
    "INTC": ["AMD", "NVDA", "QCOM", "TSM"],  # Intel: CPUs, process tech, foundry
    "QCOM": ["NVDA", "AMD", "INTC", "AVGO"],  # Qualcomm: mobile, automotive, RF
    "AVGO": ["NVDA", "AMD", "QCOM", "INTC"],  # Broadcom: chips, infrastructure
    "TSM": ["INTC", "NVDA", "AMAT", "LRCX"],  # TSMC: foundry, process leadership
    "ASML": ["AMAT", "LRCX", "KLAC", "TER"],  # ASML: lithography, semiconductor equipment
    "AMAT": ["ASML", "LRCX", "KLAC", "NVDA"],  # Applied Materials: fab equipment
    "LRCX": ["AMAT", "ASML", "KLAC", "TER"],  # Lam Research: process equipment
    
    # ─────────────────────────────────────────────────────────────────────
    # Enterprise Software — Business applications and platforms
    # ─────────────────────────────────────────────────────────────────────
    "ORCL": ["SAP", "MSFT", "IBM", "CRM"],  # Oracle: databases, ERP, cloud
    "SAP": ["ORCL", "MSFT", "IBM", "CRM"],  # SAP: enterprise software, cloud
    "CRM": ["MSFT", "ORCL", "SAP", "NOW"],  # Salesforce: CRM, analytics, cloud
    "NOW": ["CRM", "MSFT", "ORCL", "SAP"],  # ServiceNow: IT operations, workflows
    "IBM": ["ORCL", "MSFT", "HPE", "ACN"],  # IBM: cloud, enterprise, consulting
    "CSCO": ["JNPR", "PANW", "FTNT", "HPE"],  # Cisco: networking, security, cloud
    "ADBE": ["CRM", "MSFT", "NOW", "WDAY"],  # Adobe: creative, document, analytics
    "WDAY": ["CRM", "ADBE", "SAP", "ORCL"],  # Workday: HR, finance cloud
    "INTU": ["MSFT", "ADBE", "CRM", "NOW"],  # Intuit: financial software, small business
    
    # ─────────────────────────────────────────────────────────────────────
    # Cybersecurity — Network and application security
    # ─────────────────────────────────────────────────────────────────────
    "PANW": ["FTNT", "CSCO", "CRWD", "OKTA"],  # Palo Alto: network, cloud, AI security
    "CRWD": ["PANW", "FTNT", "OKTA", "ZS"],  # CrowdStrike: endpoint, threat intelligence
    "FTNT": ["PANW", "CRWD", "CSCO", "OKTA"],  # Fortinet: firewalls, cloud security
    
    # ─────────────────────────────────────────────────────────────────────
    # Cloud / SaaS — Infrastructure and productivity platforms
    # ─────────────────────────────────────────────────────────────────────
    "SNOW": ["DBRX", "CRM", "MSFT", "GOOG"],  # Snowflake: data cloud, analytics
    "DDOG": ["SNOW", "MSFT", "NOW", "SPLK"],  # Datadog: monitoring, observability
    
    # ─────────────────────────────────────────────────────────────────────
    # E-commerce / Retail — Online and physical retail
    # ─────────────────────────────────────────────────────────────────────
    "WMT": ["TGT", "AMZN", "COST", "KR"],  # Walmart: discount retail, e-commerce
    "TGT": ["WMT", "AMZN", "COST", "DLTR"],  # Target: discount retail, digital
    "COST": ["WMT", "TGT", "BJ", "SFM"],  # Costco: warehouse club retail
    
    # ─────────────────────────────────────────────────────────────────────
    # Auto — Automotive manufacturers and EV
    # ─────────────────────────────────────────────────────────────────────
    "TSLA": ["GM", "F", "NIO", "RIVN"],  # Tesla: EVs, energy storage, AI
    "GM": ["F", "TSLA", "STLA", "TM"],  # General Motors: autos, EVs, autonomous
    "F": ["GM", "TSLA", "STLA", "TM"],  # Ford: traditional + EV vehicles
    "TM": ["HMC", "GM", "F", "STLA"],  # Toyota: traditional, hybrid, autonomous
    "RIVN": ["TSLA", "NIO", "LCID", "F"],  # Rivian: EV startup, trucks, vans
    
    # ─────────────────────────────────────────────────────────────────────
    # Streaming / Media — Content delivery and production
    # ─────────────────────────────────────────────────────────────────────
    "NFLX": ["DIS", "WBD", "PARA", "CMCSA"],  # Netflix: streaming, content, ads
    "DIS": ["NFLX", "WBD", "PARA", "CMCSA"],  # Disney: streaming, parks, media
    "WBD": ["DIS", "NFLX", "PARA", "CMCSA"],  # Warner Bros: streaming, content
    
    # ─────────────────────────────────────────────────────────────────────
    # Financials — Commercial Banks
    # ─────────────────────────────────────────────────────────────────────
    "JPM": ["BAC", "WFC", "C", "GS"],  # JPMorgan: universal banking, investment banking
    "BAC": ["JPM", "WFC", "C", "USB"],  # Bank of America: retail, wealth, investment
    "WFC": ["JPM", "BAC", "C", "USB"],  # Wells Fargo: retail, wealth, wholesale
    "C": ["JPM", "BAC", "WFC", "GS"],  # Citi: investment, wealth, transaction banking
    "GS": ["MS", "JPM", "BAC", "C"],  # Goldman Sachs: investment banking, trading
    "MS": ["GS", "JPM", "BAC", "C"],  # Morgan Stanley: investment banking, wealth
    "USB": ["JPM", "BAC", "WFC", "PNC"],  # US Bancorp: regional retail banking
    "PNC": ["USB", "JPM", "BAC", "WFC"],  # PNC: regional bank, investment banking
    
    # ─────────────────────────────────────────────────────────────────────
    # Financials — Payments and Fintech
    # ─────────────────────────────────────────────────────────────────────
    "V": ["MA", "AXP", "PYPL", "FIS"],  # Visa: payments, network, digital
    "MA": ["V", "AXP", "PYPL", "FIS"],  # Mastercard: payments, network, digital
    "PYPL": ["V", "MA", "AFRM", "SQ"],  # PayPal: digital payments, fintech
    "SQ": ["PYPL", "V", "MA", "AFRM"],  # Square (Block): payments, point-of-sale
    "AXP": ["V", "MA", "JPM", "C"],  # American Express: card networks, travel
    
    # ─────────────────────────────────────────────────────────────────────
    # Pharma / Biotech — Drug development and manufacturing
    # ─────────────────────────────────────────────────────────────────────
    "JNJ": ["PFE", "ABBV", "MRK", "LLY"],  # Johnson & Johnson: pharma, medical devices
    "PFE": ["JNJ", "MRNA", "ABBV", "MRK"],  # Pfizer: pharma, vaccines, consumer health
    "MRNA": ["PFE", "BNTX", "JNJ", "AZN"],  # Moderna: mRNA vaccines, therapeutics
    "LLY": ["JNJ", "PFE", "ABBV", "MRK"],  # Eli Lilly: pharma, insulin, GLP-1
    "ABBV": ["JNJ", "PFE", "MRK", "LLY"],  # AbbVie: pharma (spinoff of ABT)
    "MRK": ["JNJ", "PFE", "ABBV", "LLY"],  # Merck: pharma, oncology, vaccines
    
    # ─────────────────────────────────────────────────────────────────────
    # Healthcare Services — Health insurance and provider networks
    # ─────────────────────────────────────────────────────────────────────
    "UNH": ["CI", "CVS", "HUM", "CNC"],  # UnitedHealth: health insurance, services
    "CI": ["UNH", "CVS", "HUM", "CNC"],  # Cigna: health insurance, pharmacy
    "CVS": ["WBA", "UNH", "CI", "HUM"],  # CVS: pharmacy, retail, health services
    
    # ─────────────────────────────────────────────────────────────────────
    # Energy — Oil and Gas
    # ─────────────────────────────────────────────────────────────────────
    "XOM": ["CVX", "BP", "SHEL", "COP"],  # ExxonMobil: upstream, downstream, chemicals
    "CVX": ["XOM", "BP", "SHEL", "COP"],  # Chevron: upstream, downstream, renewables
    "COP": ["XOM", "CVX", "EOG", "PXD"],  # ConocoPhillips: upstream, LNG
    
    # ─────────────────────────────────────────────────────────────────────
    # Telecom — Telecommunications networks
    # ─────────────────────────────────────────────────────────────────────
    "T": ["VZ", "TMUS", "CMCSA", "CHTR"],  # AT&T: wireless, broadband, media
    "VZ": ["T", "TMUS", "CMCSA", "CHTR"],  # Verizon: wireless, broadband, edge
    "TMUS": ["T", "VZ", "CMCSA", "CHTR"],  # T-Mobile: wireless, broadband, prepaid
    
    # ─────────────────────────────────────────────────────────────────────
    # Consumer Staples — Food, beverage, household products
    # ─────────────────────────────────────────────────────────────────────
    "KO": ["PEP", "MNST", "STZ", "TAP"],  # Coca-Cola: beverages, spirits
    "PEP": ["KO", "MNST", "STZ", "TAP"],  # PepsiCo: beverages, snacks, foods
    "PG": ["UL", "CL", "CLX", "KMB"],  # Procter & Gamble: household, beauty, health
    "UL": ["PG", "CL", "CLX", "KMB"],  # Unilever: household, beauty, foods
    
    # ─────────────────────────────────────────────────────────────────────
    # Restaurants — Food service and quick service restaurants
    # ─────────────────────────────────────────────────────────────────────
    "MCD": ["SBUX", "YUM", "CMG", "DPZ"],  # McDonald's: QSR, franchises
    "SBUX": ["MCD", "CMG", "YUM", "DPZ"],  # Starbucks: coffee, retail, licensed
    "CMG": ["MCD", "SBUX", "YUM", "DPZ"],  # Chipotle: fast-casual, digital, delivery
    
    # ─────────────────────────────────────────────────────────────────────
    # Airlines — Commercial air transportation
    # ─────────────────────────────────────────────────────────────────────
    "DAL": ["UAL", "AAL", "LUV", "ALK"],  # Delta: full-service carrier, cargo
    "UAL": ["DAL", "AAL", "LUV", "ALK"],  # United: full-service, transatlantic
    "AAL": ["DAL", "UAL", "LUV", "ALK"],  # American: full-service, large network
    
    # ─────────────────────────────────────────────────────────────────────
    # REITs — Real estate investment trusts
    # ─────────────────────────────────────────────────────────────────────
    "SPG": ["O", "AMT", "PLD", "WPC"],  # Simon Property: shopping malls
    "AMT": ["CCI", "SBAC", "PLD", "SPG"],  # American Tower: wireless tower REIT
    
    # ─────────────────────────────────────────────────────────────────────
    # Industrials — Manufacturing, defense, equipment
    # ─────────────────────────────────────────────────────────────────────
    "BA": ["LMT", "RTX", "NOC", "GD"],  # Boeing: aerospace, defense, commercial
    "LMT": ["BA", "RTX", "NOC", "GD"],  # Lockheed Martin: defense, aerospace
    "GE": ["HON", "MMM", "RTX", "EMR"],  # General Electric: aerospace, energy, power
    "HON": ["GE", "MMM", "RTX", "EMR"],  # Honeywell: aerospace, industrial
    
    # ─────────────────────────────────────────────────────────────────────
    # Mining / Materials — Precious metals, materials
    # ─────────────────────────────────────────────────────────────────────
    "NEM": ["GOLD", "AEM", "FNV", "WPM"],  # Newmont: gold mining, production
    "GOLD": ["NEM", "AEM", "FNV", "WPM"],  # Barrick Gold: gold mining
    
    # ─────────────────────────────────────────────────────────────────────
    # Crypto / Blockchain — Digital assets and exchanges
    # ─────────────────────────────────────────────────────────────────────
    "COIN": ["MSTR", "MARA", "RIOT", "CLSK"],  # Coinbase: crypto exchange, custody
    "MSTR": ["COIN", "MARA", "RIOT", "CLSK"],  # MicroStrategy: business intelligence, Bitcoin
    
    # ─────────────────────────────────────────────────────────────────────
    # Music / Audio — Music streaming and audio
    # ─────────────────────────────────────────────────────────────────────
    "SPOT": ["AAPL", "AMZN", "GOOG", "NFLX"],  # Spotify: music streaming, podcasts
}


# ═══════════════════════════════════════════════════════════════════════════
#  Peer Engine Class
# ═══════════════════════════════════════════════════════════════════════════

class PeerEngine:
    """Discovers and compares peer companies across three strategies.
    
    Features:
      - find_peers(ticker, max_peers=5) → list of comparable companies with relevance scores
        Strategy 1: Curated PEER_MAP industry groups (primary)
        Strategy 2: Same SIC code lookup (fallback)
        Strategy 3: User-supplied custom lists (override)
      - get_peer_comparison(ticker, peer_list, year) → comparison table with rankings
        Uses extract_financials_batch() for efficient batch data fetching
        Adds rankings: best performer in each metric
    
    Deduplicates and ranks peers by relevance.
    Gracefully handles missing data.
    """
    
    def __init__(self):
        """Initialize PeerEngine.
        
        No state needed — all methods are stateless.
        This is a utility class that orchestrates peer discovery and comparisons.
        """
        # PEER_MAP is module-level constant, shared across all instances
        pass
    
    def find_peers(
        self,
        ticker: str,
        max_peers: int = 5,
        criteria: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find comparable companies for a given ticker.
        
        Uses three-strategy approach:
          1. PEER_MAP lookup (curated industry groups) — primary source
          2. SIC code lookup (same 4-digit SIC code) — fallback
          3. User criteria overrides (e.g., specific peer list) — if provided
        
        Args:
            ticker: Stock ticker (e.g., "AAPL")
            max_peers: Maximum number of peers to return (default: 5)
            criteria: Optional dict to override strategy:
              - "custom_peers": list of specific ticker symbols to use instead
              - "sic_only": bool to use only SIC code matching, skip PEER_MAP
              - "min_relevance": float 0-1 minimum relevance score to include
        
        Returns:
            List of peer dicts, each containing:
            {
              "ticker": "MSFT",  # Peer ticker symbol
              "name": "Microsoft Corp",  # Company name (from SEC if available)
              "sic": "7372",  # 4-digit SIC code (if available)
              "relevance_score": 0.95,  # 0-1 score indicating how similar (1=most relevant)
              "reason": "Same Big Tech industry group (PEER_MAP)",  # Why this is a peer
            }
            
            Result is deduplicated and sorted by relevance_score descending.
            If ticker has no known peers, returns empty list (not error).
        
        Raises:
            No exceptions — logs warnings and returns [] on failure.
        """
        # Normalize ticker to uppercase
        ticker_upper = ticker.upper()
        
        # Initialize result list to collect peers from all strategies
        peers: dict[str, dict[str, Any]] = {}  # Map ticker → peer dict (for dedup)
        
        # ─────────────────────────────────────────────────────────────────
        # Strategy 1: Curated PEER_MAP (primary source)
        # ─────────────────────────────────────────────────────────────────
        
        # Check if criteria requests SIC-only mode (skip PEER_MAP)
        sic_only = criteria and criteria.get("sic_only", False)
        
        if not sic_only:
            # Look up ticker in PEER_MAP
            if ticker_upper in PEER_MAP:
                # Get the list of peer tickers from PEER_MAP
                peer_tickers = PEER_MAP[ticker_upper]
                
                # Add each peer with high relevance (0.95) and reason
                for peer_tick in peer_tickers:
                    # Skip if already collected from another strategy
                    if peer_tick not in peers:
                        peers[peer_tick] = {
                            "ticker": peer_tick,
                            "name": "",  # Will be filled later if needed
                            "sic": "",  # Will be filled later if needed
                            "relevance_score": 0.95,  # High relevance for curated groups
                            "reason": "Curated PEER_MAP industry group",
                        }
        
        # ─────────────────────────────────────────────────────────────────
        # Strategy 2: User-supplied custom peers (highest priority if provided)
        # ─────────────────────────────────────────────────────────────────
        
        # Check if criteria includes custom peer list (overrides PEER_MAP)
        if criteria and "custom_peers" in criteria:
            # Get custom peer list from criteria
            custom = criteria["custom_peers"]
            
            # Ensure it's a list, if not, treat as single ticker
            if isinstance(custom, str):
                custom = [custom]
            
            # Clear existing peers and use only custom list
            peers.clear()
            for peer_tick in custom:
                # Normalize ticker
                peer_tick_upper = peer_tick.upper()
                # Add to peers dict with very high relevance (1.0 = explicitly chosen)
                peers[peer_tick_upper] = {
                    "ticker": peer_tick_upper,
                    "name": "",
                    "sic": "",
                    "relevance_score": 1.0,  # Highest relevance (user-specified)
                    "reason": "User-supplied custom peer",
                }
        
        # ─────────────────────────────────────────────────────────────────
        # Strategy 3: SIC Code Lookup (fallback if PEER_MAP empty)
        # ─────────────────────────────────────────────────────────────────
        
        # Try SIC lookup if we don't have peers from PEER_MAP yet
        # This provides fallback coverage for companies not in PEER_MAP
        if len(peers) == 0:
            try:
                # Import SEC client to get SIC code
                from sec_mcp.sec_client import get_sec_client
                # Get client singleton
                client = get_sec_client()
                # Fetch company info to get SIC code
                info = client.get_company_info(ticker_upper)
                sic_code = info.sic_code
                
                # If we have a SIC code, try to find other companies with same SIC
                if sic_code:
                    # Get all company tickers from SEC
                    all_tickers = client.get_all_tickers()
                    
                    # Filter for companies with matching SIC code
                    # Note: get_all_tickers returns simplified data; SIC may not be present
                    # This is a best-effort fallback
                    for other_info in all_tickers:
                        # Check if this company has same SIC code
                        if other_info.sic_code == sic_code and other_info.ticker != ticker_upper:
                            # Add as peer with moderate relevance (0.7)
                            peers[other_info.ticker] = {
                                "ticker": other_info.ticker,
                                "name": other_info.name or "",
                                "sic": sic_code,
                                "relevance_score": 0.70,  # Moderate relevance (same SIC)
                                "reason": f"Same SIC code ({sic_code})",
                            }
                            # Stop collecting after reaching limit
                            if len(peers) >= max_peers * 2:  # Collect 2x limit, will filter later
                                break
            except Exception as e:
                # SIC lookup failed (network error, missing data, etc.)
                log.warning("find_peers(%s): SIC lookup failed: %s", ticker_upper, e)
        
        # ─────────────────────────────────────────────────────────────────
        # Filter by relevance threshold if specified in criteria
        # ─────────────────────────────────────────────────────────────────
        
        min_relevance = 0.0  # Default: include all scores
        if criteria and "min_relevance" in criteria:
            # Use specified minimum relevance (0-1 scale)
            min_relevance = criteria["min_relevance"]
        
        # Filter peers by minimum relevance score
        filtered_peers = [
            p for p in peers.values()
            if p["relevance_score"] >= min_relevance
        ]
        
        # ─────────────────────────────────────────────────────────────────
        # Sort and truncate to max_peers
        # ─────────────────────────────────────────────────────────────────
        
        # Sort by relevance_score descending (highest first)
        filtered_peers.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Limit to max_peers results
        result = filtered_peers[:max_peers]
        
        # Log the result
        log.debug("find_peers(%s): found %d peers", ticker_upper, len(result))
        
        # Return the final list to caller
        return result
    
    def get_peer_comparison(
        self,
        ticker: str,
        peer_tickers: list[str] | None = None,
        year: int | None = None,
        form_type: str = "10-K",
    ) -> dict[str, Any] | None:
        """Compare financial metrics across a company and its peers.
        
        Uses extract_financials_batch() for efficient parallel data fetching.
        Adds ranking columns showing best/median/worst performer in each metric.
        
        Args:
            ticker: Primary company ticker (e.g., "AAPL")
            peer_tickers: List of peer tickers to compare. If None, auto-discovers peers.
            year: Fiscal year to fetch data for (e.g., 2024). If None, uses most recent.
            form_type: Filing type ("10-K" annual, "10-Q" quarterly, default: "10-K")
        
        Returns:
            Dict structure:
            {
              "ticker": "AAPL",
              "peers": ["MSFT", "GOOG", "AMZN", "META"],
              "year": 2024,
              "comparison": {
                "revenue": {
                  "AAPL": {
                    "value": 383285000000,
                    "rank": 2,  # 2nd highest
                    "percentile": 75,  # Top 25% (4 companies total)
                  },
                  "MSFT": {...},
                  ...
                },
                "net_income": {...},
                ...
              },
              "rankings": {
                # Maps metric → best_ticker (e.g., "MSFT" had highest revenue)
                "revenue": "AAPL",
                "net_income": "MSFT",
                ...
              }
            }
            
            Returns None if data unavailable for ticker or peers.
        
        Raises:
            No exceptions — logs errors and returns None on failure.
        """
        # Normalize primary ticker to uppercase
        ticker_upper = ticker.upper()
        
        # If peer_tickers not provided, auto-discover them
        if peer_tickers is None:
            # Use find_peers to discover peers automatically
            peer_list = self.find_peers(ticker_upper, max_peers=5)
            # Extract just the tickers from peer discovery results
            peer_tickers = [p["ticker"] for p in peer_list]
        
        # If still no peers found, return None (can't do comparison)
        if not peer_tickers:
            log.warning("get_peer_comparison(%s): no peers found", ticker_upper)
            return None
        
        # Combine primary ticker with peers for batch extraction
        # Put primary ticker first for easy reference
        all_tickers = [ticker_upper] + [p.upper() for p in peer_tickers]
        
        try:
            # Fetch financials for all companies in batch (parallel processing)
            # extract_financials_batch uses ThreadPoolExecutor for efficiency
            all_data = extract_financials_batch(
                all_tickers,
                year=year,
                form_type=form_type,
                include_statements=True,
                include_segments=False,  # Skip segments for simpler comparison
            )
        except Exception as e:
            # Batch extraction failed (network, SEC API, etc.)
            log.warning("get_peer_comparison(%s): batch extract failed: %s",
                        ticker_upper, e)
            return None
        
        # Build comparison table: metric → {ticker → {value, rank, percentile}}
        comparison: dict[str, dict[str, dict[str, Any]]] = {}
        
        # Iterate through all data results to extract metrics
        for data in all_data:
            # Skip if this result has error or missing metrics
            if data is None or data.get("error") or not data.get("metrics"):
                continue
            
            # Get ticker for this result
            result_ticker = data.get("ticker_or_cik", "").upper()
            metrics = data.get("metrics", {})
            
            # For each metric in this company's data
            for metric_name, metric_value in metrics.items():
                # Initialize metric dict if first time seeing this metric
                if metric_name not in comparison:
                    comparison[metric_name] = {}
                
                # Store this company's value for this metric
                comparison[metric_name][result_ticker] = {
                    "value": metric_value,
                    "rank": None,  # Will be computed below
                    "percentile": None,  # Will be computed below
                }
        
        # Now compute rankings and percentiles for each metric
        rankings: dict[str, str] = {}  # Maps metric → best_ticker
        
        # Iterate through each metric to compute rankings
        for metric_name, values_dict in comparison.items():
            # Sort companies by their metric value (descending, highest first)
            sorted_companies = sorted(
                values_dict.items(),
                key=lambda x: x[1]["value"] if x[1]["value"] is not None else -float('inf'),
                reverse=True,
            )
            
            # Assign ranks: 1 = best (highest), 2 = second, etc.
            for rank, (comp_ticker, comp_data) in enumerate(sorted_companies, start=1):
                # Store rank in comparison dict
                comp_data["rank"] = rank
                # Compute percentile: how many companies are better/same
                percentile = 100 - ((rank - 1) / len(sorted_companies) * 100)
                # Store percentile (0-100 scale)
                comp_data["percentile"] = round(percentile, 1)
            
            # Top ranked company (first in sorted list) is the best
            best_ticker, _ = sorted_companies[0]
            # Store in rankings dict
            rankings[metric_name] = best_ticker
        
        # Build result dict to return
        result = {
            "ticker": ticker_upper,  # Primary company
            "peers": [t for t in all_tickers if t != ticker_upper],  # Peer list
            "year": year,  # Fiscal year (may be None)
            "comparison": comparison,  # Detailed metric comparisons
            "rankings": rankings,  # Best performer per metric
        }
        
        # Log success and return
        log.debug("get_peer_comparison(%s): compared %d peers, %d metrics",
                  ticker_upper, len(peer_tickers), len(rankings))
        return result
