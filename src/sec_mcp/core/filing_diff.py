"""Filing diff engine — compares financials and narrative sections between periods.

Compares metrics (% change, significance levels) and sections (added/removed text).
Uses extract_financials for metrics, get_filing_content for sections.
Optionally uses Claude to summarize section changes; graceful fallback to raw diffs.
"""

from __future__ import annotations

import logging
from typing import Any

# Import existing financials engine for metric extraction
from sec_mcp.financials import extract_financials

# Import existing filing client for section content retrieval
from sec_mcp.edgar_client import get_filing_content

# Module logger for warnings and errors
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Filing Diff Engine
# ═══════════════════════════════════════════════════════════════════════════

class FilingDiffEngine:
    """Compares financial metrics and filing sections between two periods.
    
    Features:
      - diff_metrics(ticker, year1, year2) → shows % change + significance
        (e.g., "major" if |% change| > 20%, "moderate" if > 10%, "minor" < 10%)
      - diff_sections(ticker, section, year1, year2) → added/removed text + optional summary
        Uses Claude narrative if available; falls back to word count diff
      - Supports 10-K (annual) and 10-Q (quarterly) filings
      - Graceful fallback if financials unavailable or Claude API missing
    
    Works with existing modules:
      - Uses extract_financials() to get XBRL data for both years
      - Uses get_filing_content() to fetch full section text for comparison
      - Optionally uses narrator.explain_changes() for Claude summaries
    """
    
    # Thresholds for significance classification (in percent)
    # If abs(percent_change) > MAJOR_THRESHOLD, significance = "major"
    MAJOR_THRESHOLD = 20
    # If abs(percent_change) > MODERATE_THRESHOLD, significance = "moderate"
    MODERATE_THRESHOLD = 10
    # Otherwise significance = "minor" (for changes between 0 and MODERATE_THRESHOLD)
    
    def __init__(self):
        """Initialize FilingDiffEngine.
        
        No state needed — all methods are stateless and work with external data.
        This is a utility class that orchestrates comparisons between existing modules.
        """
        # No initialization needed; class acts as a namespace for comparison methods
        pass
    
    def diff_metrics(
        self,
        ticker: str,
        year1: int,
        year2: int,
        form_type: str = "10-K",
    ) -> dict[str, Any] | None:
        """Compare financial metrics between two years.
        
        Args:
            ticker: Stock ticker (e.g., "AAPL")
            year1: First year to compare (e.g., 2023)
            year2: Second year to compare (e.g., 2024)
            form_type: Filing type — "10-K" for annual, "10-Q" for quarterly (default: "10-K")
        
        Returns:
            Dict structure:
            {
              "ticker": "AAPL",
              "year1": 2023,
              "year2": 2024,
              "metrics": {
                "revenue": {
                  "year1": 383285000000,
                  "year2": 391035000000,
                  "change": 7750000000,
                  "change_pct": 2.02,
                  "significance": "minor"
                },
                "net_income": {
                  "year1": 96995000000,
                  "year2": 93736000000,
                  "change": -3259000000,
                  "change_pct": -3.36,
                  "significance": "minor"
                },
                ...
              },
              "summary": "Revenue grew 2.0%, net income declined 3.4%. Moderate expansion."
            }
            
            Returns None if either year's data unavailable.
        
        Raises:
            No exceptions — logs errors and returns None on failure.
        """
        # Extract financials for first year (year1)
        try:
            data1 = extract_financials(ticker, year=year1, form_type=form_type)
        except Exception as e:
            # Log error and return None if year1 data unavailable
            log.warning("diff_metrics(%s): extract_financials failed for year %d: %s",
                        ticker, year1, e)
            data1 = None
        
        # Extract financials for second year (year2)
        try:
            data2 = extract_financials(ticker, year=year2, form_type=form_type)
        except Exception as e:
            # Log error and return None if year2 data unavailable
            log.warning("diff_metrics(%s): extract_financials failed for year %d: %s",
                        ticker, year2, e)
            data2 = None
        
        # If either year's data is None, we can't compute diffs
        if data1 is None or data2 is None:
            log.warning("diff_metrics(%s): insufficient data (y1=%s, y2=%s)",
                        ticker, "present" if data1 else "missing", "present" if data2 else "missing")
            return None
        
        # Extract metrics dict from both years (contains numerical XBRL data)
        # Structure: {"metrics": {"revenue": 123000000, "net_income": 456000000, ...}}
        metrics1 = data1.get("metrics", {})
        metrics2 = data2.get("metrics", {})
        
        # Initialize result dict to be returned to caller
        result = {
            "ticker": ticker.upper(),
            "year1": year1,
            "year2": year2,
            "metrics": {},
            "summary": "",
        }
        
        # Collect all unique metric keys from both years (union of both dicts)
        # This ensures we capture metrics that existed in only one year
        all_metric_keys = set(metrics1.keys()) | set(metrics2.keys())
        
        # Iterate through each metric and compute changes
        for metric_key in sorted(all_metric_keys):
            # Get value from year1; default to None if metric didn't exist that year
            value1 = metrics1.get(metric_key)
            # Get value from year2; default to None if metric doesn't exist this year
            value2 = metrics2.get(metric_key)
            
            # Skip comparison if both values are None (metric has no data either year)
            if value1 is None and value2 is None:
                continue
            
            # Treat missing values as 0 for change calculation (e.g., new metric added)
            v1 = value1 if value1 is not None else 0
            v2 = value2 if value2 is not None else 0
            
            # Compute dollar change (year2 value minus year1 value)
            change = v2 - v1
            
            # Compute percent change; handle division by zero
            if v1 != 0:
                # Percent change = (change / old value) * 100
                change_pct = (change / v1) * 100
            else:
                # If old value is 0, percent change is undefined; report as 0 or inf
                # For display purposes, use 0 if both are 0; otherwise indicate large change
                change_pct = 0 if v2 == 0 else 999.9
            
            # Classify significance based on absolute percent change
            abs_change_pct = abs(change_pct)
            if abs_change_pct > self.MAJOR_THRESHOLD:
                # Large change (> 20%)
                significance = "major"
            elif abs_change_pct > self.MODERATE_THRESHOLD:
                # Medium change (> 10%)
                significance = "moderate"
            else:
                # Small change (< 10%)
                significance = "minor"
            
            # Store metric comparison in result
            result["metrics"][metric_key] = {
                "year1": value1,
                "year2": value2,
                "change": change,
                "change_pct": round(change_pct, 2),  # Round to 2 decimal places
                "significance": significance,
            }
        
        # Generate a brief text summary of major changes
        # Extract metrics with "major" significance to highlight
        major_changes = [
            f"{k}: {v['change_pct']:+.1f}%"
            for k, v in result["metrics"].items()
            if v.get("significance") == "major"
        ]
        
        # Build summary string
        if major_changes:
            # If there are major changes, list them
            result["summary"] = f"Major changes: {', '.join(major_changes)}"
        else:
            # Otherwise note that changes are moderate/minor
            result["summary"] = "No major changes; all metrics shifted < 20%"
        
        # Log success and return result dict
        log.debug("diff_metrics(%s): compared years %d and %d (%d metrics)",
                  ticker, year1, year2, len(result["metrics"]))
        return result
    
    def diff_sections(
        self,
        ticker: str,
        section: str,
        year1: int,
        year2: int,
        form_type: str = "10-K",
    ) -> dict[str, Any] | None:
        """Compare a narrative section between two filing periods.
        
        Fetches the same section from two years of filings and identifies
        additions/removals. Optionally uses Claude to summarize the changes;
        falls back to simple text length and word count diff if Claude unavailable.
        
        Args:
            ticker: Stock ticker (e.g., "AAPL")
            section: Section name (e.g., "risk_factors", "mda", "business")
            year1: First year (e.g., 2023)
            year2: Second year (e.g., 2024)
            form_type: Filing type — "10-K" or "10-Q" (default: "10-K")
        
        Returns:
            Dict structure:
            {
              "ticker": "AAPL",
              "section": "risk_factors",
              "year1": 2023,
              "year2": 2024,
              "text_length_y1": 12500,
              "text_length_y2": 15200,
              "text_length_change": 2700,
              "added_text": "...new risk factors...",
              "removed_text": "...deleted risk factors...",
              "summary": "Added 2700 chars of new risk disclosures. Expanded discussion..."
            }
            
            Returns None if either year's section unavailable.
        
        Raises:
            No exceptions — logs errors and returns None on failure.
        """
        # Fetch section text for year1
        try:
            text1 = get_filing_content(ticker, year=year1, form_type=form_type,
                                       section=section, max_length=100000)
        except Exception as e:
            # Log error and set to None if year1 section unavailable
            log.warning("diff_sections(%s, %s): failed to fetch year %d: %s",
                        ticker, section, year1, e)
            text1 = None
        
        # Fetch section text for year2
        try:
            text2 = get_filing_content(ticker, year=year2, form_type=form_type,
                                       section=section, max_length=100000)
        except Exception as e:
            # Log error and set to None if year2 section unavailable
            log.warning("diff_sections(%s, %s): failed to fetch year %d: %s",
                        ticker, section, year2, e)
            text2 = None
        
        # If either year's section is None, we can't compute diff
        if text1 is None or text2 is None:
            log.warning("diff_sections(%s, %s): insufficient text (y1=%s, y2=%s)",
                        ticker, section, "present" if text1 else "missing",
                        "present" if text2 else "missing")
            return None
        
        # Initialize result dict
        result = {
            "ticker": ticker.upper(),
            "section": section,
            "year1": year1,
            "year2": year2,
            "text_length_y1": len(text1),
            "text_length_y2": len(text2),
            "text_length_change": len(text2) - len(text1),
            "added_text": "",
            "removed_text": "",
            "summary": "",
        }
        
        # Simple text diff: find lines that appear in y2 but not y1 (additions)
        # and lines that appear in y1 but not y2 (removals)
        # Split both texts into paragraphs (separated by double newlines)
        # This is a crude diff but avoids external diff libraries
        paras1 = text1.split("\n\n")
        paras2 = text2.split("\n\n")
        
        # Create sets of paragraphs for comparison (ignoring order/whitespace variations)
        # Normalize by stripping whitespace and converting to lowercase
        def normalize_para(p: str) -> str:
            """Normalize paragraph for comparison."""
            return " ".join(p.split()).lower()[:100]  # Compare first 100 chars normalized
        
        normalized1 = {normalize_para(p): p for p in paras1 if p.strip()}
        normalized2 = {normalize_para(p): p for p in paras2 if p.strip()}
        
        # Find paragraphs in year2 that are new (added)
        # Collect paragraphs where the normalized form exists in y2 but not y1
        added = []
        for norm_key, para in normalized2.items():
            # If this normalized paragraph key is not in year1, it's new
            if norm_key not in normalized1:
                added.append(para)
        
        # Find paragraphs in year1 that are gone (removed)
        # Collect paragraphs where the normalized form exists in y1 but not y2
        removed = []
        for norm_key, para in normalized1.items():
            # If this normalized paragraph key is not in year2, it's been removed
            if norm_key not in normalized2:
                removed.append(para)
        
        # Truncate added/removed text to reasonable size (first 5000 chars each)
        # and join with ellipsis if necessary
        max_display_len = 5000
        if added:
            # Join added paragraphs with spacing
            added_text = "\n\n".join(added)
            # Truncate if too long
            if len(added_text) > max_display_len:
                added_text = added_text[:max_display_len] + "\n[...truncated...]"
            result["added_text"] = added_text
        
        if removed:
            # Join removed paragraphs with spacing
            removed_text = "\n\n".join(removed)
            # Truncate if too long
            if len(removed_text) > max_display_len:
                removed_text = removed_text[:max_display_len] + "\n[...truncated...]"
            result["removed_text"] = removed_text
        
        # Generate summary of the text changes
        # Attempt to use Claude if available for a narrative summary
        summary = None
        try:
            # Try to import narrator module (optional feature)
            from sec_mcp.narrator import explain_changes
            # Call explain_changes function if it exists (graceful failure if not)
            summary = explain_changes(
                ticker=ticker,
                section=section,
                year1=year1,
                year2=year2,
                added=result["added_text"][:2000],  # Limit to first 2000 chars
                removed=result["removed_text"][:2000],
            )
        except ImportError:
            # narrator module not available
            log.debug("diff_sections(%s): narrator not available for Claude summary", ticker)
        except Exception as e:
            # Claude API call failed or narrator function doesn't exist
            log.warning("diff_sections(%s): Claude summary failed: %s", ticker, e)
        
        # Fallback to simple text-based summary if Claude unavailable
        if summary is None:
            # Build summary from text length changes
            len_change = result["text_length_change"]
            num_added = len(added)
            num_removed = len(removed)
            
            if len_change > 0:
                # Section grew in size
                summary = f"Section expanded by {len_change:,} chars ({num_added} new paragraphs, {num_removed} removed)."
            elif len_change < 0:
                # Section shrank in size
                summary = f"Section contracted by {abs(len_change):,} chars ({num_added} new, {num_removed} removed)."
            else:
                # Same length overall
                summary = f"Section length unchanged ({num_added} added, {num_removed} removed)."
        
        # Store summary in result
        result["summary"] = summary
        
        # Log success and return result
        log.debug("diff_sections(%s, %s): compared %d and %d (%d added, %d removed)",
                  ticker, section, year1, year2, len(added), len(removed))
        return result
