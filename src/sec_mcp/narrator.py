"""Claude-powered financial narrative generator.

Takes raw structured financial data from the extraction engine and uses
the Anthropic Claude API to produce a clear, readable explanation that
a non-finance person can understand.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-init the Anthropic client."""
    global _client
    if _client is not None:
        return _client

    config = get_config()
    if not config.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file to use the explain_financials tool."
        )

    import anthropic
    _client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client


def _fmt_val(v: Any) -> str:
    """Format a value for the prompt."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) >= 1e12:
            return f"${v / 1e12:,.2f}T"
        if abs(v) >= 1e9:
            return f"${v / 1e9:,.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:,.2f}M"
        if abs(v) < 1 and v != 0:
            return f"{v:.2%}"
        return f"${v:,.0f}"
    return str(v)


def _build_data_section(data: dict) -> str:
    """Format the financial data dict into a readable block for the prompt."""
    parts: list[str] = []

    parts.append(f"Company: {data.get('company_name', 'Unknown')}")
    parts.append(f"Ticker: {data.get('ticker_or_cik', '?')}")
    parts.append(f"Industry class: {data.get('industry_class', '?')}")
    parts.append(f"Fiscal year: {data.get('fiscal_year', 'latest')}")
    parts.append("")

    # Metrics
    metrics = data.get("metrics", {})
    if metrics:
        parts.append("KEY METRICS:")
        for k, v in metrics.items():
            parts.append(f"  {k}: {_fmt_val(v)}")
        parts.append("")

    # Ratios
    ratios = data.get("ratios", {})
    if ratios:
        parts.append("FINANCIAL RATIOS:")
        for k, v in ratios.items():
            if v is not None:
                parts.append(f"  {k}: {v:.2%}" if isinstance(v, float) else f"  {k}: {v}")
        parts.append("")

    # Segments
    segments = data.get("segments", {})
    if segments:
        rev_segs = segments.get("revenue_segments", [])
        if rev_segs:
            parts.append("REVENUE SEGMENTS (Product/Service):")
            for s in rev_segs:
                parts.append(f"  {s.get('segment', '?')}: {_fmt_val(s.get('value'))}")
            parts.append("")
        geo_segs = segments.get("geographic_segments", [])
        if geo_segs:
            parts.append("GEOGRAPHIC SEGMENTS:")
            for s in geo_segs:
                parts.append(f"  {s.get('segment', '?')}: {_fmt_val(s.get('value'))}")
            parts.append("")

    # Validation warnings
    validation = data.get("validation", [])
    if validation:
        parts.append("DATA QUALITY WARNINGS:")
        for w in validation:
            parts.append(f"  [{w.get('severity', '?').upper()}] {w.get('message', '')}")
        parts.append("")

    return "\n".join(parts)


SYSTEM_PROMPT = """\
You are a senior financial analyst writing for an intelligent but non-technical audience.
Given structured financial data from SEC XBRL filings, produce a clear, well-organized
narrative that explains the company's financial position.

Guidelines:
- Start with a one-paragraph executive summary.
- Use actual numbers from the data (formatted in $B / $M as appropriate).
- Highlight key strengths and concerns.
- If revenue segments or geographic segments are provided, discuss diversification.
- If ratios are provided, contextualize them (e.g. "a current ratio of 1.5 is healthy").
- If there are validation warnings, mention them diplomatically.
- Use section headers (## format) to organize: Summary, Revenue, Profitability,
  Balance Sheet, Cash Flow, Segments (if data), Outlook / Concerns.
- Be concise — aim for 400-600 words.
- Do NOT make up numbers that aren't in the data.
"""


def explain_financials(
    data: dict,
    *,
    focus: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2000,
) -> str:
    """Generate a readable narrative explanation of financial data.

    Args:
        data: the dict returned by extract_financials()
        focus: optional focus area (e.g. "profitability", "cash flow", "segments")
        model: Claude model to use
        max_tokens: max response tokens

    Returns:
        Markdown-formatted narrative string.
    """
    client = _get_client()

    data_text = _build_data_section(data)

    user_msg = f"Here is the financial data:\n\n{data_text}"
    if focus:
        user_msg += f"\n\nPlease focus especially on: {focus}"

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    # Extract text from response
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    return "\n".join(text_parts)


def explain_comparison(
    results: list[dict],
    *,
    focus: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 3000,
) -> str:
    """Generate a readable comparison narrative for multiple companies."""
    client = _get_client()

    parts: list[str] = []
    for data in results:
        parts.append(f"--- {data.get('company_name', '?')} ---")
        parts.append(_build_data_section(data))
        parts.append("")

    data_text = "\n".join(parts)

    user_msg = (
        f"Here is financial data for {len(results)} companies to compare:\n\n"
        f"{data_text}"
    )
    if focus:
        user_msg += f"\n\nPlease focus especially on: {focus}"

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT.replace(
            "produce a clear, well-organized\nnarrative that explains the company's financial position.",
            "produce a clear comparative analysis of these companies' financial positions. "
            "Use tables where helpful. Identify which company is strongest in each area.",
        ),
        messages=[{"role": "user", "content": user_msg}],
    )

    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    return "\n".join(text_parts)
