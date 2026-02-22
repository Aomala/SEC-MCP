"""EDGAR API wrapper — DEPRECATED, use sec_client directly.

This module re-exports functions from sec_client for backward compatibility.
New code should import from sec_mcp.sec_client instead.
"""

from __future__ import annotations

# Re-export all public functions from sec_client
from sec_mcp.sec_client import (  # noqa: F401
    get_company,
    get_filing_content,
    get_sec_client,
    list_filings,
    search_companies,
)

# Re-export aliases from the canonical source for backward compatibility
from sec_mcp.section_segmenter import SECTION_ALIASES as _SEG_ALIASES

# Map item-ids back to "Item N" format for legacy callers
SECTION_ALIASES: dict[str, str] = {
    name: f"Item {item_id}" for name, item_id in _SEG_ALIASES.items()
}
