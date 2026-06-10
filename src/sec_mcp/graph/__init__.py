"""Concept-graph layer: per-filing calculation/presentation trees drive
concept resolution; canonical mappings + taxonomy aliases are the fallback.

Resolution philosophy (see UPGRADE notes): the right tag for "revenue" is
whatever concept sits at the revenue position of THIS filing's own
calculation tree — global priority lists can never get every filer right
because companyfacts strips statement structure by design.
"""

from sec_mcp.graph.resolver import GraphResolver, get_graph_resolver

__all__ = ["GraphResolver", "get_graph_resolver"]
