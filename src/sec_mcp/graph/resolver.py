"""Graph-based concept resolution.

For a given canonical metric, the resolver asks the FILING's own calculation
tree which concept expresses it: candidates are the canonical seed tags that
actually appear in the statement's tree (plus, for revenue, any tree node
named like revenue — catches custom extensions), ordered by tree depth so the
most-total concept wins. Values still flow through the legacy `_lookup_fact`
so period/duration selection is identical to the 4-pass resolver.

Returns None whenever it has nothing better to say — the caller falls back
to the legacy resolver, so this layer can only add correctness.
"""

from __future__ import annotations

import logging
import re
import threading

import pandas as pd

from sec_mcp.graph import store
from sec_mcp.graph.canonical import seed_concepts, statement_of
from sec_mcp.graph.filing_parser import FilingGraph, parse_filing_graph
from sec_mcp.xbrl_mappings import IndustryClass

log = logging.getLogger(__name__)

_REVENUE_NODE_RE = re.compile(
    r"(Revenue|Revenues|NetSales|SalesNet|Turnover)", re.IGNORECASE)
_REVENUE_EXCLUDE_RE = re.compile(
    r"(Cost|Deferred|Unearned|Remaining|Expense|Receivable|Guidance|"
    r"Contract(Asset|Liabilit)|TaxesPayable)", re.IGNORECASE)


class GraphResolver:
    def __init__(self):
        self._graphs: dict[str, FilingGraph | None] = {}  # None = parse failed
        self._lock = threading.Lock()

    # ── Graph cache ────────────────────────────────────────────────────────
    def _graph_for(self, ticker_or_cik: str, accession: str) -> FilingGraph | None:
        with self._lock:
            if accession in self._graphs:
                return self._graphs[accession]
        fg = store.load_graph(accession)
        if fg is None:
            fg = parse_filing_graph(ticker_or_cik, accession)
            if fg is not None:
                store.save_graph(fg)
        with self._lock:
            self._graphs[accession] = fg  # cache failures too (None)
        return fg

    # ── Resolution ─────────────────────────────────────────────────────────
    def resolve(
        self,
        facts_df: pd.DataFrame | None,
        canonical_key: str,
        *,
        ticker_or_cik: str,
        accession: str | None,
        industry: IndustryClass = IndustryClass.STANDARD,
        period_index: int = 0,
        duration_pref: str | None = None,
        target_fp: str | None = None,
        target_fy: int | None = None,
    ):
        """ResolvedMetric or None (→ legacy fallback)."""
        if facts_df is None or facts_df is not None and facts_df.empty:
            return None
        if not accession:
            return None
        fg = self._graph_for(ticker_or_cik, accession)
        if fg is None:
            return None

        from sec_mcp.financials import ResolvedMetric, _lookup_fact  # lazy: avoid cycle

        stmt = statement_of(canonical_key)
        seeds = seed_concepts(canonical_key, industry)

        for tree_kind, confidence, method in (("calc", 0.97, "graph_calc"),
                                              ("pres", 0.93, "graph_pres")):
            tree_concepts = fg.concepts_in(stmt, tree_kind)
            if not tree_concepts:
                continue
            order = {c: i for i, c in enumerate(tree_concepts)}  # depth order

            # Seed order is primary: it encodes canonical MEANING (e.g. parent
            # equity over consolidated-incl-NCI, which sits shallower in the
            # tree). The tree's value-add is arbitrating which seeds exist in
            # THIS filing's statement and surfacing custom extensions — depth
            # only orders the non-seed extras below.
            in_tree = [s for s in seeds if s in order]

            if canonical_key == "revenue":
                extras = [
                    c for c in tree_concepts  # already shallowest-first
                    if c not in in_tree
                    and _REVENUE_NODE_RE.search(c)
                    and not _REVENUE_EXCLUDE_RE.search(c)
                ]
                in_tree.extend(extras)  # custom-extension revenue nodes

            for concept in in_tree:
                val = _lookup_fact(
                    facts_df, concept, period_index,
                    match_mode="exact", duration_pref=duration_pref,
                    target_fp=target_fp, target_fy=target_fy,
                )
                if val is not None:
                    return ResolvedMetric(
                        val, f"{concept} ({method})",
                        confidence=confidence, method=method,
                    )
        return None


_resolver: GraphResolver | None = None
_resolver_lock = threading.Lock()


def get_graph_resolver() -> GraphResolver:
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = GraphResolver()
    return _resolver
