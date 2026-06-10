"""Parse a filing's calculation + presentation linkbases into a FilingGraph.

Uses edgartools to download and parse the filing's own XBRL (the company's
linkbases override the standard taxonomy — that's how we know what THIS
filer means by revenue). Parsing costs 2-5s per filing, so graphs are cached
forever via graph.store (filings are immutable).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sec_mcp.config import get_config

log = logging.getLogger(__name__)

PARSER_VERSION = "1.0"

# Statement role classification (matched against the role URI, uppercased)
_ROLE_PATTERNS = {
    "income": re.compile(
        r"(STATEMENTSOFOPERATIONS|STATEMENTSOFINCOME|INCOMESTATEMENT|"
        r"STATEMENTOFOPERATIONS|STATEMENTOFINCOME|RESULTSOFOPERATIONS|"
        r"STATEMENTSOFEARNINGS|PROFITANDLOSS|PROFITORLOSS)"),
    "balance": re.compile(
        r"(BALANCESHEET|STATEMENTSOFFINANCIALPOSITION|STATEMENTOFFINANCIALPOSITION|"
        r"FINANCIALCONDITION)"),
    "cashflow": re.compile(
        r"(CASHFLOW|STATEMENTSOFCASHFLOWS|STATEMENTOFCASHFLOWS)"),
}
_ROLE_EXCLUDE = re.compile(r"(COMPREHENSIVE|PARENTHETICAL|DETAILS|EQUITY)")


@dataclass
class FilingGraph:
    """Per-filing concept trees, JSON-serialisable."""
    accession: str
    # role label ("income"/"balance"/"cashflow") → list of edges
    # edge: {"child": concept, "parent": concept|None, "weight": float, "order": float}
    calc: dict[str, list[dict]] = field(default_factory=dict)
    pres: dict[str, list[dict]] = field(default_factory=dict)
    parser_version: str = PARSER_VERSION

    def to_dict(self) -> dict:
        return {"accession": self.accession, "calc": self.calc,
                "pres": self.pres, "parser_version": self.parser_version}

    @classmethod
    def from_dict(cls, d: dict) -> FilingGraph:
        return cls(accession=d.get("accession", ""), calc=d.get("calc") or {},
                   pres=d.get("pres") or {}, parser_version=d.get("parser_version", ""))

    # ── Graph queries ──────────────────────────────────────────────────────
    def concepts_in(self, statement: str, tree: str = "calc") -> list[str]:
        """Concept names in a statement tree, shallowest (most total) first."""
        edges = (self.calc if tree == "calc" else self.pres).get(statement) or []
        if not edges:
            return []
        parent_of = {e["child"]: e.get("parent") for e in edges}

        def depth(c: str) -> int:
            d, cur, seen = 0, c, set()
            while parent_of.get(cur) and cur not in seen:
                seen.add(cur)
                cur = parent_of[cur]
                d += 1
            return d

        return sorted(parent_of.keys(), key=depth)

    def children_of(self, statement: str, parent: str, tree: str = "calc") -> list[dict]:
        edges = (self.calc if tree == "calc" else self.pres).get(statement) or []
        return [e for e in edges if e.get("parent") == parent]


def _classify_role(role_uri: str) -> str | None:
    upper = role_uri.upper()
    if _ROLE_EXCLUDE.search(upper):
        return None
    for label, pat in _ROLE_PATTERNS.items():
        if pat.search(upper):
            return label
    return None


def _strip_prefix(name: str) -> str:
    """'us-gaap_NetIncomeLoss' → 'NetIncomeLoss'; custom prefixes kept as-is
    minus the namespace separator so they match companyfacts concept names."""
    return name.split("_", 1)[1] if "_" in name else name


def _tree_to_edges(tree) -> list[dict]:
    edges: list[dict] = []
    nodes = getattr(tree, "all_nodes", None) or {}
    items = nodes.items() if hasattr(nodes, "items") else enumerate(nodes)
    for name, node in items:
        parent = getattr(node, "parent", None)
        edges.append({
            "child": _strip_prefix(str(name)),
            "parent": _strip_prefix(str(parent)) if parent else None,
            "weight": float(getattr(node, "weight", 1.0) or 1.0),
            "order": float(getattr(node, "order", 0.0) or 0.0),
        })
    return edges


def parse_filing_graph(ticker_or_cik: str, accession: str,
                       form_type: str | None = None) -> FilingGraph | None:
    """Download + parse one filing's linkbases. Returns None on any failure
    (caller falls back to the legacy resolver)."""
    try:
        import edgar
    except ImportError:
        log.warning("edgartools not installed — graph resolver unavailable")
        return None

    try:
        cfg = get_config()
        edgar.set_identity(cfg.edgar_identity)
        filing = edgar.find(accession)
        if filing is None:
            log.info("filing %s not found via edgartools", accession)
            return None
        x = filing.xbrl()
        if x is None:
            return None

        fg = FilingGraph(accession=accession)
        for attr, bucket in (("calculation_trees", fg.calc),
                             ("presentation_trees", fg.pres)):
            trees = getattr(x, attr, None) or {}
            items = trees.items() if hasattr(trees, "items") else []
            for role_uri, tree in items:
                label = _classify_role(str(role_uri))
                if label is None or label in bucket:
                    continue  # first matching role per statement wins
                edges = _tree_to_edges(tree)
                if edges:
                    bucket[label] = edges
        if not fg.calc and not fg.pres:
            return None
        return fg
    except Exception as exc:
        log.warning("filing graph parse failed for %s/%s: %s",
                    ticker_or_cik, accession, exc)
        return None
