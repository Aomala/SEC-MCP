"""Persistence for parsed filing graphs.

Two tiers, both optional and graceful:
  1. Disk:     ~/.sec_mcp_cache/_graphs/<accession>.json  (no TTL — immutable)
  2. Supabase: filing_graphs table (when SUPABASE_URL/KEY configured and the
               concept-graph migration has been applied)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sec_mcp.graph.filing_parser import PARSER_VERSION, FilingGraph

log = logging.getLogger(__name__)

_GRAPH_DIR = Path.home() / ".sec_mcp_cache" / "_graphs"


def _disk_path(accession: str) -> Path:
    safe = accession.replace("/", "_")
    return _GRAPH_DIR / f"{safe}.json"


def load_graph(accession: str) -> FilingGraph | None:
    p = _disk_path(accession)
    try:
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("parser_version") == PARSER_VERSION:
                return FilingGraph.from_dict(d)
    except Exception as exc:
        log.debug("graph disk load failed for %s: %s", accession, exc)

    row = _supabase_load(accession)
    if row is not None:
        save_graph_disk(row)  # warm the disk tier
        return row
    return None


def save_graph(fg: FilingGraph) -> None:
    save_graph_disk(fg)
    _supabase_save(fg)


def save_graph_disk(fg: FilingGraph) -> None:
    try:
        _GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        _disk_path(fg.accession).write_text(json.dumps(fg.to_dict()))
    except Exception as exc:
        log.debug("graph disk save failed for %s: %s", fg.accession, exc)


# ── Supabase tier (optional) ───────────────────────────────────────────────

def _supabase_client():
    try:
        from sec_mcp.supabase_cache import _get_client
        return _get_client()
    except Exception:
        return None


def _supabase_load(accession: str) -> FilingGraph | None:
    client = _supabase_client()
    if client is None:
        return None
    try:
        res = (client.table("filing_graphs").select("calc_tree,pres_tree,parser_version")
               .eq("accession", accession).limit(1).execute())
        rows = res.data or []
        if rows and rows[0].get("parser_version") == PARSER_VERSION:
            return FilingGraph(accession=accession,
                               calc=rows[0].get("calc_tree") or {},
                               pres=rows[0].get("pres_tree") or {},
                               parser_version=rows[0]["parser_version"])
    except Exception as exc:
        log.debug("filing_graphs select failed (%s): %s", accession, exc)
    return None


def _supabase_save(fg: FilingGraph) -> None:
    client = _supabase_client()
    if client is None:
        return
    try:
        # filing_graphs.accession references sec_filings — upsert the parent
        # stub first so the FK never blocks a graph save.
        client.table("sec_filings").upsert(
            {"accession": fg.accession, "cik": 0, "form": "unknown"},
            on_conflict="accession", ignore_duplicates=True).execute()
        client.table("filing_graphs").upsert({
            "accession": fg.accession,
            "calc_tree": fg.calc,
            "pres_tree": fg.pres,
            "parser_version": fg.parser_version,
        }, on_conflict="accession").execute()
    except Exception as exc:
        log.debug("filing_graphs upsert failed (%s): %s", fg.accession, exc)
