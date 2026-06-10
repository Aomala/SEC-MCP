-- SEC-MCP concept-graph + facts schema
-- Relational layer for the graph-based concept resolver (Phase 2) and the
-- queryable facts store (Phase 3). financial_cache (response-level JSONB
-- cache) is unchanged and stays alongside.

-- ── Filings ────────────────────────────────────────────────────────────────
create table if not exists sec_filings (
  accession        text primary key,
  cik              integer not null,
  ticker           text,
  form             text not null,
  filed            date,
  period_of_report date,
  fiscal_year_end  text,            -- "MMDD" from submissions
  is_amendment     boolean not null default false,
  created_at       timestamptz not null default now()
);
create index if not exists idx_sec_filings_cik_form on sec_filings (cik, form, period_of_report desc);

-- ── Facts (relational, deduped, point-in-time aware) ──────────────────────
create table if not exists sec_facts (
  id            bigint generated always as identity primary key,
  cik           integer not null,
  accession     text not null references sec_filings (accession) on delete cascade,
  taxonomy      text not null,            -- 'us-gaap' | 'ifrs-full' | 'dei' | 'custom:<prefix>'
  concept       text not null,
  unit          text not null,
  value         numeric not null,
  period_start  date,                     -- null = instant
  period_end    date not null,
  period_type   text not null,            -- 'instant'|'Q'|'H'|'9M'|'FY'|'other'
  dims          jsonb not null default '{}'::jsonb,
  dim_hash      text not null default '', -- '' = default (consolidated) context
  fy            smallint,
  fp            text,
  filed         date not null,
  unique (cik, taxonomy, concept, unit, period_start, period_end, dim_hash, accession)
);
create index if not exists idx_sec_facts_lookup
  on sec_facts (cik, concept, period_end desc) where dim_hash = '';
create index if not exists idx_sec_facts_accession on sec_facts (accession);

-- ── Taxonomy concept graph ─────────────────────────────────────────────────
create table if not exists concept_nodes (
  id            bigint generated always as identity primary key,
  taxonomy      text not null,
  taxonomy_year smallint not null,
  name          text not null,
  label         text,
  is_abstract   boolean not null default false,
  balance       text,                     -- 'debit' | 'credit' | null
  period_type   text,                     -- 'instant' | 'duration'
  deprecated    boolean not null default false,
  unique (taxonomy, taxonomy_year, name)
);
create index if not exists idx_concept_nodes_name on concept_nodes (name);

create table if not exists concept_edges (
  parent_id   bigint not null references concept_nodes (id) on delete cascade,
  child_id    bigint not null references concept_nodes (id) on delete cascade,
  edge_type   text not null check (edge_type in ('calc', 'pres', 'def', 'alias', 'deprecation')),
  weight      numeric,                    -- calc: +1 / -1
  order_index numeric,                    -- pres: display order
  role        text,                       -- ELR (statement role) the edge belongs to
  source      text not null,              -- 'fasb-2024' | 'filing:<accession>' | 'manual'
  primary key (parent_id, child_id, edge_type, source)
);
create index if not exists idx_concept_edges_child on concept_edges (child_id, edge_type);

-- ── Canonical concept layer ───────────────────────────────────────────────
create table if not exists canonical_concepts (
  key         text primary key,           -- 'revenue', 'net_income', ...
  statement   text,                       -- 'income' | 'balance' | 'cashflow' | 'other'
  fmp_field   text,                       -- field name in the FMP-shaped API
  description text
);

create table if not exists concept_mappings (
  canonical_key text not null references canonical_concepts (key) on delete cascade,
  taxonomy      text not null default 'us-gaap',
  concept_name  text not null,
  industry      text not null default 'ALL',  -- 'ALL' | IndustryClass values
  confidence    numeric not null default 0.5,
  is_total      boolean not null default true,
  source        text not null default 'manual', -- 'manual' | 'edgartools' | 'learned'
  primary key (canonical_key, taxonomy, concept_name, industry)
);
create index if not exists idx_concept_mappings_concept on concept_mappings (concept_name);

-- ── Per-filing parsed linkbase trees (immutable cache) ─────────────────────
create table if not exists filing_graphs (
  accession      text primary key references sec_filings (accession) on delete cascade,
  calc_tree      jsonb,
  pres_tree      jsonb,
  parser_version text not null,
  parsed_at      timestamptz not null default now()
);

-- ── Resolved-metric audit trail / query store ──────────────────────────────
create table if not exists metric_observations (
  cik            integer not null,
  ticker         text not null,
  canonical_key  text not null,
  period_end     date not null,
  period_type    text not null,           -- 'FY' | 'Q1'..'Q4' | 'TTM'
  fiscal_year    smallint,
  value          numeric,
  currency       text not null default 'USD',
  source_concept text,                    -- the XBRL tag that produced the value
  method         text,                    -- 'exact'|'contains'|'custom_ext'|'aggregate'|'graph_calc'|'graph_pres'|'synthesized'
  confidence     numeric,
  quality        text,                    -- 'standalone'|'standalone_decumulated'|'ytd_fallback'|'q4_synthesized'
  accession      text,
  computed_at    timestamptz not null default now(),
  primary key (cik, canonical_key, period_type, period_end)
);
create index if not exists idx_metric_obs_ticker on metric_observations (ticker, canonical_key, period_end desc);

-- ── Shadow-mode resolver disagreements (Phase 2.3) ─────────────────────────
create table if not exists resolver_diffs (
  id            bigint generated always as identity primary key,
  ticker        text not null,
  accession     text,
  canonical_key text not null,
  legacy_value  numeric,
  graph_value   numeric,
  legacy_source text,
  graph_source  text,
  rel_diff      numeric,
  created_at    timestamptz not null default now()
);

-- ── RLS: service-role only (server-side data; no public client access) ─────
alter table sec_filings        enable row level security;
alter table sec_facts          enable row level security;
alter table concept_nodes      enable row level security;
alter table concept_edges      enable row level security;
alter table canonical_concepts enable row level security;
alter table concept_mappings   enable row level security;
alter table filing_graphs      enable row level security;
alter table metric_observations enable row level security;
alter table resolver_diffs     enable row level security;
-- No policies created: anon/authenticated get nothing; service_role bypasses RLS.
