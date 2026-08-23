-- API commerce layer: keys, usage metering, quota.
--
-- Additive only. Creates two new tables plus one function; touches nothing that
-- already exists (financial_cache and the concept-graph tables are untouched).
--
-- Safe to re-run: every statement is guarded.

-- ── Collision guard ─────────────────────────────────────────────────────────
-- An abandoned `api_keys` table already exists in this project from an earlier
-- attempt, with an INCOMPATIBLE and insecure shape:
--     (id bigint, key text, email, tier, calls_today, daily_limit, active)
-- It stores the API key in PLAINTEXT and has none of the columns this code
-- writes. `create table if not exists` would silently skip and leave it in
-- place, so every insert would then fail on a missing column.
--
-- It is empty (verified 0 rows on 2026-08-23), so dropping it loses nothing.
-- The row-count guard makes that assumption enforce itself: if anyone has
-- since written a row, this migration aborts loudly instead of destroying data.
do $$
declare
    n bigint;
begin
    if to_regclass('public.api_keys') is not null
       and not exists (
           select 1 from information_schema.columns
           where table_schema = 'public' and table_name = 'api_keys'
             and column_name = 'key_hash')
    then
        execute 'select count(*) from public.api_keys' into n;
        if n > 0 then
            raise exception
                'Legacy api_keys has % row(s) — migrate them by hand before rerunning. '
                'Note the legacy `key` column is PLAINTEXT: re-issue those keys rather '
                'than copying them across.', n;
        end if;
        raise notice 'Dropping empty legacy api_keys (plaintext schema)';
        execute 'drop table public.api_keys';
    end if;
end $$;

-- ── API keys ────────────────────────────────────────────────────────────────
-- The secret itself is NEVER stored. `key_hash` is sha256(full_key) and
-- `key_prefix` is the non-secret display fragment ("fin_live_7Kq2c8Zx").
create table if not exists api_keys (
    id                     uuid primary key default gen_random_uuid(),
    key_prefix             text        not null unique,
    key_hash               text        not null unique,
    name                   text        not null default 'default',
    owner_email            text        not null,
    plan                   text        not null default 'free',
    status                 text        not null default 'active',
    stripe_customer_id     text,
    stripe_subscription_id text,
    metadata               jsonb       not null default '{}'::jsonb,
    created_at             timestamptz not null default now(),
    last_used_at           timestamptz,
    revoked_at             timestamptz,
    constraint api_keys_status_check check (status in ('active', 'revoked'))
);

-- Auth runs on every request and looks up by hash — this index is the whole
-- reason auth can stay off the critical path.
create index if not exists idx_api_keys_hash_active
    on api_keys (key_hash) where status = 'active';
create index if not exists idx_api_keys_owner        on api_keys (owner_email);
create index if not exists idx_api_keys_stripe_cust  on api_keys (stripe_customer_id);

-- ── Usage rollup ────────────────────────────────────────────────────────────
-- One row per (key, UTC day, endpoint). The app keeps live counters in memory
-- and flushes deltas here every ~20s, so this table is a durable rollup rather
-- than a per-request log.
create table if not exists api_usage (
    id        bigserial   primary key,
    key_id    text        not null,   -- uuid for real keys, "anon:<ip>" otherwise
    day       date        not null,
    endpoint  text        not null,
    calls     integer     not null default 0,
    errors    integer     not null default 0,
    total_ms  bigint      not null default 0,
    updated_at timestamptz not null default now(),
    unique (key_id, day, endpoint)
);

create index if not exists idx_api_usage_key_day on api_usage (key_id, day desc);
create index if not exists idx_api_usage_day     on api_usage (day desc);

-- ── Atomic increment ────────────────────────────────────────────────────────
-- Flushes must ADD to whatever is already there. Doing read-modify-write in the
-- client would lose counts whenever two workers flush the same key at once, so
-- the arithmetic happens inside one statement.
create or replace function increment_api_usage(
    p_key_id   text,
    p_day      date,
    p_endpoint text,
    p_calls    integer,
    p_errors   integer,
    p_total_ms bigint
) returns void
language sql
as $$
    insert into api_usage (key_id, day, endpoint, calls, errors, total_ms, updated_at)
    values (p_key_id, p_day, p_endpoint, p_calls, p_errors, p_total_ms, now())
    on conflict (key_id, day, endpoint) do update
        set calls      = api_usage.calls    + excluded.calls,
            errors     = api_usage.errors   + excluded.errors,
            total_ms   = api_usage.total_ms + excluded.total_ms,
            updated_at = now();
$$;

-- ── Row level security ──────────────────────────────────────────────────────
-- Both tables hold commercial data and must never be readable with the anon
-- key. The API server uses the service role, which bypasses RLS; enabling it
-- with no permissive policy denies everyone else by default.
alter table api_keys  enable row level security;
alter table api_usage enable row level security;

comment on table  api_keys  is 'Public API credentials. Stores sha256 hashes only — never a usable key.';
comment on table  api_usage is 'Daily per-endpoint usage rollup, flushed from in-process counters.';
comment on column api_usage.key_id is 'api_keys.id for authenticated callers; "anon:<ip>" for keyless ones.';
