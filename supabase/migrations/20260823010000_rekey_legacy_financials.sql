-- Recover ~5,200 warm tickers that are invisible to the serving path.
--
-- THE PROBLEM
-- core/cache_keys.py defines two period keys for the same data:
--     financials_period_key()        -> 'v2|10-K|latest'   (what every read uses)
--     legacy_financials_period_key() -> '10-K|latest'      (dual-written by the batch)
--
-- Measured 2026-08-23:
--     v2|10-K|latest : 1,210 rows   <- all reads go here
--        10-K|latest : 6,369 rows   <- nothing reads this
--     v2|10-Q|latest : 1,137 rows
--        10-Q|latest :   474 rows
--
-- So ~5,200 tickers already have a warm annual extraction that the API cannot
-- see, and every request for one pays a 20-45s cold rebuild. Re-extracting them
-- instead would be ~43 hours of batch time against SEC EDGAR's 8 req/s budget.
-- This is a rename.
--
-- WHY IT IS SAFE
-- The v1->v2 bump existed because v1 blobs lacked `industry_class` and the
-- sector-aware `income_statement[]`. That is why this cannot be a blind rename.
-- Sampling 1,040 legacy rows across the table found 0 missing either marker —
-- they were written by the batch's dual-write, after v2 landed, so they are
-- already v2-shaped. The WHERE clause below still requires both markers, so a
-- genuine v1 leftover is skipped rather than promoted into the serving path.
--
-- Idempotent: NOT EXISTS means a ticker that already has a v2 row keeps it (the
-- newer key wins), and a re-run finds nothing left to move.

begin;

-- ── 10-K ────────────────────────────────────────────────────────────────────
update financial_cache c
   set cache_key = c.ticker || '|financials|v2|10-K|latest||',
       period    = 'v2|10-K|latest'
 where c.data_type = 'financials'
   and c.period    = '10-K|latest'
   and c.data ? 'industry_class'          -- v2 marker 1
   and c.data ? 'income_statement'        -- v2 marker 2
   and c.data ? 'metrics'
   and not exists (
       select 1 from financial_cache v
        where v.cache_key = c.ticker || '|financials|v2|10-K|latest||');

-- ── 10-Q ────────────────────────────────────────────────────────────────────
update financial_cache c
   set cache_key = c.ticker || '|financials|v2|10-Q|latest||',
       period    = 'v2|10-Q|latest'
 where c.data_type = 'financials'
   and c.period    = '10-Q|latest'
   and c.data ? 'industry_class'
   and c.data ? 'income_statement'
   and c.data ? 'metrics'
   and not exists (
       select 1 from financial_cache v
        where v.cache_key = c.ticker || '|financials|v2|10-Q|latest||');

-- Whatever is still on a legacy key after this is either a true v1 leftover or
-- a duplicate of a row we already serve. Neither is readable by any code path,
-- and both are re-derivable from EDGAR, so they are dropped rather than left to
-- confuse the next person who counts rows.
delete from financial_cache
 where data_type = 'financials'
   and period in ('10-K|latest', '10-Q|latest');

commit;

-- Verify (expect v2|10-K|latest to jump ~1,210 -> ~6,400 and both legacy keys to be 0):
--   select period, count(*) from financial_cache
--    where data_type = 'financials' group by 1 order by 2 desc;
