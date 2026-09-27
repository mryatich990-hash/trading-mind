-- ============================================================
-- Supabase setup: realtime + row level security
-- Run once in Supabase Dashboard -> SQL Editor (schema tables are
-- created automatically by the engine on first boot via config/schema.sql;
-- you can also run config/schema.sql here first if you want to inspect them).
--
-- Security model:
--   * Engine (Render) uses SUPABASE_SERVICE_KEY  -> bypasses RLS, full access
--   * Dashboard (Vercel) uses anon key           -> read-only via policies
--   * Nobody can write with the anon key
-- ============================================================

-- ---- Realtime: put the hot tables on the supabase_realtime publication ----
alter publication supabase_realtime add table trades with (publish = ['insert','update','delete']);
alter publication supabase_realtime add table research_cycles with (publish = ['insert']);
alter publication supabase_realtime add table circuit_breakers with (publish = ['insert','update']);

-- ---- Row Level Security ----
alter table trades                enable row level security;
alter table research_cycles       enable row level security;
alter table circuit_breakers      enable row level security;
alter table strategy_weights      enable row level security;
alter table system_state          enable row level security;
alter table audit_log             enable row level security;
alter table backtest_results      enable row level security;

-- Anon (dashboard) may READ these; the service key bypasses RLS entirely.
create policy "anon read trades"           on trades           for select using (true);
create policy "anon read research"         on research_cycles  for select using (true);
create policy "anon read breakers"         on circuit_breakers for select using (true);
create policy "anon read weights"          on strategy_weights for select using (true);
create policy "anon read state"            on system_state     for select using (true);
create policy "anon read backtests"        on backtest_results for select using (true);

-- Engine writes over a DIRECT Postgres connection as role `tradingbot`
-- (created manually; password lives in DATABASE_URL on Render). Give it
-- full DML on every public table, plus schema CREATE for aux tables.
DO $$
DECLARE t record;
BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname='public' LOOP
    BEGIN
      EXECUTE format('DROP POLICY IF EXISTS "engine all access" ON public.%I', t.tablename);
      EXECUTE format('CREATE POLICY "engine all access" ON public.%I FOR ALL TO tradingbot USING (true) WITH CHECK (true)', t.tablename);
    EXCEPTION WHEN OTHERS THEN
      RAISE NOTICE 'skip %: %', t.tablename, SQLERRM;
    END;
  END LOOP;
END $$;
GRANT USAGE, CREATE ON SCHEMA public TO tradingbot;
GRANT ALL ON ALL TABLES IN SCHEMA public TO tradingbot;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO tradingbot;

-- No INSERT/UPDATE/DELETE policies for anon -> all writes denied for anon.
-- (Dashboard control endpoints use the service key server-side.)

-- Dashboard control endpoints (pause/close) use the service key server-side,
-- so no anon write policies exist. system_state stays read-only to the world.

-- ---- Optional hardening: revoke direct table access from anon on sensitive tables ----
revoke all on groq_audit    from anon;
revoke all on groq_rejections from anon;
