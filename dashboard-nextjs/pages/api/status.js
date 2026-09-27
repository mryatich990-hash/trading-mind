import { supabase } from '../../lib/supabase'

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

function startOfUtcDay() {
  const now = new Date()
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()))
}

export default async function handler(req, res) {
  Object.entries(CORS).forEach(([k, v]) => res.setHeader(k, v))
  if (req.method === 'OPTIONS') return res.status(204).end()
  if (req.method !== 'GET') return res.status(405).json({ error: 'GET only' })

  try {
    const since = new Date(Date.now() - 5 * 60 * 1000).toISOString()

    const [beat, pausedRow, openTrades, breakers, today] = await Promise.all([
      supabase.from('system_state').select('value').eq('key', 'engine_heartbeat').maybeSingle(),
      supabase.from('system_state').select('value').eq('key', 'trading_paused').maybeSingle(),
      supabase.from('trades').select('id', { count: 'exact', head: true }).eq('status', 'open'),
      supabase.from('circuit_breakers').select('breaker').eq('resolved', false).eq('severity', 'halt'),
      supabase.from('trades').select('status,pnl_usd').gte('created_at', startOfUtcDay().toISOString()),
    ])

    const beatTs = beat?.data?.value ? new Date(beat.data.value) : null
    const alive = beatTs ? Date.now() - beatTs.getTime() < 5 * 60 * 1000 : false
    const halted = (breakers?.data || []).map((r) => r.breaker)
    const paused = pausedRow?.data?.value === '1'
    const closed = (today?.data || []).filter((t) => t.status === 'closed')
    const dailyPnl = closed.reduce((s, t) => s + (t.pnl_usd || 0), 0)

    res.status(200).json({
      running: alive && halted.length === 0 && !paused,
      engine: alive ? 'alive' : 'offline',
      paused,
      halted_by: halted,
      open_trades: openTrades?.count ?? 0,
      trades_today: (today?.data || []).length,
      daily_pnl: Math.round(dailyPnl * 100) / 100,
      last_heartbeat: beatTs ? beatTs.toISOString() : null,
      server_time: since,
      timestamp: new Date().toISOString(),
    })
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
}
