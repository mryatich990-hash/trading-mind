import { supabaseAdmin } from '../../lib/supabase'

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type,X-Dashboard-Key',
}

/**
 * POST /api/close  { }  -> pause trading + request engine-wide close-all.
 *
 * The Vercel dashboard cannot reach the Render engine directly; instead it
 * writes state keys the engine consumes every cycle:
 *   trading_paused=1       -> no new signals
 *   close_all_requested=1  -> engine closes open trades at market, then resets
 *
 * Guarded by the shared DASHBOARD_KEY header (set in both Vercel + Render env).
 */
export default async function handler(req, res) {
  Object.entries(CORS).forEach(([k, v]) => res.setHeader(k, v))
  if (req.method === 'OPTIONS') return res.status(204).end()
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' })

  const key = req.headers['x-dashboard-key']
  if (!process.env.DASHBOARD_KEY || key !== process.env.DASHBOARD_KEY) {
    return res.status(401).json({ error: 'unauthorized' })
  }

  try {
    const db = supabaseAdmin()
    const now = new Date().toISOString()
    await db.from('system_state').upsert([
      { key: 'trading_paused', value: '1', updated_at: now },
      { key: 'close_all_requested', value: '1', updated_at: now },
    ])
    await db.from('audit_log').insert({
      category: 'operator',
      action: 'close_all',
      detail: 'requested from Vercel dashboard',
      source: 'dashboard',
    })
    res.status(200).json({ ok: true, message: 'Pause + close-all requested; engine executes within one cycle (~60s).' })
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message })
  }
}
