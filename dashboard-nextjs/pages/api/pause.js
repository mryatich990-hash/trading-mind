import { supabaseAdmin } from '../../lib/supabase'

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type,X-Dashboard-Key',
}

/**
 * POST /api/pause  { action: "pause" | "resume" }
 * Toggles the trading_paused state key the engine checks every cycle
 * (same mechanism as the Telegram /pause command). Research continues.
 * Guarded by the shared DASHBOARD_KEY header.
 */
export default async function handler(req, res) {
  Object.entries(CORS).forEach(([k, v]) => res.setHeader(k, v))
  if (req.method === 'OPTIONS') return res.status(204).end()
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' })

  const key = req.headers['x-dashboard-key']
  if (!process.env.DASHBOARD_KEY || key !== process.env.DASHBOARD_KEY) {
    return res.status(401).json({ error: 'unauthorized' })
  }

  const action = req.body?.action
  if (action !== 'pause' && action !== 'resume') {
    return res.status(400).json({ error: 'action must be "pause" or "resume"' })
  }

  try {
    const db = supabaseAdmin()
    const now = new Date().toISOString()
    const value = action === 'pause' ? '1' : '0'
    await db.from('system_state').upsert([{ key: 'trading_paused', value, updated_at: now }])
    await db.from('audit_log').insert({
      category: 'operator',
      action,
      detail: 'from Vercel dashboard',
      source: 'dashboard',
    })
    res.status(200).json({ ok: true, paused: action === 'pause' })
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message })
  }
}
