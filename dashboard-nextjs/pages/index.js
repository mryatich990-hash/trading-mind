import { useEffect, useState } from 'react'
import Head from 'next/head'
import { supabase } from '../lib/supabase'
import StatusBar from '../components/StatusBar'
import AccountPanel from '../components/AccountPanel'
import OpenTrades from '../components/OpenTrades'
import ResearchFeed from '../components/ResearchFeed'
import SystemHealth from '../components/SystemHealth'
import StrategyLeaderboard from '../components/StrategyLeaderboard'

export default function Dashboard() {
  const [status, setStatus] = useState(null)
  const [trades, setTrades] = useState([])
  const [research, setResearch] = useState([])
  const [breakers, setBreakers] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const dayStart = new Date()
        dayStart.setUTCHours(0, 0, 0, 0)

        const [beat, paused, open, brk, today, research] = await Promise.all([
          supabase.from('system_state').select('value').eq('key', 'engine_heartbeat').maybeSingle(),
          supabase.from('system_state').select('value').eq('key', 'trading_paused').maybeSingle(),
          supabase.from('trades').select('*').eq('status', 'open').order('opened_at', { ascending: false }),
          supabase.from('circuit_breakers').select('breaker,reason,triggered_at').eq('resolved', false).eq('severity', 'halt'),
          supabase.from('trades').select('status,pnl_usd').gte('created_at', dayStart.toISOString()),
          supabase.from('research_cycles').select('*').order('id', { ascending: false }).limit(10),
        ])
        if (cancelled) return

        const beatTs = beat?.data?.value ? new Date(beat.data.value) : null
        const alive = beatTs ? Date.now() - beatTs.getTime() < 5 * 60 * 1000 : false
        const haltedBy = (brk?.data || []).map((r) => r.breaker)
        const closed = (today?.data || []).filter((t) => t.status === 'closed')
        const dailyPnl = closed.reduce((s, t) => s + (t.pnl_usd || 0), 0)

        setStatus({
          engine: alive ? 'alive' : 'offline',
          paused: paused?.data?.value === '1',
          halted_by: haltedBy,
          running: alive && haltedBy.length === 0 && paused?.data?.value !== '1',
          open_trades: (open?.data || []).length,
          trades_today: (today?.data || []).length,
          daily_pnl: Math.round(dailyPnl * 100) / 100,
          last_heartbeat: beatTs ? beatTs.toISOString() : null,
        })
        setTrades(open?.data || [])
        setBreakers(brk?.data || [])
        setResearch(research?.data || [])
        setError('')
      } catch (e) {
        if (!cancelled) setError(e.message)
      }
    }

    load()
    const poll = setInterval(load, 30000) // heartbeat refresh; realtime covers the rest

    // Realtime: instant updates when the engine writes.
    const tradesCh = supabase
      .channel('trades-realtime')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'trades' }, () => load())
      .subscribe()
    const researchCh = supabase
      .channel('research-realtime')
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'research_cycles' }, () => load())
      .subscribe()
    const breakersCh = supabase
      .channel('breakers-realtime')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'circuit_breakers' }, () => load())
      .subscribe()

    return () => {
      cancelled = true
      clearInterval(poll)
      supabase.removeChannel(tradesCh)
      supabase.removeChannel(researchCh)
      supabase.removeChannel(breakersCh)
    }
  }, [])

  return (
    <>
      <Head>
        <title>Trading Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <div className="dashboard">
        {breakers.length > 0 && (
          <div className="banner">
            ⚠ HALTED: {breakers.map((b) => `${b.breaker} — ${b.reason || ''}`).join(' | ')}
          </div>
        )}
        {error && <div className="banner">Supabase error: {error}</div>}

        <StatusBar status={status} />
        <AccountPanel status={status} />
        <OpenTrades trades={trades} />
        <ResearchFeed research={research} />
        <StrategyLeaderboard />
        <SystemHealth status={status} />
      </div>
    </>
  )
}
