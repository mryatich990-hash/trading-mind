import { useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import { fmt, pnlClass } from '../lib/format'

export default function StrategyLeaderboard() {
  const [rows, setRows] = useState([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      const { data } = await supabase
        .from('trades')
        .select('strategy,status,pnl_usd')
        .eq('status', 'closed')
        .limit(500)
      if (cancelled) return
      const agg = {}
      for (const t of data || []) {
        const s = t.strategy || 'unknown'
        agg[s] = agg[s] || { trades: 0, wins: 0, pnl: 0 }
        agg[s].trades += 1
        if ((t.pnl_usd || 0) > 0) agg[s].wins += 1
        agg[s].pnl += t.pnl_usd || 0
      }
      setRows(
        Object.entries(agg)
          .map(([strategy, s]) => ({
            strategy,
            ...s,
            win_rate: s.trades ? (s.wins / s.trades) * 100 : 0,
          }))
          .sort((a, b) => b.pnl - a.pnl)
      )
    }
    load()
    const ch = supabase
      .channel('leaderboard')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'trades' }, () => load())
      .subscribe()
    return () => {
      cancelled = true
      supabase.removeChannel(ch)
    }
  }, [])

  return (
    <div className="card">
      <h2>Strategy Leaderboard</h2>
      {!rows.length ? (
        <p className="muted">No closed trades yet.</p>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th className="num">Trades</th>
                <th className="num">Win rate</th>
                <th className="num">PnL</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.strategy}>
                  <td>{r.strategy}</td>
                  <td className="num">{r.trades}</td>
                  <td className="num">{r.win_rate.toFixed(1)}%</td>
                  <td className={`num ${pnlClass(r.pnl)}`}>
                    {r.pnl > 0 ? '+' : ''}${fmt(r.pnl)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
