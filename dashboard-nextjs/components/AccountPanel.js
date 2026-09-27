import { fmt, pnlClass } from '../lib/format'

export default function AccountPanel({ status }) {
  if (!status) return <div className="card muted">Loading account…</div>

  return (
    <div className="card">
      <h2>Account</h2>
      <div className="kv">
        <span className="k">Daily PnL</span>
        <span className={`v ${pnlClass(status.daily_pnl)}`}>
          {status.daily_pnl > 0 ? '+' : ''}${fmt(status.daily_pnl)}
        </span>
      </div>
      <div className="kv">
        <span className="k">Open positions</span>
        <span className="v">{status.open_trades}</span>
      </div>
      <p className="muted" style={{ marginTop: 8 }}>
        Balance/equity render here once the broker account snapshot table is
        wired (engine writes it with the next deploy).
      </p>
    </div>
  )
}
