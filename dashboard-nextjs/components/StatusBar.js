import { ago } from '../lib/format'

export default function StatusBar({ status }) {
  if (!status) return <div className="card muted">Loading status…</div>

  const cls = status.running ? 'running' : status.paused ? 'paused' : 'halted'
  const label = status.running
    ? 'RUNNING'
    : status.paused
      ? 'PAUSED'
      : status.halted_by?.length
        ? `HALTED: ${status.halted_by.join(', ')}`
        : 'ENGINE OFFLINE'

  return (
    <div className="card">
      <div className="kv">
        <span className="k">Status</span>
        <span className={`badge ${cls}`}>{label}</span>
      </div>
      <div className="kv">
        <span className="k">Engine</span>
        <span className={status.engine === 'alive' ? 'pos' : 'neg'}>
          {status.engine}
        </span>
      </div>
      <div className="kv">
        <span className="k">Last heartbeat</span>
        <span className="v">{ago(status.last_heartbeat)}</span>
      </div>
      <div className="kv">
        <span className="k">Trades today</span>
        <span className="v">{status.trades_today}</span>
      </div>
    </div>
  )
}
