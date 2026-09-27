export default function SystemHealth({ status }) {
  return (
    <div className="card">
      <h2>System Health</h2>
      <div className="kv">
        <span className="k">Engine</span>
        <span className={status?.engine === 'alive' ? 'pos' : 'neg'}>
          {status?.engine || '—'}
        </span>
      </div>
      <div className="kv">
        <span className="k">Trading paused</span>
        <span className="v">{status?.paused ? 'yes' : 'no'}</span>
      </div>
      <div className="kv">
        <span className="k">Halt breakers</span>
        <span className="v">{status?.halted_by?.length || 0}</span>
      </div>
      {process.env.NEXT_PUBLIC_RENDER_HEALTH_URL && (
        <p className="muted" style={{ marginTop: 8 }}>
          Health endpoint:{' '}
          <a href={process.env.NEXT_PUBLIC_RENDER_HEALTH_URL} target="_blank" rel="noreferrer">
            {process.env.NEXT_PUBLIC_RENDER_HEALTH_URL}
          </a>
        </p>
      )}
    </div>
  )
}
