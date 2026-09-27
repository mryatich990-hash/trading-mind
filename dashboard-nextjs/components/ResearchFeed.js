import { ago } from '../lib/format'

const RESULT_CLASS = {
  accepted: 'pos',
  rejected: 'neg',
  blocked: 'warn',
}

export default function ResearchFeed({ research }) {
  return (
    <div className="card">
      <h2>Research Feed (live)</h2>
      {!research?.length ? (
        <p className="muted">No research yet.</p>
      ) : (
        research.map((r) => (
          <div className="kv" key={r.id}>
            <span className="k">
              <span className={RESULT_CLASS[r.result] || ''}>{r.result}</span>{' '}
              {r.pair} · {r.reason}
            </span>
            <span className="v muted">{ago(r.created_at)}</span>
          </div>
        ))
      )}
    </div>
  )
}
