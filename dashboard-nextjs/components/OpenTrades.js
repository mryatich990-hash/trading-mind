import { fmt, ago } from '../lib/format'

export default function OpenTrades({ trades }) {
  return (
    <div className="card">
      <h2>Open Trades</h2>
      {!trades?.length ? (
        <p className="muted">No open positions.</p>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Pair</th>
                <th>Side</th>
                <th className="num">Entry</th>
                <th className="num">SL</th>
                <th className="num">TP</th>
                <th>Strategy</th>
                <th>Opened</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.id}>
                  <td>{t.pair}</td>
                  <td className={t.direction === 'buy' ? 'pos' : 'neg'}>
                    {(t.direction || '').toUpperCase()}
                  </td>
                  <td className="num">{fmt(t.entry_price, 5)}</td>
                  <td className="num">{fmt(t.sl, 5)}</td>
                  <td className="num">{fmt(t.tp, 5)}</td>
                  <td>{t.strategy}</td>
                  <td>{ago(t.opened_at || t.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
