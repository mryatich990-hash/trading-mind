#!/usr/bin/env bash
# session_watcher.sh — watch trade #3's close + any new organic trade until a
# UTC deadline. Logs every event with engine-log confirmation lines.
# Unlike trade_count_watcher.sh this keeps running (multiple events expected).
set -u
cd "$(dirname "$0")/.."
source .env
mkdir -p logs
DEADLINE_EPOCH=$(date -u -d "2026-10-01 16:30:00" +%s)
last_trade_id=$(curl -s "$SUPABASE_URL/rest/v1/trades?select=id&order=id.desc&limit=1" \
  -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['id'])")
echo "$(date -u '+%F %T') session watcher started, last trade id = $last_trade_id"

while [ "$(date -u +%s)" -lt "$DEADLINE_EPOCH" ]; do
  sleep 90
  now=$(date -u '+%F %T')
  rows=$(curl -s "$SUPABASE_URL/rest/v1/trades?select=id,pair,direction,status,strategy,lots,entry_price,exit_price,pnl_usd,created_at&order=id.desc&limit=6" \
    -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY")
  events=$(echo "$rows" | python3 -c "
import sys, json
rows = json.load(sys.stdin)
for r in rows:
    if r['id'] > $last_trade_id:
        print(f\"NEW organic trade #{r['id']} {r['pair']} {r['direction']} {r['strategy']} status={r['status']}\")
    elif str(r.get('status')) == 'closed' and r['id'] == 3 and not r.get('exit_price'):
        pass
    elif r['id'] == 3 and str(r.get('status')) == 'closed':
        print(f\"TRADE 3 CLOSED pnl={r.get('pnl_usd')}\")
")
  if [ -n "$events" ] && [ "$events" != "$(tail -1 logs/.session_seen 2>/dev/null)" ]; then
    echo "$now $events"
    echo "$events" > logs/.session_seen
    curl -s https://trading-bot-f3yn.onrender.com/health | python3 -c "
import sys, json
cl = json.load(sys.stdin).get('child_log', '')
keep = [ln for ln in cl.splitlines() if any(k in ln for k in ('EXECUTED', 'CLOSED', 'close_matching', 'notify failed'))]
print('\n'.join(keep[-6:]) if keep else '(no engine log matches)')
"
    if echo "$events" | grep -q "NEW organic"; then
      last_trade_id=$((last_trade_id + 1))
    fi
    if echo "$events" | grep -q "TRADE 3 CLOSED"; then
      source .env
      curl -s "$SUPABASE_URL/rest/v1/system_state?select=key,value&key=eq.paper_balance" \
        -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
        | python3 -c "import sys,json;print('paper_balance after close:', json.load(sys.stdin)[0]['value'])"
    fi
  fi
done
echo "$(date -u '+%F %T') session watcher done (deadline reached)"
