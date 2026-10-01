#!/usr/bin/env bash
# trade_count_watcher.sh — poll Supabase for the next trade row (open or close)
# after deploy 667f47b. When a new trade appears:
#   1. print + log the row (id, pair, status, strategy, created_at)
#   2. fetch the engine child_log tail to look for the "Trade #N (M today)"
#      notification / EXECUTED line
#   3. self-disable by writing logs/.trade_count_watcher_state
# Usage: nohup bash scripts/trade_count_watcher.sh >> logs/trade_count_watcher.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source .env
mkdir -p logs
STATE="logs/.trade_count_watcher_state"

# baseline = current max trade id at watcher start
baseline=$(curl -s "$SUPABASE_URL/rest/v1/trades?select=id&order=id.desc&limit=1" \
  -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['id'])")
echo "$(date -u '+%F %T') watcher started, baseline trade id = $baseline"

while [ ! -f "$STATE" ]; do
  sleep 60
  now=$(date -u '+%F %T')
  rows=$(curl -s "$SUPABASE_URL/rest/v1/trades?select=id,pair,direction,status,strategy,lots,entry_price,exit_price,pnl_usd,created_at&order=id.desc&limit=5" \
    -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY")
  new=$(echo "$rows" | python3 -c "
import sys, json
rows = json.load(sys.stdin)
new = [r for r in rows if r['id'] > $baseline]
if new:
    for r in new:
        print('NEW TRADE:', json.dumps(r))
else:
    print('none')
")
  if [ "$new" != "none" ]; then
    echo "$now $new"
    # engine log tail around the fill/close to confirm the alert fired
    curl -s https://trading-bot-f3yn.onrender.com/health | python3 -c "
import sys, json
cl = json.load(sys.stdin).get('child_log', '')
tail = cl[-4000:] if isinstance(cl, str) else str(cl)[-4000:]
keep = [ln for ln in tail.splitlines() if any(k in ln for k in ('EXECUTED', 'CLOSED', 'notify', 'Trade #', 'Telegram'))]
print('ENGINE LOG MATCHES:')
print('\n'.join(keep[-15:]) if keep else '(no EXECUTED/CLOSED lines in recent log window)')
"
    echo "trade_count_watcher: FIRED — new trade row seen; alert text should contain 'Trade #N (M today)'"
    touch "$STATE"
    break
  fi
done
echo "$(date -u '+%F %T') watcher done"
