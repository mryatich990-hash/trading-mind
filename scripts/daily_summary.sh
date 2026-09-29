#!/bin/bash
# daily_summary.sh — one-glance "did the bot trade?" report.
# Run manually:   bash ~/trading-bot/scripts/daily_summary.sh
# Via cron:       5 3 * * * /bin/bash $HOME/trading-bot/scripts/daily_summary.sh >> $HOME/trading-bot/logs/daily_history.txt 2>&1
# Notes: DB/log timestamps are UTC. Log *wall clock* shows EAT (UTC+3) because the
#        machine timezone is Africa/Nairobi — that is timezone, not drift.

export PATH=/usr/bin:/bin
BOT="$HOME/trading-bot"
DB="$BOT/trading_system.db"
LOG="$BOT/logs/bot.log"
OUT="$BOT/logs/daily_summary.txt"
RENDER="https://trading-bot-f3yn.onrender.com"

# Never let two runs overlap
if command -v flock >/dev/null 2>&1; then
  exec 9>/tmp/daily_summary.lock
  flock -n 9 || exit 0
fi

UTC_TODAY=$(date -u +%F)
TODAY_START="$UTC_TODAY 00:00:00"
H24_AGO=$(date -u -d '24 hours ago' +'%F %T')
LOG_DAY=$(date +%F)   # log file date line (machine-local day)

q() { sqlite3 "$DB" "$1" 2>/dev/null || echo "n/a"; }

{
echo "==============================================="
echo " TRADING BOT DAILY SUMMARY"
echo " Generated: $(date -u +'%Y-%m-%d %H:%M UTC')  ($(date +'%H:%M %Z'))"
echo " Window:    $UTC_TODAY (UTC day)  |  log-day: $LOG_DAY (EAT)"
echo "==============================================="

echo
echo "--- 1) LOCAL DATABASE (source of truth) ---"
TOTAL=$(q "SELECT COUNT(*) FROM trades;")
echo "Trades ever: $TOTAL"
STATUS_ROWS=$(q "SELECT COALESCE(status,'-')||': '||COUNT(*) FROM trades GROUP BY status ORDER BY 2 DESC;")
[ "$STATUS_ROWS" != "n/a" ] && echo "$STATUS_ROWS" | sed 's/^/  by status: /'
TODAY_N=$(q "SELECT COUNT(*) FROM trades WHERE created_at >= '$TODAY_START';")
H24_N=$(q "SELECT COUNT(*) FROM trades WHERE created_at >= '$H24_AGO';")
echo "New today (UTC): $TODAY_N   |  LAST 24H: $H24_N  <-- headline (full previous UTC day when run just after UTC midnight)"
q "SELECT 'closed today: '||COUNT(*)||'  pnl_usd='||ROUND(COALESCE(SUM(pnl_usd),0),2) FROM trades WHERE status='closed' AND closed_at >= '$TODAY_START';"
q "SELECT 'wins today: '||COUNT(*) FROM trades WHERE status='closed' AND closed_at >= '$TODAY_START' AND pnl_usd > 0;"
q "SELECT 'losses today: '||COUNT(*) FROM trades WHERE status='closed' AND closed_at >= '$TODAY_START' AND pnl_usd < 0;"
LAST=$(q "SELECT 'last trade ever: '||id||' '||pair||' '||direction||' '||status||' entry='||COALESCE(entry_price,'-')||' opened='||COALESCE(opened_at,'-') FROM trades ORDER BY id DESC LIMIT 1;")
[ -z "$LAST" ] && LAST="last trade ever: NONE (table empty)"
echo "$LAST"

echo
echo "--- 2) LOG ACTIVITY (bot.log, day $LOG_DAY, EAT clock) ---"
OPN=0
if [ -f "$LOG" ]; then
  TODAY_LINES=$(grep -a "^$LOG_DAY" "$LOG")
  DEN=$(echo "$TODAY_LINES" | grep -ac "DENIED" || true)
  APP=$(echo "$TODAY_LINES" | grep -aic "APPROVED" || true)
  OPN=$(echo "$TODAY_LINES" | grep -aicE "Trade opened|opened_at|PAPER (BUY|SELL)" || true)
  FF=$(echo "$TODAY_LINES" | grep -ac "feed yfinance failed" || true)
  FR=$(echo "$TODAY_LINES" | grep -ac "recovery: data feed healthy" || true)
  echo "Research denials: $DEN   approvals: $APP   trade-open lines: $OPN"
  echo "Feed failures: $FF   feed recoveries: $FR"
  echo "Top denial reasons:"
  echo "$TODAY_LINES" | grep -aoE "step[0-9]+[^:]*" | sort | uniq -c | sort -rn | head -5 | sed 's/^/  /'
  echo "$TODAY_LINES" | grep -aE "research [A-Z]{6} (buy|sell)" | tail -1 | cut -c1-170 | sed 's/^/  latest verdict: /'
else
  echo "bot.log not found"
fi

echo
echo "--- 3) BOT PROCESS ---"
PID=$(pgrep -f "\.venv/bin/python main\.py" | head -1 || true)
if [ -n "$PID" ]; then
  echo "Local engine: RUNNING (pid $PID)"
else
  echo "Local engine: NOT RUNNING  <-- investigate"
fi
if [ -f "$LOG" ]; then
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  [ "$AGE" -lt 0 ] && AGE=0
  echo "Log age: ${AGE}s since last write"
fi

echo
echo "--- 4) CLOUD (Render) ---"
R=$(curl -s --max-time 10 "$RENDER/status" || true)
if [ -n "$R" ]; then
  echo "$R" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("Render: engine=%s trades_today=%s open=%s halted_by=%s" % (d.get("engine"), d.get("trades_today"), d.get("open_trades"), d.get("halted_by")))' 2>/dev/null || echo "Render: unparseable response: $(echo "$R" | head -c 200)"
else
  echo "Render: UNREACHABLE"
fi

echo
echo "--- 5) SUPABASE (cloud mirror) ---"
if [ -f "$BOT/.env" ]; then
  set -a; . "$BOT/.env" 2>/dev/null; set +a
fi
if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_KEY:-}" ]; then
  N=$(curl -s --max-time 10 -D - -o /dev/null "$SUPABASE_URL/rest/v1/trades?select=id" \
      -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
      -H "Prefer: count=exact" | grep -i "^content-range:" | awk -F/ '{print $2}' | tr -d '\r')
  echo "Supabase trades total: ${N:-unknown}"
  # cloud-primary: the headline is what the CLOUD engine traded in the last 24h
  C24=$(curl -s --max-time 10 -D - -o /dev/null \
      "$SUPABASE_URL/rest/v1/trades?select=id&created_at=gte.$H24_AGO" \
      -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
      -H "Prefer: count=exact" | grep -i "^content-range:" | awk -F/ '{print $2}' | tr -d '\r')
  echo "Cloud trades last 24h: ${C24:-unknown}  <-- PRIMARY ENGINE"
else
  echo "Supabase: skipped (no SUPABASE_URL/SERVICE_KEY in .env)"
fi

echo
echo "==============================================="
if { [ "$TOTAL" = "0" ] || [ "$TOTAL" = "n/a" ]; } && [ "${OPN:-0}" = "0" ] && [ "${H24_N:-0}" = "0" ]; then
  echo "BOTTOM LINE: still zero trades. See denial reasons above."
else
  echo "BOTTOM LINE: TRADING ACTIVITY DETECTED — check details above."
fi
echo "==============================================="
} > "$OUT" 2>&1

cat "$OUT"
