#!/bin/bash
# first_trade_watcher.sh — Telegram alert the moment the bot's FIRST trade appears.
# Runs every minute via cron; checks local SQLite AND Supabase (Render writes there).
# Manual test:  bash ~/trading-bot/scripts/first_trade_watcher.sh
# Alert history: ~/trading-bot/logs/first_trade_alerts.log
# Self-disables after the first alert (comment stays in crontab; the state
# file makes it a no-op until you delete the state file to re-arm).

export PATH=/usr/bin:/bin
BOT="$HOME/trading-bot"
DB="$BOT/trading_system.db"
STATE="$BOT/logs/.first_trade_watcher_state"
ALERTLOG="$BOT/logs/first_trade_alerts.log"
RENDER="https://trading-bot-f3yn.onrender.com"

# load Telegram + Supabase creds
set -a; . "$BOT/.env" 2>/dev/null; set +a

tg_send() {
  [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ] || return 0
  curl -s --max-time 10 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="$TELEGRAM_CHAT_ID" -d text="$1" >/dev/null 2>&1
}

# already fired? stay quiet (self-disabled)
[ -f "$STATE" ] && exit 0

mkdir -p "$BOT/logs" 2>/dev/null

# ---- count local trades ----
LOCAL_N=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trades;" 2>/dev/null || echo 0)
LAST_LOCAL=$(sqlite3 "$DB" "SELECT id||'|'||pair||'|'||direction||'|'||status||'|'||COALESCE(entry_price,0)||'|'||COALESCE(opened_at,'-') FROM trades ORDER BY id DESC LIMIT 1;" 2>/dev/null)

# ---- count Supabase trades (Render writes here too) ----
SB_N=""
if [ -n "$SUPABASE_URL" ] && [ -n "$SUPABASE_SERVICE_KEY" ]; then
  SB_N=$(curl -s --max-time 10 -D - -o /dev/null "$SUPABASE_URL/rest/v1/trades?select=id" \
    -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
    -H "Prefer: count=exact" 2>/dev/null | grep -i "^content-range:" | awk -F/ '{print $2}' | tr -d '\r')
fi

FIRE=""
if [ "${LOCAL_N:-0}" != "0" ]; then
  FIRE="LOCAL"
elif [ -n "$SB_N" ] && [ "$SB_N" != "0" ] && [ "$SB_N" != "unknown" ]; then
  FIRE="SUPABASE"
fi

if [ -n "$FIRE" ]; then
  TS=$(date -u +'%Y-%m-%d %H:%M UTC')
  {
    echo "=== FIRST TRADE DETECTED ($FIRE) at $TS ==="
    echo "local count: $LOCAL_N   supabase count: ${SB_N:-n/a}"
    echo "latest row: $LAST_LOCAL"
  } >> "$ALERTLOG"
  # pull the newest trade row for the alert body
  ROW=$(sqlite3 "$DB" "SELECT pair||' '||direction||' @ '||COALESCE(entry_price,0)||' sl='||COALESCE(sl,0)||' tp='||COALESCE(tp,0)||' ['||status||'] '||COALESCE(opened_at,'') FROM trades ORDER BY id DESC LIMIT 1;" 2>/dev/null)
  [ -z "$ROW" ] && ROW="row details in DB (source: $FIRE)"
  tg_send "🚨 FIRST TRADE! ($TS)
$ROW
local=$LOCAL_N supabase=${SB_N:-n/a}
$RENDER"
  echo "$TS" > "$STATE"   # self-disable
  exit 0
fi

# no trade yet — every 30 min, send a quiet heartbeat so you know it's alive
MINUTE=$(date +%M)
if [ "$((10#$MINUTE % 30))" -eq 0 ]; then
  ENGINE=$(curl -s --max-time 8 "$RENDER/status" 2>/dev/null | grep -oE '"engine":"[a-z]+"' || echo "?")
  echo "$(date -u +'%F %T') no trades yet (local=$LOCAL_N sb=${SB_N:-n/a}) render=$ENGINE" >> "$BOT/logs/first_trade_watch.log"
fi
exit 0
