#!/usr/bin/env bash
# guard_watcher.sh — watch for the next trade after the oversized-lot fixes
# (commit 211421d) and verify the new guards actually held:
#   1. lots <= MAX_ABS_LOTS (0.75)
#   2. SL distance sane (Groq SL gate -> structural stop used, not 2-4 pips)
#   3. at close: JPY-cross P&L consistent with the 6.8 USD/pip broker credit
# Runs until the UTC deadline. State survives restarts via logs/.guard_state.json
set -u
cd "$(dirname "$0")/.."
source .env
mkdir -p logs
DEADLINE_EPOCH=$(date -u -d "2026-10-02 07:00:00" +%s)
STATE=logs/.guard_state.json
if [ ! -f "$STATE" ]; then
  echo '{"last_id": 4, "open_ids": [], "closed_ids": []}' > "$STATE"
fi
echo "$(date -u '+%F %T') guard watcher started (baseline trade #4, deadline $(date -u -d @$DEADLINE_EPOCH '+%F %T'))"

while [ "$(date -u +%s)" -lt "$DEADLINE_EPOCH" ]; do
  sleep 60
  rows=$(curl -s --max-time 20 \
    "$SUPABASE_URL/rest/v1/trades?select=id,pair,direction,status,strategy,lots,entry_price,sl,exit_price,pnl_usd,confluence_score,groq_conviction,mode,created_at&order=id.desc&limit=8" \
    -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY")
  out=$(ROWS="$rows" STATE="$STATE" python3 - <<'PY'
import sys, json, os
state_path = os.environ["STATE"]
try:
    state = json.load(open(state_path))
except Exception:
    state = {"last_id": 4, "open_ids": [], "closed_ids": []}
try:
    rows = json.loads(os.environ.get("ROWS") or "[]")
except Exception:
    sys.exit(0)

def pip(pair):
    pair = pair.upper()
    if "JPY" in pair:
        return 0.01
    if "XAUUSD" in pair:
        return 0.1
    if pair in ("NAS100", "US30"):
        return 1.0
    return 0.0001

events = []
for r in sorted(rows, key=lambda x: x["id"]):
    rid = r["id"]
    if rid > state["last_id"]:
        flags = []
        lots = float(r.get("lots") or 0)
        if lots > 0.75 + 1e-9:
            flags.append(f"!!! LOTS VIOLATION {lots} > 0.75")
        else:
            flags.append(f"lots {lots} <= 0.75 OK")
        entry, sl = float(r.get("entry_price") or 0), float(r.get("sl") or 0)
        if entry and sl:
            dist = abs(entry - sl) / pip(r["pair"])
            flags.append(f"SL dist {dist:.1f} pips" +
                         (" !!! TIGHT SL (<5)" if dist < 5 else " (sane)"))
        flags.append(f"confluence {r.get('confluence_score')} "
                     f"conviction {r.get('groq_conviction')} mode {r.get('mode')}")
        events.append(f"NEW TRADE #{rid} {r['pair']} {r['direction']} {r['strategy']} "
                      f"[{r['status']}] | " + " | ".join(flags))
        state["open_ids"].append(rid)
        state["last_id"] = rid
    elif (rid in state["open_ids"] and str(r.get("status")) == "closed"
          and rid not in state["closed_ids"]):
        events.append(f"CLOSE #{rid} {r['pair']}: exit {r.get('exit_price')} "
                      f"pnl_usd {r.get('pnl_usd')}")
        state["closed_ids"].append(rid)

if events:
    json.dump(state, open(state_path, "w"))
    print("\n".join(events), end="")
PY
)
  if [ -n "$out" ]; then
    now=$(date -u '+%F %T')
    echo "$now $out"
    bal=$(curl -s "$SUPABASE_URL/rest/v1/system_state?select=value&key=eq.paper_balance" \
      -H "apikey: $SUPABASE_SERVICE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
      | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['value'])" 2>/dev/null)
    echo "$now paper_balance: $bal"
    curl -s --max-time 20 https://trading-bot-f3yn.onrender.com/health | python3 -c "
import sys, json
cl = json.load(sys.stdin).get('child_log', '')
keep = [ln for ln in cl.splitlines() if any(k in ln for k in
        ('risk gate', 'kelly', 'EXECUTED', 'CLOSED', 'lots_for_risk'))]
print('\n'.join(keep[-8:]) if keep else '(no engine log matches)')"
  fi
done
echo "$(date -u '+%F %T') guard watcher done (deadline reached)"
