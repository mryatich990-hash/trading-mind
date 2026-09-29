#!/usr/bin/env python3
"""Sync local .env tuning keys to the Render service (one-off migration).

- Keeps every key already set on Render (DATABASE_URL, SUPABASE_*, etc.)
- Adds local keys that are missing on Render (risk limits, gates, tuning)
- Skips local-only infra keys that must NOT be copied (sqlite path, ports,
  redis, MT5 terminal vars, Vercel-side NEXT_PUBLIC_*)
- Never prints secret values; writes a short summary
"""
import json
import os
import subprocess
import sys

SERVICE = "srv-dasmij8jo6nc73cf1ob0"
TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""
ENV_FILE = os.path.expanduser("~/trading-bot/.env")

SKIP = {
    "DATABASE_URL", "PORT", "REDIS_URL", "TV_WEBHOOK_PORT",
    "DASHBOARD_ENABLED", "DASHBOARD_REFRESH_SEC", "DEMO_GATE_TRADES",
    "DEMO_GATE_WIN_RATE", "DEMO_GATE_PROFIT_FACTOR",  # demo-gate tuning stays local
}
PREFIX_SKIP = ("MT5_", "MT", "NEXT_PUBLIC_")


def load_env(path):
    env = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if k and v.strip() != "":
            env[k] = v.strip()
    return env


def api(method, path, payload=None):
    """Render API call via curl (local Python TLS gets reset by the ISP)."""
    cmd = ["curl", "-s", "--max-time", "40", "-X", method,
           f"https://api.render.com/v1{path}",
           "-H", f"Authorization: Bearer {TOKEN}"]
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json",
                "--data-binary", json.dumps(payload)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"curl failed: {out.stderr[:200]}")
    return json.loads(out.stdout) if out.stdout.strip() else {}


local = load_env(ENV_FILE)
existing = {e["envVar"]["key"]: e["envVar"]["value"]
            for e in api("GET", f"/services/{SERVICE}/env-vars")}

add = {}
for k, v in local.items():
    if k in SKIP or k in existing or any(k.startswith(p) for p in PREFIX_SKIP):
        continue
    add[k] = v

if not add:
    print("nothing to sync")
    sys.exit(0)

merged = dict(existing)
merged.update(add)
payload = {"envVars": [{"key": k, "value": v} for k, v in merged.items()]}
api("PUT", f"/services/{SERVICE}/env-vars", payload)
print(f"synced {len(add)} keys to Render (total {len(merged)}). Added:")
for k in sorted(add):
    v = add[k]
    show = f"<set:{len(v)}>" if any(s in k for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")) else v[:40]
    print(f"  {k} = {show}")
