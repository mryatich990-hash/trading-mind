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
import time

SERVICE = "srv-dasmij8jo6nc73cf1ob0"
TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""
ENV_FILE = os.path.expanduser("~/trading-bot/.env")

SKIP = {
    "DATABASE_URL", "PORT", "REDIS_URL", "TV_WEBHOOK_PORT",
    "DASHBOARD_ENABLED", "DASHBOARD_REFRESH_SEC", "DEMO_GATE_TRADES",
    "DEMO_GATE_WIN_RATE", "DEMO_GATE_PROFIT_FACTOR",  # demo-gate tuning stays local
    "LOCAL_SHADOW_MODE",  # laptop-only: cloud must NEVER shadow itself
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


def _get_all_env_vars() -> list[dict]:
    """GET env-vars with pagination (Render defaults to 20-item pages, which
    made full sets LOOK like silent PUT failures)."""
    out: list[dict] = []
    cursor: Optional[str] = None
    while True:
        path = f"/services/{SERVICE}/env-vars?limit=100"
        if cursor:
            path += f"&cursor={cursor}"
        page = api("GET", path)
        if not isinstance(page, list):
            break
        out.extend(page)
        cursor = page[-1].get("cursor") if page else None
        if not cursor or not page:
            break
    return out


local = load_env(ENV_FILE)
# NOTE: full pagination — a single GET returns one 20-item page, and a merge
# against one page silently DROPPED server-side-only keys (DATABASE_URL) on
# the next replace-all PUT.
existing = {e["envVar"]["key"]: e["envVar"]["value"]
            for e in _get_all_env_vars()}

add = {}
update = {}  # keys that exist remotely but with a DIFFERENT local value
for k, v in local.items():
    if k in SKIP or any(k.startswith(p) for p in PREFIX_SKIP):
        continue
    if k in existing:
        if existing[k] != v:
            update[k] = v  # value drift must propagate, not be skipped
        continue
    add[k] = v

if not add and not update:
    print("nothing to sync")
    sys.exit(0)

merged = dict(existing)
merged.update(add)
merged.update(update)

# Render expects a BARE ARRAY of {key, value}; the old {"envVars": [...]} wrapper
# was rejected with "invalid JSON" while the script printed success anyway.
# The GET side defaults to 20-item pages, which made full sets LOOK like they
# failed to land; verification is paginated (see _get_all_env_vars above).


def verify(expected: set[str]) -> list[str]:
    """Re-GET env vars (paginated); return expected keys NOT live."""
    live = {e["envVar"]["key"] for e in _get_all_env_vars()}
    return sorted(expected - live)


def chunked_put(items: list[dict], chunk_size: int = 12) -> None:
    """Cumulative-chunk PUT with adaptive shrink.

    Each PUT is replace-all, so every chunk must be a SUPERSET of the last.
    Render silently drops large PUTs (200 OK, nothing lands), so after each
    PUT we verify; on a partial landing the chunk SHRINKS and retries the
    same frontier until it verifies (floor: single key -> hard error).
    """
    keys = [i["key"] for i in items]
    done = 0
    take = chunk_size
    n = 0
    # first try the whole set in ONE put (works fine; the 20-item 'cap' was
    # a pagination mirage in the old verification code)
    res = api("PUT", f"/services/{SERVICE}/env-vars", items)
    if isinstance(res, dict) and res.get("message"):
        raise RuntimeError(f"PUT rejected: {res['message']}")
    time.sleep(4)
    missing = verify({i["key"] for i in items})
    if not missing:
        print(f"  single PUT: all {len(items)} keys verified live")
        return
    print(f"  single PUT incomplete ({len(missing)} missing); "
          "falling back to adaptive chunks")
    while done < len(items):
        n += 1
        take = max(1, min(take, len(items) - done))
        upto = done + take
        subset = items[:upto]
        res = api("PUT", f"/services/{SERVICE}/env-vars", subset)
        if isinstance(res, dict) and res.get("message"):
            raise RuntimeError(f"chunk PUT rejected: {res['message']} "
                               f"(keys {keys[done]}..{keys[upto - 1]})")
        time.sleep(3)  # let the env-var write settle before verification
        missing = verify({i["key"] for i in subset})
        if missing:
            if take == 1:
                raise RuntimeError(f"single key refuses to land: {keys[done]} "
                                   f"(check value format/length)")
            take = max(1, take // 2)  # shrink and retry the same frontier
            print(f"  chunk {n}: partial landing, shrinking to {take} "
                  f"(missing e.g. {missing[0]})")
            continue
        print(f"  chunk {n}: {upto}/{len(items)} keys verified live (take={take})")
        done = upto


chunked_put([{"key": k, "value": v} for k, v in merged.items()])
final_missing = verify(set(merged))
if final_missing:
    print(f"FINAL CHECK INCOMPLETE: {final_missing}", file=sys.stderr)
    sys.exit(1)
print(f"VERIFIED {len(merged)} keys live on Render")

print(f"synced {len(add)} keys to Render. Added:")
for k in sorted(add):
    v = add[k]
    show = f"<set:{len(v)}>" if any(s in k for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")) else v[:40]
    print(f"  {k} = {show}")
