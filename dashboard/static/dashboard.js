/* Unified dashboard: fetch-based refresh, no page reloads.
   Sections: status, account+equity, open trades, research feed,
   institutional panel, strategies, heatmap, montecarlo, ML, history. */

const REFRESH_MS = 30000;  // aligned with DASHBOARD_REFRESH_SEC (default 30s)

function fmt(n, d) { if (n === null || n === undefined) return "-"; return Number(n).toFixed(d === undefined ? 2 : d); }
function signed(n, d) { const v = Number(n || 0); return (v >= 0 ? "+" : "") + fmt(v, d); }
function cls(n) { return Number(n || 0) >= 0 ? "pos" : "neg"; }

async function getJSON(url) {
  const r = await fetch(url, {cache: "no-store"});
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}

async function control(action) {
  await fetch("/api/control", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({action: action})
  });
  refreshAll();
}

/* ---- section 1+2: status & account ---- */
async function loadStatus() {
  try {
    const s = await getJSON("/api/status");
    set("st-status", s.status);
    tint("st-status", s.status);
    set("st-session", s.session || "-");
    set("st-regime", s.regime || "-");
    set("st-vix", fmt(s.vix, 1));
    set("ac-pnl", "$" + signed(s.daily_pnl));
    el("ac-pnl").className = "v mono " + cls(s.daily_pnl);
    set("ac-trades", s.trades_today + " (" + s.wins_today + "W/" + s.losses_today + "L)");
    set("ac-open", s.open_trades);
    const banner = el("breaker-banner");
    if (s.breakers && s.breakers.length) {
      banner.style.display = "block";
      banner.textContent = "🚨 HALTED: " + s.breakers.join(", ");
    } else { banner.style.display = "none"; }
  } catch (e) { console.warn("status", e); }
}

async function loadEquity() {
  try {
    const e = await getJSON("/api/equity");
    set("ac-dd", fmt(e.drawdown_pct, 1) + "%");
    drawSparkline(e.points.map(p => p.v));
  } catch (e) { console.warn("equity", e); }
}

/* ---- section 3: open trades ---- */
async function loadOpen() {
  try {
    const d = await getJSON("/api/open_trades");
    const tb = el("open-trades").querySelector("tbody");
    tb.innerHTML = "";
    (d.trades || []).forEach(t => {
      tb.insertAdjacentHTML("beforeend",
        `<tr><td>${t.pair}</td><td>${t.direction}</td><td>${t.lots}</td>` +
        `<td class="mono">${fmt(t.entry, 5)}</td><td class="mono">${fmt(t.sl, 5)}</td>` +
        `<td class="mono">${fmt(t.tp, 5)}</td><td>${t.strategy}</td><td>${t.opened}</td></tr>`);
    });
    if (!(d.trades || []).length) tb.innerHTML = "<tr><td colspan=8 class='small'>No open trades</td></tr>";
  } catch (e) { console.warn("open", e); }
}

/* ---- section 4: research feed ---- */
async function loadResearch() {
  try {
    const d = await getJSON("/api/research");
    const tb = el("research").querySelector("tbody");
    tb.innerHTML = "";
    (d.cycles || []).forEach(r => {
      const badge = r.result === "accepted" ? "b-green" : r.result === "blocked" ? "b-amber" : "b-red";
      tb.insertAdjacentHTML("beforeend",
        `<tr><td>${r.time || ""}</td><td>${r.pair}</td>` +
        `<td><span class="badge ${badge}">${r.result}</span></td>` +
        `<td>${r.confluence_score ?? "-"}</td><td>${r.conviction ?? "-"}</td>` +
        `<td class="small">${(r.reason || "").slice(0, 60)}</td></tr>`);
    });
    if (!(d.cycles || []).length) tb.innerHTML = "<tr><td colspan=6 class='small'>No research cycles yet</td></tr>";
  } catch (e) { console.warn("research", e); }
}

/* ---- section 5: institutional ---- */
async function loadInstitutional() {
  try {
    const d = await getJSON("/api/institutional");
    el("cot-list").innerHTML = (d.cot || []).map(c =>
      `<div>${c.currency}: comm ${fmt(c.commercial_net, 0)} vs noncomm ${fmt(c.noncommercial_net, 0)} (${fmt(c.pctile, 0)}%)</div>`
    ).join("") || "<div class='small'>No COT data yet</div>";
    const b = d.biases || {};
    el("bias-list").innerHTML = Object.entries(b).map(([k, v]) =>
      `<div>${k}: <span class="${cls(v)}">${signed(v, 0)}</span></div>`
    ).join("") + `<div class='small'>DXY: ${d.dxy_trend || "-"}</div>`;
    el("retail-list").innerHTML = (d.retail || []).map(r =>
      `<div>${r.pair}: ${fmt(r.pct_long, 0)}% long</div>`).join("") ||
      "<div class='small'>No sentiment data yet</div>";
    el("vix-list").innerHTML = (d.vix_series || []).slice(-8).map(v =>
      `${v.t}: ${fmt(v.v, 1)}`).join(" · ");
  } catch (e) { console.warn("inst", e); }
}

/* ---- section 6: strategies ---- */
async function loadStrategies() {
  try {
    const d = await getJSON("/api/strategies");
    const tb = el("strategies").querySelector("tbody");
    tb.innerHTML = "";
    (d.strategies || []).forEach(s => {
      const wrCls = s.win_rate >= 50 ? "pos" : s.win_rate > 0 ? "neg" : "";
      tb.insertAdjacentHTML("beforeend",
        `<tr><td>${s.name}</td><td>${s.signals}</td>` +
        `<td class="${wrCls}">${fmt(s.win_rate, 0)}%</td>` +
        `<td class="${cls(s.pnl)}">${signed(s.pnl)}</td>` +
        `<td><div class="gauge"><div style="width:${Math.min(100, s.weight * 50)}%;background:var(--blue)"></div></div>${fmt(s.weight, 2)}</td>` +
        `<td>${s.enabled ? '<span class="badge b-green">ON</span>' : '<span class="badge b-red">OFF</span>'}</td></tr>`);
    });
    if (!(d.strategies || []).length) tb.innerHTML = "<tr><td colspan=6 class='small'>No strategies registered</td></tr>";
  } catch (e) { console.warn("strats", e); }
}

/* ---- section 7: heatmap ---- */
async function loadHeatmap() {
  try {
    const d = await getJSON("/api/heatmap");
    const host = el("heatmap");
    const cells = d.cells || [];
    if (!cells.length) { host.textContent = "No data yet (needs trades)."; return; }
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri"];
    let html = "<table><tr><th></th>" + days.map(d0 => `<th>${d0}</th>`).join("") + "</tr>";
    const byHour = {};
    cells.forEach(c => {
      const key = c.hour;
      byHour[key] = byHour[key] || {};
      if (c.day_of_week < 5) byHour[key][c.day_of_week] = c;
    });
    Object.keys(byHour).sort((a, b) => a - b).forEach(h => {
      html += `<tr><td class="small">${String(h).padStart(2, "0")}:00</td>`;
      for (let dow = 0; dow < 5; dow++) {
        const c = byHour[h][dow];
        if (!c) { html += "<td style='color:var(--border)'>·</td>"; continue; }
        const color = c.blacklisted ? "var(--red)" :
          c.win_rate >= 52 ? "var(--green)" : c.win_rate >= 40 ? "var(--amber)" : "var(--red)";
        html += `<td style="color:${color}">${fmt(c.win_rate, 0)}</td>`;
      }
      html += "</tr>";
    });
    host.innerHTML = html + "</table>";
  } catch (e) { console.warn("heatmap", e); }
}

/* ---- section 8: montecarlo ---- */
async function loadMonteCarlo() {
  try {
    const m = await getJSON("/api/montecarlo");
    if (!m.simulations) { el("montecarlo").textContent = "No simulation yet. Hit \"Run Monte Carlo\"."; return; }
    el("montecarlo").innerHTML =
      `<div class="row">
        <div class="stat"><div class="v">${fmt(m.prob_ruin, 1)}%</div><div class="l">Risk of ruin</div></div>
        <div class="stat"><div class="v">${fmt(m.prob_dd10, 1)}%</div><div class="l">P(DD ≥ 10%)</div></div>
        <div class="stat"><div class="v">${fmt(m.prob_dd20, 1)}%</div><div class="l">P(DD ≥ 20%)</div></div>
        <div class="stat"><div class="v">${fmt(m.prob_dd30, 1)}%</div><div class="l">P(DD ≥ 30%)</div></div>
      </div><div class="small">Run at ${m.run_at} · ${m.simulations} simulations</div>`;
  } catch (e) { console.warn("mc", e); }
}

/* ---- section 9: ML ---- */
async function loadML() {
  try {
    const m = await getJSON("/api/ml");
    if (!m.trained_at) { el("ml-status").textContent = "No model metrics yet (retrains every 25 trades)."; return; }
    const imps = (m.importances || []).slice(0, 5).map(([k, v]) =>
      `<div>${k}: <span class="gauge" style="display:inline-block;width:80px"><div style="width:${v * 100}%;background:var(--amber)"></div></span> ${fmt(v, 2)}</div>`).join("");
    el("ml-status").innerHTML =
      `<div>Accuracy train/test: ${fmt(m.train_accuracy * 100, 0)}% / ${fmt(m.test_accuracy * 100, 0)}% · n=${m.n_trades} · trained ${m.trained_at}</div>` +
      `<div class="small">Top features: ${imps}</div>` +
      `<div class="small">A/B test: ${m.ab_running ? "RUNNING" : "idle"}</div>`;
  } catch (e) { console.warn("ml", e); }
}

/* ---- section 10: history ---- */
async function loadHistory() {
  try {
    const pair = el("f-pair").value, outcome = el("f-outcome").value;
    const d = await getJSON(`/api/history?pair=${encodeURIComponent(pair)}&outcome=${outcome}`);
    const tb = el("history").querySelector("tbody");
    tb.innerHTML = "";
    (d.trades || []).forEach(t => {
      tb.insertAdjacentHTML("beforeend",
        `<tr><td>${t.closed}</td><td>${t.pair}</td><td>${t.direction}</td>` +
        `<td class="${cls(t.pips)}">${signed(t.pips, 1)}</td>` +
        `<td class="${cls(t.pnl)}">${signed(t.pnl)}</td><td>${t.strategy}</td>` +
        `<td>${t.confluence ?? "-"}</td></tr>`);
    });
    if (!(d.trades || []).length) tb.innerHTML = "<tr><td colspan=7 class='small'>No trades</td></tr>";
    const pairs = [...new Set((d.trades || []).map(t => t.pair))];
    const sel = el("f-pair");
    if (sel.options.length <= 1 && pairs.length) {
      pairs.forEach(p => sel.insertAdjacentHTML("beforeend", `<option>${p}</option>`));
    }
  } catch (e) { console.warn("history", e); }
}

/* ---- equity sparkline ---- */
function drawSparkline(values) {
  const c = el("equity-chart");
  if (!c || !values || values.length < 2) return;
  const ctx = c.getContext("2d");
  const w = c.width = c.offsetWidth * 2, h = c.height = 240;
  ctx.clearRect(0, 0, w, h);
  const min = Math.min(...values), max = Math.max(...values), rng = (max - min) || 1;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = i / (values.length - 1) * (w - 8) + 4;
    const y = h - 8 - (v - min) / rng * (h - 16);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = values[values.length - 1] >= values[0] ? "#3fb950" : "#f85149";
  ctx.lineWidth = 3; ctx.stroke();
}

/* ---- helpers ---- */
function el(id) { return document.getElementById(id); }
function set(id, v) { el(id).textContent = v; }
function tint(id, status) {
  const map = {RUNNING: "b-green", LIVE: "b-green", PAUSED: "b-amber", DEMO: "b-blue",
               HALTED: "b-red", OFFLINE: "b-muted"};
  el(id).className = "v badge " + (map[status] || "b-muted");
}

function refreshAll() {
  loadStatus(); loadEquity(); loadOpen(); loadResearch();
  loadInstitutional(); loadStrategies(); loadHeatmap();
  loadMonteCarlo(); loadML(); loadHistory();
}

refreshAll();
setInterval(refreshAll, REFRESH_MS);
setInterval(loadOpen, 10000);
