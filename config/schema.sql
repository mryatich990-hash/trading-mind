-- Unified schema: creates every table needed by every module.
-- Idempotent: CREATE TABLE IF NOT EXISTS throughout. SQLite-compatible types.

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    pair VARCHAR(16) NOT NULL,
    direction VARCHAR(8) NOT NULL,
    lots DOUBLE PRECISION DEFAULT 0,
    entry_price DOUBLE PRECISION,
    sl DOUBLE PRECISION,
    tp DOUBLE PRECISION,
    tp2 DOUBLE PRECISION,
    tp3 DOUBLE PRECISION,
    opened_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    status VARCHAR(16) DEFAULT 'signal',
    strategy VARCHAR(48) DEFAULT '',
    session VARCHAR(16) DEFAULT '',
    confluence_score INTEGER DEFAULT 0,
    groq_conviction INTEGER DEFAULT 0,
    groq_reasoning TEXT DEFAULT '',
    is_reentry BOOLEAN DEFAULT FALSE,
    exit_price DOUBLE PRECISION,
    pips DOUBLE PRECISION DEFAULT 0,
    pnl_usd DOUBLE PRECISION DEFAULT 0,
    rr_achieved DOUBLE PRECISION DEFAULT 0,
    slippage_pips DOUBLE PRECISION DEFAULT 0,
    swap_cost_usd DOUBLE PRECISION DEFAULT 0,
    signal_hash VARCHAR(64) DEFAULT '',
    ml_loss_prob DOUBLE PRECISION,
    features_json TEXT DEFAULT '',
    mode VARCHAR(8) DEFAULT 'live',
    strategy_version VARCHAR(16) DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair);
CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at);

CREATE TABLE IF NOT EXISTS research_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    pair VARCHAR(16),
    result VARCHAR(16),              -- accepted / rejected / blocked
    reason VARCHAR(255) DEFAULT '',
    confluence_score INTEGER DEFAULT 0,
    conviction INTEGER DEFAULT 0,
    research_json TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS groq_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    pair VARCHAR(16),
    stage VARCHAR(32),
    prompt TEXT,
    response TEXT,
    parsed_json TEXT DEFAULT '',
    attempt INTEGER DEFAULT 1,
    temperature DOUBLE PRECISION DEFAULT 0,
    accepted BOOLEAN DEFAULT FALSE,
    reject_reason VARCHAR(255) DEFAULT '',
    verify_score DOUBLE PRECISION,
    latency_ms INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS groq_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    pair VARCHAR(16),
    reason VARCHAR(255),
    score DOUBLE PRECISION DEFAULT 0,
    failed_claims TEXT DEFAULT '',
    raw_response TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS strategy_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    pair VARCHAR(16),
    scores TEXT DEFAULT '{}',
    selected VARCHAR(48) DEFAULT '',
    reason VARCHAR(255) DEFAULT ''
);

CREATE TABLE IF NOT EXISTS strategy_weights (
    strategy VARCHAR(48) PRIMARY KEY,
    weight DOUBLE PRECISION DEFAULT 1.0,
    enabled BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    reason VARCHAR(255) DEFAULT ''
);

CREATE TABLE IF NOT EXISTS psychology_state (
    key VARCHAR(64) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS circuit_breakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    breaker VARCHAR(64),
    reason TEXT,
    severity VARCHAR(16) DEFAULT 'halt',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS feed_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    component VARCHAR(64),
    healthy BOOLEAN DEFAULT FALSE,
    detail VARCHAR(255) DEFAULT ''
);

CREATE TABLE IF NOT EXISTS news_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    title TEXT,
    country VARCHAR(8) DEFAULT '',
    impact VARCHAR(16) DEFAULT 'low',
    event_time TIMESTAMPTZ,
    actual VARCHAR(32) DEFAULT '',
    forecast VARCHAR(32) DEFAULT '',
    surprise DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS news_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    headline TEXT,
    source VARCHAR(32) DEFAULT '',
    currencies TEXT DEFAULT '[]',
    sentiment VARCHAR(16) DEFAULT 'neutral',
    impact VARCHAR(16) DEFAULT 'low'
);

CREATE TABLE IF NOT EXISTS cot_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date DATE,
    currency VARCHAR(8),
    commercial_net DOUBLE PRECISION,
    noncommercial_net DOUBLE PRECISION,
    retail_net DOUBLE PRECISION,
    open_interest DOUBLE PRECISION,
    commercial_pctile DOUBLE PRECISION,
    noncommercial_pctile DOUBLE PRECISION,
    UNIQUE(report_date, currency)
);

CREATE TABLE IF NOT EXISTS retail_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    pair VARCHAR(16),
    pct_long DOUBLE PRECISION,
    pct_short DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS intermarket_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    usd_bias DOUBLE PRECISION DEFAULT 0,
    eur_bias DOUBLE PRECISION DEFAULT 0,
    gbp_bias DOUBLE PRECISION DEFAULT 0,
    jpy_bias DOUBLE PRECISION DEFAULT 0,
    xau_bias DOUBLE PRECISION DEFAULT 0,
    dxy_trend VARCHAR(16) DEFAULT '',
    vix DOUBLE PRECISION,
    vix3m DOUBLE PRECISION,
    yield_10y DOUBLE PRECISION,
    yield_2y DOUBLE PRECISION,
    yield_spread DOUBLE PRECISION,
    spx_direction VARCHAR(8) DEFAULT ''
);

CREATE TABLE IF NOT EXISTS market_regime (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    regime VARCHAR(16),
    confidence INTEGER DEFAULT 0,
    recommended TEXT DEFAULT '[]',
    avoid TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS volume_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    pair VARCHAR(16),
    session_date DATE,
    poc DOUBLE PRECISION,
    vah DOUBLE PRECISION,
    val DOUBLE PRECISION,
    vpoc_prev DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS startup_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    check_name VARCHAR(48),
    passed BOOLEAN,
    detail VARCHAR(255) DEFAULT ''
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(48),
    pair VARCHAR(16),
    start_date DATE,
    end_date DATE,
    trades INTEGER,
    win_rate DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    passed BOOLEAN,
    detail_json TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS shadow_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    live_trade_id INTEGER,
    pair VARCHAR(16),
    direction VARCHAR(8),
    intended_entry DOUBLE PRECISION,
    live_fill DOUBLE PRECISION,
    shadow_fill DOUBLE PRECISION,
    shadow_status VARCHAR(16) DEFAULT 'open',
    shadow_exit DOUBLE PRECISION,
    shadow_pnl DOUBLE PRECISION DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ml_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    n_trades INTEGER,
    train_accuracy DOUBLE PRECISION,
    test_accuracy DOUBLE PRECISION,
    features_json TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS ab_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    name VARCHAR(48),
    variant VARCHAR(8),
    params_json TEXT DEFAULT '{}',
    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    promoted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS session_heatmap (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(48),
    hour INTEGER,
    day_of_week INTEGER,
    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    win_rate DOUBLE PRECISION DEFAULT 0,
    blacklisted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS monte_carlo_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    simulations INTEGER,
    prob_dd10 DOUBLE PRECISION,
    prob_dd20 DOUBLE PRECISION,
    prob_dd30 DOUBLE PRECISION,
    prob_ruin DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS withdrawals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    amount_usd DOUBLE PRECISION,
    amount_kes DOUBLE PRECISION,
    exchange_rate DOUBLE PRECISION,
    kind VARCHAR(16) DEFAULT 'weekly',
    status VARCHAR(16) DEFAULT 'submitted',
    authorization VARCHAR(48) DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    category VARCHAR(32),
    action VARCHAR(64),
    detail TEXT DEFAULT '',
    source VARCHAR(32) DEFAULT 'system'
);

CREATE TABLE IF NOT EXISTS system_state (
    key VARCHAR(64) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    version VARCHAR(8),
    prompt TEXT,
    trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT FALSE
);
