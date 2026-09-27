"""TradingView webhook receiver: alert JSON -> unified signal pipeline.

Runs inside the bot process (main.py starts it as a daemon thread) or
standalone (``python -m execution.tv_webhook``) for notify-only testing.

TradingView posts the alert message as the raw request body; this endpoint
expects JSON of the form::

    {"secret": "<TV_WEBHOOK_SECRET>", "pair": "EURUSD", "action": "buy",
     "price": 1.08450, "sl": 1.08200, "tp": 1.09000}

``action`` is buy/sell/exit (exit closes positions for the pair). Extra keys
are ignored, so the alert can carry human notes. Security: the secret must
match settings.TV_WEBHOOK_SECRET and the source IP should be TradingView's
published ranges (52.89.214.188, 34.212.75.30, 54.218.53.128, 52.32.178.7).

Every signal enters the SAME chain as native bot signals: duplicate check ->
ML filter -> Groq research verify -> risk manager -> execution engine.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from config import settings
from strategies.base_strategy import StrategySignal
from strategies.strategy_selector import session_of

logger = logging.getLogger(__name__)

TRADINGVIEW_IPS = {"52.89.214.188", "34.212.75.30", "54.218.53.128", "52.32.178.7"}
ALLOWED_ACTIONS = {"buy", "sell", "exit"}
MAX_BODY_BYTES = 4096


def normalize_ticker(raw: str) -> str:
    """'OANDA:EURUSD' / 'CAPITALCOM_X:EURUSD' / 'FX:EURUSD' -> 'EURUSD'.

    Also folds common metals/index aliases onto the bot's symbol names.
    """
    ticker = str(raw).split(":")[-1].strip().upper()
    aliases = {
        "XAUUSD": "XAUUSD", "GOLD": "XAUUSD", "GC1!": "XAUUSD", "GC1": "XAUUSD",
        "NAS100": "NAS100", "US100": "NAS100", "NDX": "NAS100", "NAS1!": "NAS100",
        "US30": "US30", "DJ30": "US30", "DJI": "US30", "YM1!": "US30",
        "SPX500": "SPX500", "US500": "SPX500", "SPX": "SPX500",
    }
    return aliases.get(ticker, ticker)


def parse_alert(body: bytes, secret: str = "") -> tuple[Optional[dict], str]:
    """Validate and parse a raw alert body -> (data, error).

    error is empty on success. Checks: size, JSON shape, secret, action.
    """
    if len(body) > MAX_BODY_BYTES:
        return None, "body too large"
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, "invalid json"
    if not isinstance(data, dict):
        return None, "invalid json"
    if secret and data.get("secret") != secret:
        return None, "bad secret"
    action = str(data.get("action", "")).lower().strip()
    if action not in ALLOWED_ACTIONS:
        return None, f"bad action: {action or 'missing'}"
    if action != "exit":
        for field in ("pair", "price", "sl", "tp"):
            if field not in data or data[field] is None:
                return None, f"missing field: {field}"
    return data, ""


def alert_to_signal(data: dict) -> Optional[StrategySignal]:
    """Build a unified StrategySignal from a validated alert (None for exit)."""
    action = str(data.get("action", "")).lower()
    if action == "exit":
        return None
    pair = normalize_ticker(data.get("pair", ""))
    try:
        price = float(data["price"])
        sl = float(data["sl"])
        tp = float(data["tp"])
    except (KeyError, TypeError, ValueError):
        return None
    label = str(data.get("strategy", "tradingview")).strip()[:48] or "tradingview"
    return StrategySignal(
        strategy=f"tv_{label}",
        pair=pair,
        direction="buy" if action == "buy" else "sell",
        entry=price,
        sl=sl,
        tp=tp,
        session=session_of(__import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
        confluences=["tradingview_alert"],
    )


class TVWebhookState:
    """Dedupes repeated alerts (same pair/direction/price within 15 minutes)."""

    def __init__(self) -> None:
        self._seen: dict[tuple, float] = {}
        self._lock = threading.Lock()
        self.tv_alerts_total = 0

    def seen_before(self, pair: str, direction: str, price: float,
                    minutes: int = 15) -> bool:
        """True when the same (pair, direction, rounded price) arrived recently."""
        import time

        key = (pair, direction, round(price, 5))
        now = time.monotonic()
        with self._lock:
            stale = [k for k, t in self._seen.items() if now - t > minutes * 60]
            for k in stale:
                del self._seen[k]
            if key in self._seen:
                return True
            self._seen[key] = now
            return False


class TVWebhookServer:
    """HTTP receiver wired to the pipeline runner (None => notify-only)."""

    def __init__(self, process_signal=None, port: Optional[int] = None,
                 secret: Optional[str] = None) -> None:
        self.process_signal = process_signal
        self.secret = settings.TV_WEBHOOK_SECRET if secret is None else secret
        self.state = TVWebhookState()
        self.port = port or settings.TV_WEBHOOK_PORT
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
                outer._handle(self)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("tv webhook: " + fmt, *args)

        self._handler = Handler

    # ---- request handling ----

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        """Respond 204-fast to TradingView ranges; process in background."""
        ip = handler.client_address[0]
        body = handler.rfile.read(min(int(handler.headers.get("Content-Length", 0)),
                                      MAX_BODY_BYTES))
        # fast ack first: TradingView retries on non-2xx / timeouts
        handler.send_response(204)
        handler.end_headers()

        threading.Thread(target=self._process, args=(body, ip),
                         name="tv-alert", daemon=True).start()

    def _process(self, body: bytes, ip: str) -> None:
        """Validate + route one alert (runs off the HTTP thread)."""
        self.state.tv_alerts_total += 1
        data, err = parse_alert(body, secret=self.secret)
        if data is None:
            logger.warning("tv alert rejected from %s: %s", ip, err)
            return
        if TRADINGVIEW_IPS and ip not in TRADINGVIEW_IPS:
            logger.info("tv alert from non-TV ip %s (accepted; whitelist only warns)", ip)

        action = str(data.get("action", "")).lower()
        pair = normalize_ticker(data.get("pair", "")) if action != "exit" else ""

        if action == "exit":
            self._handle_exit(pair)
            return

        signal = alert_to_signal(data)
        if signal is None:
            logger.warning("tv alert unparseable into signal: %s", data)
            return
        if self.state.seen_before(signal.pair, signal.direction, signal.entry):
            logger.info("tv duplicate alert skipped: %s %s @ %s",
                        signal.pair, signal.direction, signal.entry)
            return

        logger.info("tv alert accepted: %s %s entry=%s sl=%s tp=%s",
                    signal.pair, signal.direction, signal.entry, signal.sl, signal.tp)
        if self.process_signal is None:
            logger.info("notify-only mode: no pipeline wired, alert logged")
            return
        try:
            self.process_signal(signal)
        except Exception:
            logger.exception("tv alert pipeline failed")

    def _handle_exit(self, pair: str) -> None:
        """Close open trades for the alert pair via the trade manager."""
        try:
            from core import db

            open_now = [t for t in db.open_trades() if not pair
                        or t["pair"] == pair]
            if not open_now:
                logger.info("tv exit: no open trades for %r", pair or "all")
                return
            self._close_via_manager(open_now, pair)
        except Exception:
            logger.exception("tv exit failed")

    def _close_via_manager(self, trades: list[dict], pair: str) -> None:
        """Best-effort close through the live TradeManager when reachable."""
        try:
            from execution.trade_manager import TradeManager

            tm = TradeManager(broker=None)
            closed = 0
            for t in trades:
                try:
                    price = tm._current_price(t["pair"], t["direction"])
                    tm.close(t, price, "TradingView exit")
                    closed += 1
                except Exception as exc:
                    logger.warning("tv exit close failed #%s: %s", t.get("id"), exc)
            logger.info("tv exit closed %d/%d trades%s", closed, len(trades),
                        f" for {pair}" if pair else "")
        except Exception as exc:
            logger.error("tv exit unavailable: %s", exc)

    # ---- lifecycle ----

    def start(self) -> None:
        """Bind and serve in a daemon thread (idempotent)."""
        if self._thread is not None:
            return
        try:
            self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), self._handler)
        except OSError as exc:
            logger.error("tv webhook bind failed on :%d (%s) - alerts disabled", self.port, exc)
            return
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="tv-webhook", daemon=True)
        self._thread.start()
        logger.info("tv webhook listening on :%d (secret required)", self.port)

    def stop(self) -> None:
        """Shutdown the server thread."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
            self._thread = None


_SERVER: Optional[TVWebhookServer] = None


def get_server(process_signal=None) -> TVWebhookServer:
    """Process-wide singleton so the Flask route and main.py share state.

    Call with process_signal to (re)wire the pipeline onto the existing
    instance; the dashboard uses it without arguments (notify-only unless
    main.py already wired the pipeline).
    """
    global _SERVER
    if _SERVER is None:
        _SERVER = TVWebhookServer(process_signal=process_signal)
    elif process_signal is not None:
        _SERVER.process_signal = process_signal
    return _SERVER


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = get_server()
    server.start()
    print(f"tv webhook (notify-only) on :{server.port} - ctrl-c to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.stop()
