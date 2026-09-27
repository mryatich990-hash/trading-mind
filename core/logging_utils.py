"""Logging with automatic secret redaction."""

import logging
import os
import sys

_CONFIGURED = False
_SECRET_NAMES = ("KEY", "PASSWORD", "TOKEN", "SECRET")


def get_logger(name: str) -> logging.Logger:
    """Return a logger; installs a redacting stdout handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

        class _Redact(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    msg = record.getMessage()
                    for key, val in os.environ.items():
                        if any(s in key.upper() for s in _SECRET_NAMES) and val and len(val) > 8:
                            msg = msg.replace(val, "***REDACTED***")
                    record.msg = msg
                    record.args = None
                except Exception:
                    pass
                return True

        handler.addFilter(_Redact())
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)
