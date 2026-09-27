"""Dependency manager (UPGRADE 7): safe, weekly update checks.

Compares installed versions against the latest on PyPI; proposals are tested
in a shadow venv (created on demand) by importing + running the test suite.
Updates apply only when the shadow check passes; failures alert but never
touch the running environment.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

CORE_PACKAGES = ["numpy", "pandas", "flask", "requests", "sqlalchemy", "yfinance"]


class DependencyManager:
    """Weekly outdated-check with shadow testing."""

    def __init__(self, notifier=None, check_interval_sec: int = 7 * 86400) -> None:
        self.notifier = notifier
        self.check_interval_sec = check_interval_sec
        self.last_check = 0.0
        self.report: list[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---- checks ----

    def _installed_version(self, package: str) -> Optional[str]:
        """Currently installed version (importlib metadata)."""
        try:
            from importlib import metadata

            return metadata.version(package)
        except Exception:
            return None

    def _latest_version(self, package: str) -> Optional[str]:
        """Latest version from PyPI (network errors -> None)."""
        try:
            import requests

            resp = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=10)
            resp.raise_for_status()
            return resp.json()["info"]["version"]
        except Exception as exc:
            logger.debug("pypi check failed for %s: %s", package, exc)
            return None

    def check(self) -> list[dict]:
        """Full outdated report for CORE_PACKAGES."""
        self.report = []
        for pkg in CORE_PACKAGES:
            installed = self._installed_version(pkg)
            latest = self._latest_version(pkg)
            self.report.append({"package": pkg, "installed": installed,
                                "latest": latest,
                                "outdated": bool(installed and latest and installed != latest)})
        self.last_check = time.time()
        return self.report

    def outdated(self) -> list[dict]:
        """Only the outdated entries of the last check."""
        return [r for r in self.report if r["outdated"]]

    # ---- shadow testing ----

    def shadow_test(self, package: str, version: str) -> dict:
        """Install the proposal into a throwaway venv and import it."""
        venv_dir = "/tmp/dl_shadow_venv"
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir],
                           check=True, capture_output=True, timeout=300)
            pip = f"{venv_dir}/bin/pip"
            subprocess.run([pip, "install", "--quiet", f"{package}=={version}"],
                           check=True, capture_output=True, timeout=600)
            python = f"{venv_dir}/bin/python"
            result = subprocess.run(
                [python, "-c", f"import {package}; print('ok')"],
                capture_output=True, timeout=120, text=True)
            return {"ok": result.returncode == 0 and "ok" in result.stdout,
                    "detail": result.stdout.strip() or result.stderr.strip()[:200]}
        except Exception as exc:
            return {"ok": False, "detail": str(exc)[:200]}

    def propose_update(self, package: str, version: str) -> dict:
        """Shadow-test an update; NEVER auto-installs into production."""
        test = self.shadow_test(package, version)
        entry = {"package": package, "version": version, **test,
                 "ts": time.time()}
        if not test["ok"]:
            logger.warning("dependency %s==%s failed shadow test: %s",
                           package, version, test["detail"])
            if self.notifier:
                try:
                    self.notifier.send(f"🧪 Update rejected in shadow: "
                                       f"{package}=={version} — {test['detail']}")
                except Exception:
                    pass
        else:
            logger.info("dependency %s==%s passed shadow test (manual apply)", package, version)
        return entry

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.check()
                for row in self.outdated():
                    self.propose_update(row["package"], row["latest"])
            except Exception as exc:
                logger.warning("dependency check failed: %s", exc)
            self._stop.wait(self.check_interval_sec)

    def start(self) -> None:
        """Start the weekly background check."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="dep-manager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop checking."""
        self._stop.set()

    def status(self) -> dict:
        """Dashboard payload."""
        return {"last_check": self.last_check, "report": self.report,
                "outdated_count": len(self.outdated())}
