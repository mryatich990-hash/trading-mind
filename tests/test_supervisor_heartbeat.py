"""Regression: supervisor must keep the Supabase engine_heartbeat fresh.

The Next.js dashboard (and health.py's DB check) read engine_heartbeat
from system_state. run_engine._write_heartbeat referenced ROOT defined
only inside main() -> NameError swallowed by `except: pass` -> the key
went stale for 3.25 days while /health (file beats) still said alive,
and every dashboard showed OFFLINE during live trading.
"""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SUPERVISOR = ROOT / "deploy" / "run_engine.py"


class TestSupervisorHeartbeat:
    def test_no_nameerror_trap_in_write_heartbeat(self):
        """ROOT must be module-level: _write_heartbeat runs before main()."""
        tree = ast.parse(SUPERVISOR.read_text(encoding="utf-8"))

        module_names = set()
        func_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_names.add(node.name)
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name):
                                func_names.add(t.id)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        module_names.add(t.id)

        assert "ROOT" in module_names, (
            "ROOT must be defined at module level so _write_heartbeat can use it"
        )
        assert "_write_heartbeat" in func_names

    def test_heartbeat_failures_are_logged_not_swallowed(self):
        """A silent except:pass hid the failure for 3 days."""
        src = SUPERVISOR.read_text(encoding="utf-8")
        fn_src = src.split("def _write_heartbeat", 1)[1].split("\ndef ", 1)[0]
        assert "print" in fn_src and "failed" in fn_src, (
            "heartbeat write failures must print a diagnostic line"
        )

    def test_heartbeat_actually_writes_the_state_key(self, temp_db, monkeypatch):
        """Calling _write_heartbeat refreshes engine_heartbeat in system_state."""
        import importlib
        import sys as _sys

        real_run_engine = _sys.modules.pop("deploy.run_engine", None)
        try:
            monkeypatch.syspath_prepend(str(ROOT))
            mod = importlib.import_module("deploy.run_engine")
            from core import db

            assert db.get_state("engine_heartbeat", "") in ("", None)
            mod._write_heartbeat()
            ts = db.get_state("engine_heartbeat", "")
            assert ts, "engine_heartbeat must be written to system_state"
        finally:
            if real_run_engine is not None:
                _sys.modules["deploy.run_engine"] = real_run_engine

    def test_dashboard_reads_this_key_so_it_must_stay_named_the_same(self, temp_db):
        from core import db

        db.set_state("engine_heartbeat", "2026-10-01T00:00:00+00:00")
        assert db.get_state("engine_heartbeat", "") == "2026-10-01T00:00:00+00:00"
