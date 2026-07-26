"""Parent-liveness guard for externally supervised serve/gateway children."""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from hermes_cli import parent_liveness


def test_noop_without_env(monkeypatch):
    monkeypatch.delenv(parent_liveness.PARENT_PID_ENV, raising=False)
    assert parent_liveness.start_parent_liveness_guard() is None


def test_noop_for_invalid_or_init_pid(monkeypatch):
    monkeypatch.setenv(parent_liveness.PARENT_PID_ENV, "not-a-pid")
    assert parent_liveness.start_parent_liveness_guard() is None
    monkeypatch.setenv(parent_liveness.PARENT_PID_ENV, "1")
    assert parent_liveness.start_parent_liveness_guard() is None


def test_parent_alive_probe():
    assert parent_liveness._parent_alive(os.getpid()) is True
    # A PID from a process we just reaped is gone.
    probe = subprocess.Popen([sys.executable, "-c", "pass"])
    probe.wait()
    # PID reuse is theoretically possible but not within this test's window.
    assert parent_liveness._parent_alive(probe.pid) is False


def test_child_exits_when_declared_parent_dies(tmp_path):
    """End-to-end: a child guarding on a fake parent exits once it dies."""
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    child_code = textwrap.dedent(
        f"""
        import os, sys, time
        sys.path.insert(0, {os.getcwd()!r})
        os.environ["HERMES_PARENT_PID"] = "{parent.pid}"
        from hermes_cli.parent_liveness import start_parent_liveness_guard, _POLL_INTERVAL_S
        assert start_parent_liveness_guard(role="test") is not None
        time.sleep(60)
        """
    )
    child = subprocess.Popen([sys.executable, "-c", child_code])
    try:
        time.sleep(0.5)
        assert child.poll() is None  # parent alive → child keeps running
        parent.kill()
        parent.wait()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.2)
        assert child.poll() is not None, "child did not exit after parent death"
        assert child.returncode == 0
    finally:
        for proc in (parent, child):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
