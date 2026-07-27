"""Parent-liveness guard for externally supervised child processes.

The MyAgents Desktop supervisor spawns `hermes serve` and
`hermes gateway run --external-supervisor` as a Profile Runtime Unit that
must die with it (design: MyAgents channel-desktop-shared-session-runtime.md
§5.6/§11.1). A plain parent/child relationship does not guarantee that: when
the Electron main process is force-killed, its children are reparented and
keep running as orphans.

When the supervisor declares itself via ``HERMES_PARENT_PID``, a daemon
thread polls that PID's liveness (signal 0) and hard-exits the process once
the parent is gone. Mirrors the ``_parent_guard_loop`` precedent in
``tui_gateway/compute_host.py``. No-op when the variable is absent, so
nothing changes for systemd/launchd/interactive runs.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

PARENT_PID_ENV = "HERMES_PARENT_PID"
_POLL_INTERVAL_S = 2.0

_started = threading.Lock()
_guard_thread: Optional[threading.Thread] = None


def _parent_alive(parent_pid: int) -> bool:
    try:
        os.kill(parent_pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by someone else — treat as alive; the supervisor
        # normally runs as the same user, so this should not happen.
        return True
    except OSError:
        return True


def start_parent_liveness_guard(role: str = "child") -> Optional[threading.Thread]:
    """Start the guard if ``HERMES_PARENT_PID`` is set. Idempotent."""
    global _guard_thread
    raw = os.environ.get(PARENT_PID_ENV, "").strip()
    if not raw:
        return None
    try:
        parent_pid = int(raw)
    except ValueError:
        return None
    if parent_pid <= 1:
        return None

    with _started:
        if _guard_thread is not None and _guard_thread.is_alive():
            return _guard_thread

        def _watch() -> None:
            while True:
                time.sleep(_POLL_INTERVAL_S)
                if not _parent_alive(parent_pid):
                    try:
                        print(
                            f"[hermes:{role}] supervising parent pid={parent_pid} is gone; exiting",
                            file=sys.stderr,
                            flush=True,
                        )
                    except Exception:
                        pass
                    # Hard exit: the owner is dead, nothing can drain us and a
                    # graceful shutdown could wedge on a dead pipe. State is
                    # recovered by the owner's next-start orphan cleanup.
                    os._exit(0)

        _guard_thread = threading.Thread(
            target=_watch, name="hermes-parent-liveness", daemon=True
        )
        _guard_thread.start()
        return _guard_thread
