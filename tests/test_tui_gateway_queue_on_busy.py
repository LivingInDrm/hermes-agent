"""Per-session turn FIFO: every mid-turn input keeps its own turn identity.

The old single ``queued_prompt`` slot merged a second arrival with ``\\n\\n``;
the shared session runtime replaces it with a bounded in-memory FIFO
(design: MyAgents channel-desktop-shared-session-runtime.md §8.1):

- inputs are never merged, overwritten or silently dropped;
- over-capacity is an explicit retryable rejection (4290);
- accept order is the per-session ``sequence`` allocated under the lock;
- generation-local clientTurnId dedup answers retries with the original turn;
- legacy TUI submits (no client_turn_id) keep the busy_input_mode policy;
  identified submits always take the plain FIFO path.
"""

import threading
import time
import types

import pytest

from tui_gateway import server
from tui_gateway.turn_fifo import (
    TURN_QUEUE_MAX_PENDING,
    GenerationDedupTable,
    SessionTurnState,
)


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


@pytest.fixture(autouse=True)
def _fresh_dedup(monkeypatch):
    monkeypatch.setattr(server, "_turn_dedup", GenerationDedupTable())


def _record(session, text, client_id="", transport=None):
    return server._new_turn_record(
        session, text, sequence=0, client_turn_id=client_id, transport=transport
    )


def _queued_texts(session):
    state = session.get("turn_state")
    return [r.text for r in state.queue] if isinstance(state, SessionTurnState) else []


# ── FIFO admission ─────────────────────────────────────────────────────────

def test_busy_submits_stay_independent_turns(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(running=True)

    first = server._handle_busy_submit("r1", "sid", session, _record(session, "first", transport="ws-1"))
    second = server._handle_busy_submit("r2", "sid", session, _record(session, "second", transport="ws-2"))

    assert first["result"]["status"] == "queued"
    assert second["result"]["status"] == "queued"
    # No merging — two independent turns with distinct identities and
    # monotonically increasing accept sequences.
    assert _queued_texts(session) == ["first", "second"]
    assert first["result"]["turn_id"] != second["result"]["turn_id"]
    assert first["result"]["sequence"] < second["result"]["sequence"]
    assert second["result"]["queue_position"] == 2


def test_queue_full_is_explicit_retryable_rejection(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(running=True)
    for index in range(TURN_QUEUE_MAX_PENDING):
        resp = server._handle_busy_submit("r", "sid", session, _record(session, f"m{index}"))
        assert resp["result"]["status"] == "queued"

    overflow = server._handle_busy_submit("r-full", "sid", session, _record(session, "one too many"))
    assert overflow["error"]["code"] == 4290
    assert "retry" in overflow["error"]["message"]
    # Nothing was merged/dropped to make room.
    assert len(_queued_texts(session)) == TURN_QUEUE_MAX_PENDING


def test_queue_full_releases_dedup_claim_so_retry_can_enter(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(running=True)
    for index in range(TURN_QUEUE_MAX_PENDING):
        server._handle_busy_submit("r", "sid", session, _record(session, f"m{index}"))

    rejected = _record(session, "retry me", client_id="client-1")
    assert server._claim_turn_dedup(rejected) is None
    overflow = server._handle_busy_submit("r-full", "sid", session, rejected)
    assert overflow["error"]["code"] == 4290

    # The rejected claim was released: after room frees up, the same
    # client_turn_id may enter the queue instead of being reported duplicate.
    with session["history_lock"]:
        session["turn_state"].queue.popleft()
    retry = _record(session, "retry me", client_id="client-1")
    assert server._claim_turn_dedup(retry) is None
    accepted = server._handle_busy_submit("r-retry", "sid", session, retry)
    assert accepted["result"]["status"] == "queued"


# ── generation-local dedup ─────────────────────────────────────────────────

def test_duplicate_client_turn_id_returns_original_turn():
    session = _session(running=True)
    original = _record(session, "hello", client_id="client-7")
    assert server._claim_turn_dedup(original) is None

    retry = _record(session, "hello", client_id="client-7")
    existing = server._claim_turn_dedup(retry)
    assert existing is original

    resp = server._duplicate_turn_response("r2", session, existing)
    assert resp["result"]["duplicate"] is True
    assert resp["result"]["turn_id"] == original.turn_id
    assert resp["result"]["durability"] == "process"


def test_dedup_scoped_by_lineage():
    session_a = _session(session_key="lineage-a")
    session_b = _session(session_key="lineage-b")
    assert server._claim_turn_dedup(_record(session_a, "x", client_id="same-id")) is None
    # Same client id in a different conversation is a different turn.
    assert server._claim_turn_dedup(_record(session_b, "x", client_id="same-id")) is None


# ── busy_input_mode policy (legacy TUI submits only) ───────────────────────

def test_busy_interrupt_mode_interrupts_and_queues(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, _record(session, "redirect", transport="ws-1"))

    assert resp["result"]["status"] == "queued"
    deadline = time.monotonic() + 1
    while calls["interrupt"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls["interrupt"] == 1
    assert _queued_texts(session) == ["redirect"]


def test_busy_queue_mode_queues_without_interrupting(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, _record(session, "later"))

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 0
    assert _queued_texts(session) == ["later"]


def test_busy_steer_mode_injects_when_accepted(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: True, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, _record(session, "nudge"))

    assert resp["result"]["status"] == "steered"
    assert _queued_texts(session) == []


def test_busy_steer_mode_falls_back_to_queue_when_rejected(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: False, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, _record(session, "nudge"))

    assert resp["result"]["status"] == "queued"
    assert _queued_texts(session) == ["nudge"]


def test_identified_submit_never_steers_or_interrupts(monkeypatch):
    """A normal identified submit is FIFO-only: no implicit steer/interrupt
    (design §8.1/§10.2 — steering is an explicit separate control)."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    calls = {"steer": 0, "interrupt": 0}
    agent = types.SimpleNamespace(
        steer=lambda text: calls.__setitem__("steer", calls["steer"] + 1) or True,
        interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1),
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, _record(session, "next turn", client_id="c1"))

    assert resp["result"]["status"] == "queued"
    time.sleep(0.05)
    assert calls == {"steer": 0, "interrupt": 0}


def test_busy_interrupt_does_not_hold_history_lock_or_delay_queue(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    interrupt_started = threading.Event()
    release_interrupt = threading.Event()

    def blocking_interrupt():
        interrupt_started.set()
        release_interrupt.wait(timeout=2)

    session = _session(
        agent=types.SimpleNamespace(interrupt=blocking_interrupt),
        running=True,
    )

    started = time.monotonic()
    resp = server._handle_busy_submit("r1", "sid", session, _record(session, "keep this", transport="ws-1"))

    assert resp["result"]["status"] == "queued"
    assert time.monotonic() - started < 0.25
    assert _queued_texts(session) == ["keep this"]
    assert interrupt_started.wait(timeout=1)
    assert session["history_lock"].acquire(timeout=0.25)
    session["history_lock"].release()
    release_interrupt.set()


def test_busy_helper_retries_when_turn_finished(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    session = _session(running=False)

    assert server._handle_busy_submit("r1", "sid", session, _record(session, "run now")) is None
    assert _queued_texts(session) == []


# ── FIFO drain ─────────────────────────────────────────────────────────────

def _enqueue(session, text, transport=None):
    record = _record(session, text, transport=transport)
    with session["history_lock"]:
        record.sequence = server._turn_state(session).next_sequence()
        server._turn_state(session).enqueue(record)
    return record


def test_drain_promotes_head_and_claims_running(monkeypatch):
    fired = {}
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda rid, sid, session, text: fired.update(rid=rid, sid=sid, text=text),
    )
    session = _session()
    first = _enqueue(session, "go", transport="ws-9")
    _enqueue(session, "after", transport="ws-2")

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert fired == {"rid": "r1", "sid": "sid", "text": "go"}
    assert session["running"] is True
    assert session["transport"] == "ws-9"
    assert first.state == "running"
    # Second stays intact in FIFO order for the next drain.
    assert _queued_texts(session) == ["after"]
    # The drained turn's user text is visible as the inflight projection.
    assert session["inflight_turn"]["user"] == "go"


def test_drain_noop_when_nothing_queued(monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fire")))
    session = _session()
    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert session["running"] is False


def test_drain_noop_when_session_already_running(monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fire")))
    session = _session(running=True)
    _enqueue(session, "go")
    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert _queued_texts(session) == ["go"]


def test_drain_releases_running_and_fails_turn_on_dispatch_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("dispatch failed")
    monkeypatch.setattr(server, "_run_prompt_submit", _boom)
    session = _session()
    record = _enqueue(session, "go")

    assert server._drain_queued_prompt("r1", "sid", session) is True
    # Failure must not leave the session wedged as running, and the turn's
    # terminal state is explicit.
    assert session["running"] is False
    assert record.state == "failed"


# ── terminal turn states ───────────────────────────────────────────────────

def _activate(session, text):
    record = _record(session, text)
    with session["history_lock"]:
        record.sequence = server._turn_state(session).next_sequence()
        server._turn_state(session).activate(record)
    return record


def test_finish_active_turn_completed_by_default():
    session = _session()
    record = _activate(session, "work")
    server._finish_active_turn("sid", session)
    assert record.state == "completed"


def test_finish_active_turn_interrupted_when_cancelled():
    session = _session()
    record = _activate(session, "work")
    session["_turn_cancel_requested"] = True
    server._finish_active_turn("sid", session)
    assert record.state == "interrupted"


def test_finish_active_turn_failed_on_turn_failure_flag():
    session = _session()
    record = _activate(session, "work")
    session["_turn_failed"] = True
    server._finish_active_turn("sid", session)
    assert record.state == "failed"
    assert "_turn_failed" not in session


# ── directed interrupt (design §10.2) ──────────────────────────────────────

def test_legacy_interrupt_clears_queue_but_directed_keeps_it():
    session = _session(running=True)
    _activate(session, "active work")
    queued = _enqueue(session, "waiting")

    # Directed at the active turn: FIFO stays intact.
    active_id = session["turn_state"].active.turn_id
    server._drop_queued_turns_for_interrupt("sid", session, {"turn_id": active_id})
    assert _queued_texts(session) == ["waiting"]

    # Legacy un-scoped /stop keeps its historical clear-everything meaning.
    server._drop_queued_turns_for_interrupt("sid", session, {})
    assert _queued_texts(session) == []
    assert queued.state == "interrupted"


def test_interrupt_stale_turn_and_generation_are_rejected():
    session = _session(running=True)
    _activate(session, "active work")
    ok = server._validate_interrupt_target("r1", session, {"turn_id": session["turn_state"].active.turn_id})
    assert ok is None

    stale_turn = server._validate_interrupt_target("r2", session, {"turn_id": "turn-not-active"})
    assert stale_turn["error"]["code"] == 4033

    stale_generation = server._validate_interrupt_target(
        "r3", session, {"runtime_generation": "gen-from-before-the-restart"}
    )
    assert stale_generation["error"]["code"] == 4033
