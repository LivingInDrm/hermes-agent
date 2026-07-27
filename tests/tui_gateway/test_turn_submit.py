"""turn.submit — the transport-neutral shared-session-runtime entry (design §7).

Covers: prepared-turn validation, conversation resolution to the single live
lineage actor, delegation into the same coordinator prompt.submit uses,
generation-local dedup, FIFO admission and the HermesTurnEnqueued response
shape (durability=process, duplicate flag, turn identity + sequence).
"""

import threading
import types
from unittest.mock import MagicMock, patch

import pytest

_ORIGINAL_METHODS: dict = {}


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")
        if not _ORIGINAL_METHODS:
            _ORIGINAL_METHODS.update(mod._methods)
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()
        mod._methods.clear()
        mod._methods.update(_ORIGINAL_METHODS)
        from tui_gateway.turn_fifo import GenerationDedupTable

        mod._turn_dedup = GenerationDedupTable()


def _session(server, sid, session_key="stored-1", running=False, agent=None):
    record = {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": running,
        "transport": None,
        "attached_images": [],
        "created_at": 0.0,
    }
    server._sessions[sid] = record
    return record


def _submit(server, **overrides):
    params = {
        "client_turn_id": "client-1",
        "message": "hello",
        "conversation": {"kind": "stored", "stored_session_id": "stored-1"},
        "delivery_mode": "desktop-only",
        "busy_mode": "fifo",
    }
    params.update(overrides)
    return server.handle_request({"id": "t1", "method": "turn.submit", "params": params})


# ── prepared-turn validation ───────────────────────────────────────────────

def test_rejects_missing_client_turn_id(server):
    resp = _submit(server, client_turn_id="")
    assert resp["error"]["code"] == 4032


def test_rejects_empty_message(server):
    resp = _submit(server, message="   ")
    assert resp["error"]["code"] == 4032


def test_rejects_unknown_delivery_mode(server):
    resp = _submit(server, delivery_mode="broadcast")
    assert resp["error"]["code"] == 4032


def test_rejects_non_fifo_busy_mode(server):
    resp = _submit(server, busy_mode="interrupt")
    assert resp["error"]["code"] == 4032


def test_rejects_profile_mismatch(server, monkeypatch):
    monkeypatch.setattr(server, "_current_profile_name", lambda: "alpha")
    resp = _submit(server, profile="bravo")
    assert resp["error"]["code"] == 4032
    assert "alpha" in resp["error"]["message"]


def test_channel_create_requires_gateway_runtime_client(server):
    resp = _submit(server, conversation={
        "kind": "channel-create", "source": "feishu", "source_session_key": "feishu:chat:1",
    })
    assert resp["error"]["code"] == 4031


def test_channel_delivery_requires_gateway_runtime_client(server):
    _session(server, "sid-1")
    assert _submit(server, delivery_mode="origin-channel")["error"]["code"] == 4031
    # A desktop credential must not be able to smuggle a trusted channel
    # identity into the execution context (design §7.2).
    assert _submit(server, trusted_source={"platform": "feishu"})["error"]["code"] == 4031
    assert _submit(server, delivery_sink_id="sink-1")["error"]["code"] == 4031


# ── conversation resolution ────────────────────────────────────────────────

def test_live_session_resolved_by_stored_ref_and_delegated(server, monkeypatch):
    _session(server, "sid-live", session_key="stored-1")
    seen = {}

    def fake_prompt_submit(rid, params):
        seen.update(params)
        return server._ok(rid, {
            "status": "streaming", "turn_id": "turn-x", "sequence": 3,
            "queue_position": 0, "lineage_id": "stored-1",
            "stored_session_id": "stored-1",
        })

    monkeypatch.setitem(server._methods, "prompt.submit", fake_prompt_submit)
    resp = _submit(server)
    assert seen["session_id"] == "sid-live"
    assert seen["client_turn_id"] == "client-1"
    result = resp["result"]
    assert result["enqueued"] is True
    assert result["durability"] == "process"
    assert result["duplicate"] is False
    assert result["runtime_session_id"] == "sid-live"
    assert result["runtime_generation"] == server.RUNTIME_GENERATION
    assert result["turn_id"] == "turn-x"
    assert result["sequence"] == 3


def test_old_segment_ref_resolves_to_current_tip_actor(server, monkeypatch):
    """An old compression segment id and the current tip land on ONE actor."""
    _session(server, "sid-live", session_key="tip-2")

    db = types.SimpleNamespace(
        get_session=lambda sid: {"id": sid} if sid in ("old-1", "tip-2") else None,
        resolve_resume_session_id=lambda sid: "tip-2",
    )
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setitem(server._methods, "prompt.submit", lambda rid, params: server._ok(rid, {
        "status": "streaming", "turn_id": "turn-y", "sequence": 1,
        "queue_position": 0, "lineage_id": "old-1", "stored_session_id": "tip-2",
    }))

    resp = _submit(server, conversation={"kind": "stored", "stored_session_id": "old-1"})
    assert resp["result"]["runtime_session_id"] == "sid-live"


def test_unknown_stored_ref_is_explicit_error(server, monkeypatch):
    db = types.SimpleNamespace(
        get_session=lambda sid: None,
        resolve_resume_session_id=lambda sid: sid,
    )
    monkeypatch.setattr(server, "_get_db", lambda: db)
    resp = _submit(server, conversation={"kind": "stored", "stored_session_id": "ghost"})
    assert resp["error"]["code"] == 4007


def test_cold_ref_resumes_via_session_resume(server, monkeypatch):
    db = types.SimpleNamespace(
        get_session=lambda sid: {"id": sid} if sid == "cold-1" else None,
        resolve_resume_session_id=lambda sid: "cold-1",
    )
    monkeypatch.setattr(server, "_get_db", lambda: db)

    def fake_resume(rid, params):
        assert params["session_id"] == "cold-1"
        assert params["close_on_disconnect"] is False
        _session(server, "sid-cold", session_key="cold-1")
        return server._ok(rid, {"session_id": "sid-cold"})

    monkeypatch.setitem(server._methods, "session.resume", fake_resume)
    monkeypatch.setitem(server._methods, "prompt.submit", lambda rid, params: server._ok(rid, {
        "status": "streaming", "turn_id": "turn-z", "sequence": 1,
        "queue_position": 0, "lineage_id": "cold-1", "stored_session_id": "cold-1",
    }))

    resp = _submit(server, conversation={"kind": "stored", "stored_session_id": "cold-1"})
    assert resp["result"]["runtime_session_id"] == "sid-cold"


# ── FIFO + dedup through the real coordinator ──────────────────────────────

def test_busy_submit_enters_fifo_and_duplicate_returns_original(server, monkeypatch):
    _session(server, "sid-busy", session_key="stored-1", running=True)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")

    first = _submit(server)
    assert first["result"]["status"] == "queued"
    assert first["result"]["enqueued"] is True
    assert first["result"]["queue_position"] == 1
    turn_id = first["result"]["turn_id"]

    # Same client_turn_id retried: no second turn, original identity returned.
    retry = _submit(server)
    assert retry["result"]["duplicate"] is True
    assert retry["result"]["turn_id"] == turn_id
    state = server._sessions["sid-busy"]["turn_state"]
    assert len(state.queue) == 1

    # A different client_turn_id is an independent queued turn (no merge).
    second = _submit(server, client_turn_id="client-2", message="second input")
    assert second["result"]["duplicate"] is False
    assert second["result"]["turn_id"] != turn_id
    assert len(state.queue) == 2
    assert second["result"]["sequence"] > first["result"]["sequence"]


def test_queue_full_surfaces_retryable_busy(server, monkeypatch):
    from tui_gateway.turn_fifo import TURN_QUEUE_MAX_PENDING

    _session(server, "sid-full", session_key="stored-1", running=True)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    for index in range(TURN_QUEUE_MAX_PENDING):
        resp = _submit(server, client_turn_id=f"client-{index}")
        assert resp["result"]["status"] == "queued"

    overflow = _submit(server, client_turn_id="client-overflow")
    assert overflow["error"]["code"] == 4290


def test_turn_events_carry_full_envelope(server, monkeypatch):
    _session(server, "sid-ev", session_key="stored-1", running=True)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    events = []
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload=None: events.append((kind, sid, payload)))

    _submit(server)
    enqueued = [e for e in events if e[0] == "turn.enqueued"]
    assert len(enqueued) == 1
    payload = enqueued[0][2]
    for field in (
        "profile", "lineage_id", "stored_session_id", "runtime_session_id",
        "runtime_generation", "turn_id", "sequence", "origin", "delivery_mode",
    ):
        assert field in payload, field
    assert payload["origin"] == "desktop"
    assert payload["delivery_mode"] == "desktop-only"
    assert payload["runtime_generation"] == server.RUNTIME_GENERATION
