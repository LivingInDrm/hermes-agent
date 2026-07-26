"""Multi-subscriber event routing + per-turn delivery sink (design §9).

The single mutable ``session["transport"]`` is no longer the only event
route: write_json fans out to primary + the active turn's submitting
transport (the Gateway delivery sink) + registered observers. Nothing
steals anyone else's reply route.
"""

import threading
import types
from unittest.mock import MagicMock, patch

import pytest


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
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


class _FakeTransport:
    def __init__(self, name):
        self.name = name
        self.frames = []
        self._closed = False

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        self._closed = True


def _session(server, sid, transport=None, session_key="stored-1", running=False):
    record = {
        "agent": types.SimpleNamespace(),
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": running,
        "transport": transport,
        "attached_images": [],
        "created_at": 0.0,
    }
    server._sessions[sid] = record
    return record


def _activate_channel_turn(server, session, sink):
    record = server._new_turn_record(
        session, "channel input", sequence=0, origin="channel",
        delivery_mode="origin-channel", transport=sink,
    )
    with session["history_lock"]:
        record.sequence = server._turn_state(session).next_sequence()
        server._turn_state(session).activate(record)
    return record


def test_events_fan_out_to_primary_sink_and_subscribers(server):
    primary = _FakeTransport("desktop")
    sink = _FakeTransport("gateway")
    observer = _FakeTransport("observer")
    session = _session(server, "sid-1", transport=primary)
    _activate_channel_turn(server, session, sink)
    session["subscribers"] = {"sub-1": observer}

    server._emit("message.delta", "sid-1", {"text": "hi"})

    for transport in (primary, sink, observer):
        assert len(transport.frames) == 1, transport.name
    # Mid-turn events are attributed to the active turn for consumers that
    # need to associate the stream with a delivery target.
    params = sink.frames[0]["params"]
    assert params["turn"]["origin"] == "channel"
    assert params["turn"]["delivery_mode"] == "origin-channel"


def test_channel_turn_does_not_steal_primary_transport_on_drain(server):
    primary = _FakeTransport("desktop")
    sink = _FakeTransport("gateway")
    session = _session(server, "sid-2", transport=primary)
    record = server._new_turn_record(
        session, "queued channel turn", sequence=0, origin="channel",
        delivery_mode="origin-channel", transport=sink,
    )
    with session["history_lock"]:
        record.sequence = server._turn_state(session).next_sequence()
        server._turn_state(session).enqueue(record)

    fired = {}
    server_run = server._run_prompt_submit
    try:
        server._run_prompt_submit = lambda rid, sid, s, text: fired.update(text=text)
        assert server._drain_queued_prompt("r1", "sid-2", session) is True
    finally:
        server._run_prompt_submit = server_run

    assert fired["text"] == "queued channel turn"
    # The Gateway sink received the turn; the Desktop keeps the primary route.
    assert session["transport"] is primary


def test_desktop_turn_still_rebinds_primary_on_drain(server):
    old_primary = _FakeTransport("old")
    submitter = _FakeTransport("desktop-submitter")
    session = _session(server, "sid-3", transport=old_primary)
    record = server._new_turn_record(session, "desktop turn", sequence=0, transport=submitter)
    with session["history_lock"]:
        record.sequence = server._turn_state(session).next_sequence()
        server._turn_state(session).enqueue(record)

    server_run = server._run_prompt_submit
    try:
        server._run_prompt_submit = lambda *a, **k: None
        assert server._drain_queued_prompt("r1", "sid-3", session) is True
    finally:
        server._run_prompt_submit = server_run
    assert session["transport"] is submitter


def test_channel_turn_not_started_when_sink_is_dead(server):
    primary = _FakeTransport("desktop")
    sink = _FakeTransport("gateway")
    sink._closed = True  # dead Gateway Runtime Client socket
    session = _session(server, "sid-4", transport=primary)
    record = server._new_turn_record(
        session, "cannot deliver", sequence=0, origin="channel",
        delivery_mode="origin-channel", transport=sink,
    )
    with session["history_lock"]:
        record.sequence = server._turn_state(session).next_sequence()
        server._turn_state(session).enqueue(record)

    assert server._drain_queued_prompt("r1", "sid-4", session) is False
    assert session["running"] is False
    assert record.state == "queued"  # paused, not lost/failed


def test_subscribe_registers_observer_without_touching_primary(server, monkeypatch):
    primary = _FakeTransport("desktop")
    observer = _FakeTransport("observer")
    session = _session(server, "sid-5", transport=primary)
    monkeypatch.setattr(server, "_live_visible_history", lambda s, db, fallback: [])

    from tui_gateway.transport import bind_transport, reset_transport

    token = bind_transport(observer)
    try:
        resp = server.handle_request({
            "id": "s1", "method": "session.subscribe", "params": {"session_id": "sid-5"},
        })
    finally:
        reset_transport(token)

    subscription_id = resp["result"]["subscription_id"]
    assert subscription_id
    assert session["transport"] is primary
    assert session["subscribers"][subscription_id] is observer
    assert resp["result"]["runtime_generation"] == server.RUNTIME_GENERATION

    server._emit("status.update", "sid-5", {"state": "ready"})
    assert len(observer.frames) == 1

    # Unsubscribe stops the flow; transport death also cleans registrations.
    resp = server.handle_request({
        "id": "s2", "method": "session.unsubscribe",
        "params": {"session_id": "sid-5", "subscription_id": subscription_id},
    })
    assert resp["result"]["removed"] is True
    server._emit("status.update", "sid-5", {"state": "ready"})
    assert len(observer.frames) == 1


def test_disconnect_cleans_subscriptions(server):
    primary = _FakeTransport("desktop")
    observer = _FakeTransport("observer")
    session = _session(server, "sid-6", transport=primary)
    session["subscribers"] = {"sub-x": observer}

    server._close_sessions_for_transport(observer, end_reason="ws_disconnect")
    assert session["subscribers"] == {}
    # The primary transport is untouched by an observer disconnect.
    assert session["transport"] is primary


def test_terminal_turn_event_still_reaches_gateway_sink(server):
    """turn.completed/interrupted/failed are emitted AFTER the record flips
    terminal — the sink must still receive them (it is exactly what the
    Gateway delivery leg awaits); regression for the 1800s-timeout bug."""
    primary = _FakeTransport("desktop")
    sink = _FakeTransport("gateway")
    session = _session(server, "sid-term", transport=primary)
    _activate_channel_turn(server, session, sink)

    server._finish_active_turn("sid-term", session, "completed")

    kinds = [f["params"]["type"] for f in sink.frames]
    assert kinds == ["turn.completed"], kinds
    assert [f["params"]["type"] for f in primary.frames] == ["turn.completed"]


def test_channel_queue_full_release_frees_gateway_authority_claim(server, monkeypatch):
    """Queue-full must release the dedup claim under the SAME authority the
    claim used (gateway for channel-origin) — otherwise a retried
    client_turn_id hits a ghost record that never entered the queue."""
    from tui_gateway.turn_fifo import TURN_QUEUE_MAX_PENDING, GenerationDedupTable

    monkeypatch.setattr(server, "_turn_dedup", GenerationDedupTable())
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    session = _session(server, "sid-cq", transport=_FakeTransport("desktop"), running=True)
    sink = _FakeTransport("gateway")

    def _channel_record(text, client_id=""):
        return server._new_turn_record(
            session, text, sequence=0, origin="channel",
            delivery_mode="origin-channel", client_turn_id=client_id, transport=sink,
        )

    for index in range(TURN_QUEUE_MAX_PENDING):
        assert server._handle_busy_submit("r", "sid-cq", session, _channel_record(f"m{index}"))["result"]["status"] == "queued"

    rejected = _channel_record("retry me", client_id="chan-client-1")
    assert server._claim_turn_dedup(rejected) is None
    overflow = server._handle_busy_submit("r-full", "sid-cq", session, rejected)
    assert overflow["error"]["code"] == 4290

    # 腾出空位后，同一 client_turn_id 的重试必须能真正入队（而不是命中
    # 从未入队的幽灵 duplicate）。
    with session["history_lock"]:
        session["turn_state"].queue.popleft()
    retry = _channel_record("retry me", client_id="chan-client-1")
    assert server._claim_turn_dedup(retry) is None
    accepted = server._handle_busy_submit("r-retry", "sid-cq", session, retry)
    assert accepted["result"]["status"] == "queued"


def test_channel_turn_lifecycle_broadcasts_to_desktop_observers(server):
    """channel-origin 的 turn 生命周期事件广播给所有 desktop 连接（即使该
    Session 没有任何订阅者），桌面据此在页面未打开时创建/刷新 Task snapshot；
    desktop-origin turn 不广播。"""
    desktop = _FakeTransport("desktop-anywhere")
    server.register_desktop_observer(desktop)
    try:
        gateway_sink = _FakeTransport("gateway")
        session = _session(server, "sid-bcast", transport=gateway_sink, session_key="chan-tip")
        record = _activate_channel_turn(server, session, gateway_sink)
        server._emit_turn_event("turn.started", "sid-bcast", session, record)

        kinds = [f["params"]["type"] for f in desktop.frames]
        assert kinds == ["turn.started"]
        payload = desktop.frames[0]["params"]["payload"]
        assert payload["stored_session_id"] == "chan-tip"
        assert payload["lineage_id"]
        assert payload["origin"] == "channel"
        # 桌面从未提交过 channel turn，事件是它拿到用户原文做实时展示的
        # 唯一来源（bug：飞书用户消息不出现在 Desktop 对话流）。
        assert payload["user"] == "channel input"

        # desktop-origin turn：不广播（打开会话的订阅者才收流）。
        session2 = _session(server, "sid-desk", transport=_FakeTransport("p2"))
        record2 = server._new_turn_record(session2, "x", sequence=1)
        with session2["history_lock"]:
            server._turn_state(session2).activate(record2)
        server._emit_turn_event("turn.started", "sid-desk", session2, record2)
        assert len(desktop.frames) == 1
    finally:
        server.unregister_desktop_observer(desktop)
