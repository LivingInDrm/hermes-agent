"""Proxy mode: a busy-session follow-up is relayed, not queued locally.

In proxy mode the gateway is a relay — ``_run_agent`` hands every turn to the
remote backend and never builds a local ``AIAgent``. That backend is also what
other clients (a desktop UI, a second gateway) talk to, so it is the only place
that can order turns across all of them.

Holding a follow-up in the local FIFO until the current turn returns hides it
from the backend for the whole run. Observed consequence: a channel message that
arrived at 09:23:41 only reached the backend at 09:24:31 — right after the
previous turn's proxy response closed — by which time a message submitted
directly to the backend at 09:24:29 had already started. The message that
arrived first ran last, and no client could show it as pending in between.

Note that all three ``busy_input_mode`` values degrade to "queue" under proxy
mode: ``steer``/``interrupt`` both require ``running_agent`` to be a real agent,
and in proxy mode that slot never advances past ``_AGENT_PENDING_SENTINEL``.
So the mode knob cannot fix this — the relay has to bypass the queue.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner  # noqa: E402

PROXY_URL = "http://127.0.0.1:49953"


def _make_event(text: str = "also check on Zidane", **kwargs) -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="123",
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=kwargs.pop("message_type", MessageType.TEXT),
        source=source,
        message_id="msg2",
        **kwargs,
    )


def _make_runner(*, proxy: str | None = PROXY_URL) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._busy_input_mode = "queue"
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner._is_user_authorized = lambda _source: True
    runner._get_proxy_url = lambda: proxy
    runner._proxy_relay = bool(proxy)
    runner._queue_or_replace_pending_event = MagicMock()
    runner._forward_busy_message_via_proxy = AsyncMock()
    return runner


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value="telegram")
    return adapter


async def _dispatch(runner: GatewayRunner, event: MessageEvent) -> bool:
    adapter = _make_adapter()
    runner.adapters[event.source.platform] = adapter
    session_key = build_session_key(event.source)
    # A turn is in flight. In proxy mode the slot never holds a real agent.
    runner._running_agents[session_key] = object()
    handled = await runner._handle_active_session_busy_message(event, session_key)
    # The relay is spawned, not awaited — give the loop a tick to start it.
    await asyncio.sleep(0)
    return handled


def _stub_session_store(runner: GatewayRunner, session_id: str = "sess-1") -> None:
    """``async_session_store`` is a read-only property over ``session_store``."""
    facade = MagicMock()
    facade._store = runner.session_store
    facade.get_or_create_session = AsyncMock(
        return_value=types.SimpleNamespace(session_id=session_id)
    )
    runner._async_session_store = facade


@pytest.mark.asyncio
async def test_proxy_mode_relays_text_followup_instead_of_queueing(monkeypatch) -> None:
    """The follow-up reaches the backend now, not after the current turn."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _make_runner()
    event = _make_event()

    handled = await _dispatch(runner, event)

    assert handled is True
    runner._forward_busy_message_via_proxy.assert_awaited_once()
    forwarded_event, _key = runner._forward_busy_message_via_proxy.await_args.args
    assert forwarded_event is event
    # The whole point: it must NOT sit in the gateway's own FIFO.
    runner._queue_or_replace_pending_event.assert_not_called()


@pytest.mark.asyncio
async def test_proxy_mode_still_queues_media(monkeypatch) -> None:
    """Media needs this gateway's preprocessing — it keeps the queue path."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _make_runner()
    event = _make_event(text="", message_type=MessageType.PHOTO)

    handled = await _dispatch(runner, event)

    assert handled is True
    runner._forward_busy_message_via_proxy.assert_not_awaited()
    runner._queue_or_replace_pending_event.assert_called_once()


@pytest.mark.asyncio
async def test_without_proxy_the_followup_still_queues(monkeypatch) -> None:
    """No proxy backend → this gateway *is* the engine and must serialize."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _make_runner(proxy=None)
    event = _make_event()

    handled = await _dispatch(runner, event)

    assert handled is True
    runner._forward_busy_message_via_proxy.assert_not_awaited()
    runner._queue_or_replace_pending_event.assert_called_once()


@pytest.mark.asyncio
async def test_relay_sends_a_reply_the_stream_did_not_deliver() -> None:
    """Non-streamed replies still reach the chat."""
    runner = _make_runner()
    del runner._forward_busy_message_via_proxy  # exercise the real method
    adapter = _make_adapter()
    event = _make_event()
    runner.adapters[event.source.platform] = adapter
    _stub_session_store(runner)
    runner._reply_anchor_for_event = lambda _event: "msg2"
    runner._thread_metadata_for_source = lambda _source, _anchor: {"thread": None}
    runner._run_agent_via_proxy = AsyncMock(
        return_value={"final_response": "Zidane is fine.", "response_previewed": False}
    )

    await runner._forward_busy_message_via_proxy(event, "agent:main:telegram:dm:123")

    # The stable per-chat key must ride along, or the backend cannot tell which
    # conversation this is and opens a new one.
    assert runner._run_agent_via_proxy.await_args.kwargs["session_key"] == (
        "agent:main:telegram:dm:123"
    )
    adapter._send_with_retry.assert_awaited_once()
    assert adapter._send_with_retry.await_args.kwargs["content"] == "Zidane is fine."


@pytest.mark.asyncio
async def test_relay_does_not_double_send_a_streamed_reply() -> None:
    """``_run_agent_via_proxy`` already streamed it — sending again duplicates."""
    runner = _make_runner()
    del runner._forward_busy_message_via_proxy
    adapter = _make_adapter()
    event = _make_event()
    runner.adapters[event.source.platform] = adapter
    _stub_session_store(runner)
    runner._reply_anchor_for_event = lambda _event: "msg2"
    runner._thread_metadata_for_source = lambda _source, _anchor: {}
    runner._run_agent_via_proxy = AsyncMock(
        return_value={"final_response": "streamed already", "response_previewed": True}
    )

    await runner._forward_busy_message_via_proxy(event, "agent:main:telegram:dm:123")

    adapter._send_with_retry.assert_not_awaited()
