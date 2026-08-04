"""Proxy gateways relay busy plain-text follow-ups to the ordering backend."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _event(
    text: str = "also check on Zidane",
    *,
    message_type: MessageType = MessageType.TEXT,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="user1",
        ),
        message_id="msg2",
    )


def _runner(*, proxy: bool = True) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._background_tasks = set()
    runner._draining = False
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._proxy_relay = proxy
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner._is_user_authorized = lambda _source: True
    runner._peek_session_state = lambda _key: None
    runner._queue_or_replace_pending_event = MagicMock()
    runner._forward_busy_message_via_proxy = AsyncMock()
    return runner


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.TELEGRAM
    return adapter


async def _dispatch(runner: GatewayRunner, event: MessageEvent) -> bool:
    runner.adapters[event.source.platform] = _adapter()
    handled = await runner._handle_active_session_busy_message(
        event,
        build_session_key(event.source),
    )
    await asyncio.sleep(0)
    return handled


@pytest.mark.asyncio
async def test_proxy_relays_text_without_local_queue(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _runner()
    event = _event()

    assert await _dispatch(runner, event) is True

    runner._forward_busy_message_via_proxy.assert_awaited_once()
    assert runner._forward_busy_message_via_proxy.await_args.args[0] is event
    runner._queue_or_replace_pending_event.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        _event("", message_type=MessageType.PHOTO),
        _event("/help"),
    ],
)
async def test_proxy_keeps_media_and_commands_on_local_queue(monkeypatch, event) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _runner()

    assert await _dispatch(runner, event) is True

    runner._forward_busy_message_via_proxy.assert_not_awaited()
    runner._queue_or_replace_pending_event.assert_called_once_with(
        build_session_key(event.source), event
    )


@pytest.mark.asyncio
async def test_without_proxy_text_stays_on_local_queue(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _runner(proxy=False)
    event = _event()

    assert await _dispatch(runner, event) is True

    runner._forward_busy_message_via_proxy.assert_not_awaited()
    runner._queue_or_replace_pending_event.assert_called_once()


def _use_real_relay(runner: GatewayRunner, session_id: str = "sess-1") -> None:
    del runner._forward_busy_message_via_proxy
    facade = MagicMock()
    facade._store = runner.session_store
    facade.get_or_create_session = AsyncMock(
        return_value=types.SimpleNamespace(session_id=session_id)
    )
    runner._async_session_store = facade


@pytest.mark.asyncio
@pytest.mark.parametrize("previewed, send_count", [(False, 1), (True, 0)])
async def test_relay_only_sends_reply_not_already_streamed(previewed, send_count) -> None:
    runner = _runner()
    _use_real_relay(runner)
    event = _event()
    adapter = _adapter()
    runner.adapters[event.source.platform] = adapter
    runner._run_agent_via_proxy = AsyncMock(
        return_value={
            "final_response": "Zidane is fine.",
            "response_previewed": previewed,
        }
    )

    session_key = "agent:main:telegram:dm:123"
    await runner._forward_busy_message_via_proxy(event, session_key)

    assert runner._run_agent_via_proxy.await_args.kwargs["session_key"] == session_key
    assert runner._run_agent_via_proxy.await_args.kwargs["session_id"] == "sess-1"
    assert adapter._send_with_retry.await_count == send_count

