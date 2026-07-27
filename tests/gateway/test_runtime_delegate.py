"""Tests for gateway runtime delegation — submitting turns to `hermes serve`.

Shared-session runtime, gateway-delegation milestone: when
``gateway.runtime_delegate`` enables a platform+route, the gateway submits
prepared inbound turns to the profile's serve runtime over its WS JSON-RPC
(``turn.submit``) instead of running a local AIAgent, and delivers the
runtime's final reply through its normal delivery path.  See
docs/design/channel-desktop-shared-session-runtime.md and
gateway/runtime_client.py.
"""

import asyncio
from types import SimpleNamespace

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.run import GatewayRunner
from gateway.runtime_client import (
    RUNTIME_QUEUE_FULL_CODE,
    GatewayRuntimeClient,
    RuntimeConnectionError,
    RuntimeRequestError,
    RuntimeTurnHandle,
)
from gateway.session import SessionSource


def _make_runner(runtime_delegate=None):
    """Create a minimal GatewayRunner for delegation tests.

    Mirrors the bare-runner harness in tests/gateway/test_proxy_mode.py,
    but with a REAL GatewayConfig so the runtime_delegate field behaves
    like production (a MagicMock config would return a MagicMock for it).
    """
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = GatewayConfig(runtime_delegate=runtime_delegate or {})
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    return runner


def _make_source(platform=Platform.FEISHU):
    return SessionSource(
        platform=platform,
        chat_id="oc_chat_1",
        chat_name="Test Chat",
        chat_type="group",
        user_id="ou_user_1",
        user_id_alt="on_union_1",
        user_name="tester",
        thread_id=None,
    )


def _delegate_cfg(**overrides):
    cfg = {
        "enabled": True,
        "platforms": ["feishu"],
        "event_routes": ["message_receive"],
        "url": "",
        "token": "",
    }
    cfg.update(overrides)
    return cfg


def _set_runtime_env(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_RUNTIME_URL", "ws://127.0.0.1:8899/api/ws")
    monkeypatch.setenv("HERMES_DESKTOP_RUNTIME_TOKEN", "svc-token")


def _clear_runtime_env(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP_RUNTIME_URL", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_RUNTIME_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class TestRuntimeDelegateConfig:
    def test_default_config_is_off(self):
        from hermes_cli.config import DEFAULT_CONFIG

        rd = DEFAULT_CONFIG["gateway"]["runtime_delegate"]
        assert rd["enabled"] is False
        assert rd["platforms"] == []
        assert rd["event_routes"] == ["message_receive"]
        assert rd["url"] == ""
        assert rd["token"] == ""

    def test_gateway_config_defaults_to_empty_dict(self):
        cfg = GatewayConfig()
        assert cfg.runtime_delegate == {}
        assert "runtime_delegate" in cfg.to_dict()

    def test_from_dict_passes_dict_through(self):
        cfg = GatewayConfig.from_dict(
            {"runtime_delegate": {"enabled": True, "platforms": ["feishu"]}}
        )
        assert cfg.runtime_delegate["enabled"] is True
        assert cfg.runtime_delegate["platforms"] == ["feishu"]

    def test_from_dict_nested_gateway_fallback(self):
        cfg = GatewayConfig.from_dict(
            {"gateway": {"runtime_delegate": {"enabled": True}}}
        )
        assert cfg.runtime_delegate == {"enabled": True}

    def test_from_dict_top_level_wins_over_nested(self):
        cfg = GatewayConfig.from_dict(
            {
                "runtime_delegate": {"enabled": False},
                "gateway": {"runtime_delegate": {"enabled": True}},
            }
        )
        assert cfg.runtime_delegate == {"enabled": False}

    def test_from_dict_non_dict_falls_back_to_disabled(self):
        cfg = GatewayConfig.from_dict({"runtime_delegate": "yes please"})
        assert cfg.runtime_delegate == {}

    def test_load_gateway_config_surfaces_nested_form(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "gateway": {
                        "runtime_delegate": {
                            "enabled": True,
                            "platforms": ["feishu"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        cfg = load_gateway_config()
        assert cfg.runtime_delegate["enabled"] is True
        assert cfg.runtime_delegate["platforms"] == ["feishu"]

    def test_load_gateway_config_top_level_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "runtime_delegate": {"enabled": True, "platforms": ["feishu"]},
                    "gateway": {
                        "runtime_delegate": {"enabled": False, "platforms": []}
                    },
                }
            ),
            encoding="utf-8",
        )
        cfg = load_gateway_config()
        assert cfg.runtime_delegate["enabled"] is True
        assert cfg.runtime_delegate["platforms"] == ["feishu"]

    def test_load_gateway_config_default_off(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = load_gateway_config()
        assert cfg.runtime_delegate == {}


# ---------------------------------------------------------------------------
# GatewayRuntimeClient frame handling (pure, socket-free)
# ---------------------------------------------------------------------------


class _FakeWS:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send_json(self, frame):
        self.sent.append(frame)

    async def close(self):
        self.closed = True


def _submit_result(**overrides):
    result = {
        "enqueued": True,
        "duplicate": False,
        "durability": "process",
        "turn_id": "t-1",
        "sequence": 3,
        "queue_position": 0,
        "lineage_id": "lin-1",
        "stored_session_id": "stored-9",
        "runtime_session_id": "rt-1",
        "runtime_generation": "5363e55bba1741cfb7cab44b09b2b22c",
        "status": "streaming",
    }
    result.update(overrides)
    return result


def _turn_key(turn_id="t-1"):
    return {
        "turn_id": turn_id,
        "sequence": 3,
        "origin": "channel",
        "delivery_mode": "origin-channel",
        "runtime_generation": "5363e55bba1741cfb7cab44b09b2b22c",
    }


def _event(event_type, payload, turn=None):
    params = {"type": event_type, "session_id": "sess", "payload": payload}
    if turn is not None:
        params["turn"] = turn
    return {"method": "event", "params": params}


class TestGatewayRuntimeClientFrames:
    @pytest.mark.asyncio
    async def test_turn_lifecycle_completed(self, monkeypatch):
        client = GatewayRuntimeClient("ws://127.0.0.1:1/api/ws", "tok")
        fake_ws = _FakeWS()
        client._ws = fake_ws

        async def _noop():
            return None

        monkeypatch.setattr(client, "ensure_connected", _noop)

        task = asyncio.create_task(client.submit_turn({"message": "hi"}))
        await asyncio.sleep(0)

        # The request went out as a JSON-RPC turn.submit frame.
        frame = fake_ws.sent[0]
        assert frame["method"] == "turn.submit"
        assert frame["jsonrpc"] == "2.0"

        # Scripted sequence: submit response → delta ×2 → complete → completed.
        client.handle_frame({"jsonrpc": "2.0", "id": frame["id"], "result": _submit_result()})
        handle = await task
        assert handle.turn_id == "t-1"
        assert handle.stored_session_id == "stored-9"
        assert handle.lineage_id == "lin-1"

        client.handle_frame(_event("message.delta", {"text": "Hel"}, _turn_key()))
        client.handle_frame(_event("message.delta", {"text": "lo"}, _turn_key()))
        client.handle_frame(_event("message.complete", {"text": "Hello"}, _turn_key()))
        client.handle_frame(_event("turn.completed", {"turn_id": "t-1"}))

        outcome = await asyncio.wait_for(handle.future, timeout=1)
        assert outcome["state"] == "completed"
        assert outcome["final_text"] == "Hello"
        assert outcome["already_delivered"] is False
        # Terminal event unregisters the turn.
        assert "t-1" not in client._turns

    @pytest.mark.asyncio
    async def test_message_complete_falls_back_to_deltas(self):
        client = GatewayRuntimeClient("ws://x/api/ws", "tok")
        loop = asyncio.get_running_loop()
        handle = RuntimeTurnHandle(turn_id="t-2", future=loop.create_future())
        client._turns["t-2"] = handle

        client.handle_frame(_event("message.delta", {"text": "a"}, _turn_key("t-2")))
        client.handle_frame(_event("message.delta", {"text": "b"}, _turn_key("t-2")))
        client.handle_frame(_event("message.complete", {}, _turn_key("t-2")))
        client.handle_frame(_event("turn.completed", {"turn_id": "t-2"}))

        outcome = await handle.future
        assert outcome["state"] == "completed"
        assert outcome["final_text"] == "ab"

    @pytest.mark.asyncio
    async def test_turn_failed_resolves_failed(self):
        client = GatewayRuntimeClient("ws://x/api/ws", "tok")
        loop = asyncio.get_running_loop()
        handle = RuntimeTurnHandle(turn_id="t-3", future=loop.create_future())
        client._turns["t-3"] = handle

        client.handle_frame(
            _event("turn.failed", {"turn_id": "t-3", "error": "model exploded"})
        )
        outcome = await handle.future
        assert outcome["state"] == "failed"
        assert outcome["error"] == "model exploded"

    @pytest.mark.asyncio
    async def test_connection_lost_turns_unknown_requests_retryable(self):
        client = GatewayRuntimeClient("ws://x/api/ws", "tok")
        loop = asyncio.get_running_loop()
        req_future = loop.create_future()
        client._pending["req-1"] = req_future
        handle = RuntimeTurnHandle(turn_id="t-9", future=loop.create_future())
        client._turns["t-9"] = handle

        client.connection_lost()

        # Pending requests fail retryable; lost turns are UNKNOWN (never
        # auto-replayed — design §12.3).
        with pytest.raises(RuntimeConnectionError):
            await req_future
        outcome = await handle.future
        assert outcome["state"] == "unknown"
        assert client._pending == {}
        assert client._turns == {}

    @pytest.mark.asyncio
    async def test_error_response_raises_request_error_with_code(self, monkeypatch):
        client = GatewayRuntimeClient("ws://x/api/ws", "tok")
        fake_ws = _FakeWS()
        client._ws = fake_ws

        async def _noop():
            return None

        monkeypatch.setattr(client, "ensure_connected", _noop)

        task = asyncio.create_task(client.request("turn.submit", {}))
        await asyncio.sleep(0)
        frame = fake_ws.sent[0]
        client.handle_frame(
            {
                "jsonrpc": "2.0",
                "id": frame["id"],
                "error": {"code": RUNTIME_QUEUE_FULL_CODE, "message": "queue full"},
            }
        )
        with pytest.raises(RuntimeRequestError) as exc_info:
            await task
        assert exc_info.value.code == RUNTIME_QUEUE_FULL_CODE

    @pytest.mark.asyncio
    async def test_submit_turn_duplicate_reattaches_existing_handle(self, monkeypatch):
        client = GatewayRuntimeClient("ws://x/api/ws", "tok")
        loop = asyncio.get_running_loop()
        existing = RuntimeTurnHandle(turn_id="t-dup", future=loop.create_future())
        client._turns["t-dup"] = existing

        async def _fake_request(method, params, timeout=30.0):
            return _submit_result(turn_id="t-dup", duplicate=True, status="duplicate")

        monkeypatch.setattr(client, "request", _fake_request)
        handle = await client.submit_turn({"message": "retry"})
        assert handle is existing

    @pytest.mark.asyncio
    async def test_tip_updated_callback(self):
        seen = []
        client = GatewayRuntimeClient(
            "ws://x/api/ws", "tok", on_tip_updated=seen.append
        )
        client.handle_frame(
            _event("session.tip.updated", {"stored_session_id": "stored-77"})
        )
        assert seen == [{"stored_session_id": "stored-77"}]

    @pytest.mark.asyncio
    async def test_events_for_unknown_turn_are_ignored(self):
        client = GatewayRuntimeClient("ws://x/api/ws", "tok")
        # Must not raise even with no registered turn.
        client.handle_frame(_event("message.delta", {"text": "x"}, _turn_key("t-??")))
        client.handle_frame(_event("turn.completed", {"turn_id": "t-??"}))


# ---------------------------------------------------------------------------
# _runtime_delegate_enabled_for gating
# ---------------------------------------------------------------------------


class TestRuntimeClarifyRelay:
    """serve 在委托 turn 内发起的 clarify 中继回原渠道（能力回归修复）：
    旧路径的 clarify → 渠道按钮/文本 → 拦截回答；委托后经 clarify.respond
    RPC 回给 serve。expire 必须清掉挂起项，防止吞掉下一条渠道消息。"""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        from tools import clarify_gateway

        yield
        clarify_gateway.clear_session("relay-sess")

    @pytest.mark.asyncio
    async def test_clarify_frames_route_to_callbacks(self):
        seen = {}
        client = GatewayRuntimeClient(
            "ws://x/api/ws",
            "tok",
            on_clarify_request=lambda handle, payload: seen.update(
                request=(handle.turn_id, dict(payload))
            ),
            on_clarify_expire=lambda payload: seen.update(expire=dict(payload)),
        )
        loop = asyncio.get_running_loop()
        client._turns["t-c"] = RuntimeTurnHandle(turn_id="t-c", future=loop.create_future())

        client.handle_frame(_event(
            "clarify.request",
            {"request_id": "rid-1", "question": "哪种？", "choices": ["A", "B"]},
            _turn_key("t-c"),
        ))
        assert seen["request"][0] == "t-c"
        assert seen["request"][1]["question"] == "哪种？"

        # expire 不依赖 turn 归属（prompt 生命周期可长于 turn 记账）。
        client.handle_frame(_event("clarify.expire", {"request_id": "rid-1"}))
        assert seen["expire"] == {"request_id": "rid-1"}

    @pytest.mark.asyncio
    async def test_relay_presents_prompt_and_forwards_answer(self):
        from tools import clarify_gateway
        from gateway.runtime_clarify import relay_clarify_request

        responded = {}

        class _FakeClient:
            async def respond_clarify(self, request_id, answer):
                responded.update(request_id=request_id, answer=answer)
                return {"status": "ok"}

        sent = {}

        class _FakeAdapter:
            def pause_typing_for_chat(self, chat_id):
                sent["paused"] = chat_id

            async def send_clarify(self, **kwargs):
                sent.update(kwargs)
                return SimpleNamespace(success=True)

        armed = relay_clarify_request(
            payload={"request_id": "rid-2", "question": "去哪？", "choices": None},
            session_key="relay-sess",
            adapter=_FakeAdapter(),
            chat_id="chat-9",
            metadata={"thread": "t"},
            loop=asyncio.get_running_loop(),
            client=_FakeClient(),
        )
        assert armed is True
        await asyncio.sleep(0)  # 让 send 任务运行
        assert sent["clarify_id"] == "rid-2"
        assert sent["question"] == "去哪？"
        # 渠道用户下一条消息经既有拦截解析（此处直接解析入口）。
        assert clarify_gateway.resolve_text_response_for_session("relay-sess", "东京") is True
        for _ in range(100):
            if responded:
                break
            await asyncio.sleep(0.05)
        assert responded == {"request_id": "rid-2", "answer": "东京"}

    @pytest.mark.asyncio
    async def test_send_failure_releases_pending_without_respond(self):
        from tools import clarify_gateway
        from gateway.runtime_clarify import relay_clarify_request

        responded = {}

        class _FakeClient:
            async def respond_clarify(self, request_id, answer):
                responded.update(request_id=request_id)
                return {"status": "ok"}

        class _FailingAdapter:
            def pause_typing_for_chat(self, chat_id):
                pass

            async def send_clarify(self, **kwargs):
                return SimpleNamespace(success=False)

        relay_clarify_request(
            payload={"request_id": "rid-3", "question": "q"},
            session_key="relay-sess",
            adapter=_FailingAdapter(),
            chat_id="chat-9",
            metadata=None,
            loop=asyncio.get_running_loop(),
            client=_FakeClient(),
        )
        await asyncio.sleep(0.05)
        # 发送失败：挂起项必须释放（否则下一条渠道消息会被吞成回答），
        # 且不得向 serve 发送空答案。
        for _ in range(100):
            if clarify_gateway.get_pending_for_session("relay-sess", include_choice_prompts=True) is None:
                break
            await asyncio.sleep(0.05)
        assert clarify_gateway.get_pending_for_session("relay-sess", include_choice_prompts=True) is None
        assert responded == {}

    @pytest.mark.asyncio
    async def test_expire_clears_pending(self):
        from tools import clarify_gateway
        from gateway.runtime_clarify import relay_clarify_expire, relay_clarify_request

        class _FakeClient:
            async def respond_clarify(self, request_id, answer):
                return {"status": "ok"}

        class _FakeAdapter:
            def pause_typing_for_chat(self, chat_id):
                pass

            async def send_clarify(self, **kwargs):
                return SimpleNamespace(success=True)

        relay_clarify_request(
            payload={"request_id": "rid-4", "question": "q", "choices": ["A"]},
            session_key="relay-sess",
            adapter=_FakeAdapter(),
            chat_id="chat-9",
            metadata=None,
            loop=asyncio.get_running_loop(),
            client=_FakeClient(),
        )
        await asyncio.sleep(0)
        assert clarify_gateway.get_pending_for_session("relay-sess", include_choice_prompts=True) is not None
        # Desktop 抢答/serve 超时 → expire 广播 → 挂起项清除。
        relay_clarify_expire({"request_id": "rid-4"})
        for _ in range(100):
            if clarify_gateway.get_pending_for_session("relay-sess", include_choice_prompts=True) is None:
                break
            await asyncio.sleep(0.05)
        assert clarify_gateway.get_pending_for_session("relay-sess", include_choice_prompts=True) is None


class TestRuntimeDelegateEnabledFor:
    def test_default_off(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        runner = _make_runner()
        assert runner._runtime_delegate_enabled_for(_make_source()) is False

    def test_enabled_with_env_endpoint(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        assert runner._runtime_delegate_enabled_for(_make_source()) is True

    def test_platform_gating(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        runner = _make_runner(_delegate_cfg(platforms=["feishu"]))
        assert runner._runtime_delegate_enabled_for(
            _make_source(Platform.TELEGRAM)
        ) is False
        assert runner._runtime_delegate_enabled_for(
            _make_source(Platform.FEISHU)
        ) is True

    def test_requires_resolvable_endpoint(self, monkeypatch):
        _clear_runtime_env(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        assert runner._runtime_delegate_enabled_for(_make_source()) is False

    def test_config_url_token_fallback(self, monkeypatch):
        _clear_runtime_env(monkeypatch)
        runner = _make_runner(
            _delegate_cfg(url="ws://127.0.0.1:9001/api/ws", token="cfg-token")
        )
        assert runner._runtime_delegate_enabled_for(_make_source()) is True
        assert runner._runtime_delegate_endpoint() == (
            "ws://127.0.0.1:9001/api/ws",
            "cfg-token",
        )

    def test_env_wins_over_config(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        runner = _make_runner(
            _delegate_cfg(url="ws://config-host/api/ws", token="cfg-token")
        )
        assert runner._runtime_delegate_endpoint() == (
            "ws://127.0.0.1:8899/api/ws",
            "svc-token",
        )

    def test_internal_event_never_delegates(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        source = _make_source()
        internal_event = SimpleNamespace(internal=True)
        external_event = SimpleNamespace(internal=False)
        assert runner._runtime_delegate_enabled_for(source, event=internal_event) is False
        assert runner._runtime_delegate_enabled_for(source, event=external_event) is True

    def test_event_routes_gating(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        runner = _make_runner(_delegate_cfg(event_routes=["card_action"]))
        assert runner._runtime_delegate_enabled_for(_make_source()) is False


# ---------------------------------------------------------------------------
# The run.py seam: _run_agent delegates to _run_turn_via_runtime
# ---------------------------------------------------------------------------


class _FakeRuntimeClient:
    """Scripted stand-in for GatewayRuntimeClient."""

    def __init__(self, outcome=None, submit_error=None, resolve_immediately=True):
        self.submitted = []
        self.handles = []
        self._outcome = outcome or {
            "state": "completed",
            "final_text": "hi from serve",
            "already_delivered": False,
            "error": "",
        }
        self._submit_error = submit_error
        self._resolve_immediately = resolve_immediately

    async def submit_turn(self, params, timeout=60.0):
        self.submitted.append(params)
        if self._submit_error is not None:
            raise self._submit_error
        handle = RuntimeTurnHandle(
            turn_id=f"turn-{len(self.submitted)}",
            sequence=len(self.submitted),
            stored_session_id="stored-42",
            lineage_id="lin-1",
            status="streaming" if len(self.submitted) == 1 else "queued",
            future=asyncio.get_running_loop().create_future(),
        )
        self.handles.append(handle)
        if self._resolve_immediately:
            handle.future.set_result(dict(self._outcome))
        return handle


def _install_aiagent_guard(monkeypatch):
    """Any AIAgent construction in delegate mode is a hard failure."""
    import run_agent

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "AIAgent must not be constructed on a runtime-delegated turn"
            )

    monkeypatch.setattr(run_agent, "AIAgent", _Boom)


class TestRunAgentRuntimeDispatch:
    @pytest.mark.asyncio
    async def test_delegates_and_returns_final_text(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        _install_aiagent_guard(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        fake = _FakeRuntimeClient()
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)
        source = _make_source()

        result = await runner._run_agent(
            message="hello serve",
            context_prompt="channel context prompt",
            history=[{"role": "user", "content": "earlier"}],
            source=source,
            session_id="sess-1",
            session_key="feishu:oc_chat_1",
            run_generation=None,
            event_message_id="om_msg_1",
        )

        # (b) final text flows back through the proxy-shaped contract; the
        # CALLER delivers it via the delivery-ledger + adapter send path.
        assert result["final_response"] == "hi from serve"
        assert result["api_calls"] == 1
        assert result["history_offset"] == 1
        assert result["session_id"] == "sess-1"
        assert result["response_previewed"] is False
        # Serve owns persistence — the gateway must skip its session-DB write.
        assert result["agent_persisted"] is True

        # (c) stored-session binding recorded for the session key.
        assert runner._runtime_stored_ids["feishu:oc_chat_1"] == "stored-42"

        # (d) submit params carry the wire contract.
        params = fake.submitted[0]
        assert params["delivery_mode"] == "origin-channel"
        assert params["busy_mode"] == "fifo"
        assert "feishu:oc_chat_1" in params["client_turn_id"]
        assert params["client_turn_id"].endswith("om_msg_1")
        assert params["conversation"] == {
            "kind": "channel-create",
            "source": "feishu",
            "source_session_key": "feishu:oc_chat_1",
        }
        trusted = params["trusted_source"]
        assert trusted["platform"] == "feishu"
        assert trusted["chat_type"] == "group"
        assert trusted["user_id"] == "ou_user_1"
        assert trusted["user_id_alt"] == "on_union_1"
        assert trusted["user_name"] == "tester"
        assert trusted["role_authorized"] is True
        # The channel/system prompt travels as ephemeral_prompt, never
        # concatenated into the user message.
        assert params["ephemeral_prompt"] == "channel context prompt"
        assert params["message"] == "hello serve"

    @pytest.mark.asyncio
    async def test_known_stored_session_reuses_stored_conversation(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        _install_aiagent_guard(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        runner._runtime_stored_ids = {"feishu:oc_chat_1": "stored-42"}
        fake = _FakeRuntimeClient()
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)

        result = await runner._run_agent(
            message="again",
            context_prompt="",
            history=[],
            source=_make_source(),
            session_id="sess-1",
            session_key="feishu:oc_chat_1",
            event_message_id="om_msg_2",
        )

        assert result["final_response"] == "hi from serve"
        assert fake.submitted[0]["conversation"] == {
            "kind": "stored",
            "stored_session_id": "stored-42",
        }
        # Empty ephemeral prompt is omitted entirely.
        assert "ephemeral_prompt" not in fake.submitted[0]

    @pytest.mark.asyncio
    async def test_model_override_becomes_allowlisted_execution_hints(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        _install_aiagent_guard(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        runner._session_model_overrides = {
            "feishu:oc_chat_1": {
                "model": "some-model",
                "provider": "some-provider",
                "api_key": "SECRET-MUST-NOT-LEAK",
                "base_url": "http://local",
            }
        }
        fake = _FakeRuntimeClient()
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)

        await runner._run_agent(
            message="hi",
            context_prompt="",
            history=[],
            source=_make_source(),
            session_id="sess-1",
            session_key="feishu:oc_chat_1",
            event_message_id="om_msg_3",
        )

        hints = fake.submitted[0]["execution_hints"]
        assert hints == {"model": "some-model", "provider": "some-provider"}
        assert "api_key" not in hints
        assert "base_url" not in hints

    @pytest.mark.asyncio
    async def test_internal_event_skips_delegation(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        fake = _FakeRuntimeClient()
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)

        try:
            await runner._run_agent(
                message="",
                context_prompt="",
                history=[],
                source=_make_source(),
                session_id="sess-1",
                session_key="feishu:oc_chat_1",
                event=SimpleNamespace(internal=True),
            )
        except Exception:
            pass  # Expected — the bare runner can't run a real inline agent.

        assert fake.submitted == []

    @pytest.mark.asyncio
    async def test_queue_full_surfaces_busy_notice_no_fallback(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        _install_aiagent_guard(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        fake = _FakeRuntimeClient(
            submit_error=RuntimeRequestError(RUNTIME_QUEUE_FULL_CODE, "queue full")
        )
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)

        result = await runner._run_agent(
            message="hi",
            context_prompt="",
            history=[],
            source=_make_source(),
            session_id="sess-1",
            session_key="feishu:oc_chat_1",
            event_message_id="om_msg_4",
        )

        assert "try again" in result["final_response"].lower()
        assert result["failed"] is True

    @pytest.mark.asyncio
    async def test_connect_failure_surfaces_error_no_fallback(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        _install_aiagent_guard(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        fake = _FakeRuntimeClient(
            submit_error=RuntimeConnectionError("connection refused")
        )
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)

        result = await runner._run_agent(
            message="hi",
            context_prompt="",
            history=[],
            source=_make_source(),
            session_id="sess-1",
            session_key="feishu:oc_chat_1",
            event_message_id="om_msg_5",
        )

        # Design §12.3: no fallback to the local agent — the failure is
        # surfaced to the chat like an inline error.
        assert "Runtime delegation error" in result["final_response"]
        assert result["failed"] is True

    @pytest.mark.asyncio
    async def test_interrupted_turn_stays_silent(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        _install_aiagent_guard(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        fake = _FakeRuntimeClient(
            outcome={
                "state": "interrupted",
                "final_text": "partial",
                "already_delivered": False,
                "error": "",
            }
        )
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)

        result = await runner._run_agent(
            message="hi",
            context_prompt="",
            history=[],
            source=_make_source(),
            session_id="sess-1",
            session_key="feishu:oc_chat_1",
            event_message_id="om_msg_6",
        )

        assert result["final_response"] == ""
        assert result["interrupted"] is True

    @pytest.mark.asyncio
    async def test_unknown_outcome_surfaces_as_error(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        _install_aiagent_guard(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        fake = _FakeRuntimeClient(
            outcome={
                "state": "unknown",
                "final_text": "",
                "already_delivered": False,
                "error": "",
            }
        )
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)

        result = await runner._run_agent(
            message="hi",
            context_prompt="",
            history=[],
            source=_make_source(),
            session_id="sess-1",
            session_key="feishu:oc_chat_1",
            event_message_id="om_msg_7",
        )

        assert result["failed"] is True
        assert "unknown" in result["error"]

    @pytest.mark.asyncio
    async def test_busy_followup_goes_to_runtime_fifo(self, monkeypatch):
        """Design §8.1: two rapid inputs both hold runtime sequences before
        the first completes — the gateway keeps NO post-enqueue model-wait
        queue for delegated routes."""
        from unittest.mock import AsyncMock, MagicMock

        from gateway.platforms.base import MessageEvent

        _set_runtime_env(monkeypatch)
        _install_aiagent_guard(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        runner._draining = False
        fake = _FakeRuntimeClient(resolve_immediately=False)
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)
        monkeypatch.setattr(runner, "_is_user_authorized", lambda source: True)

        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock()
        adapter.send_typing = AsyncMock()
        runner.adapters = {Platform.FEISHU: adapter}

        # Spy: the gateway busy queue must never be touched on this path.
        gateway_queued = []
        monkeypatch.setattr(
            runner,
            "_queue_or_replace_pending_event",
            lambda sk, ev: gateway_queued.append(ev),
        )

        # Deterministic preprocessing (the real pipeline is exercised by the
        # normal-path tests; here it would need a full runner).
        async def _prep(**kwargs):
            return kwargs["event"].text

        monkeypatch.setattr(
            runner, "_prepare_profile_scoped_inbound_message_text", _prep
        )

        source = _make_source()

        # First delegated turn: submitted, still awaiting its outcome.
        turn1 = asyncio.create_task(
            runner._run_agent(
                message="first",
                context_prompt="",
                history=[],
                source=source,
                session_id="sess-1",
                session_key="feishu:oc_chat_1",
                event_message_id="om_1",
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)
            if fake.submitted:
                break
        assert len(fake.submitted) == 1
        assert not fake.handles[0].future.done()

        # Second message arrives while the first turn is still running.
        event2 = MessageEvent(text="second", source=source, message_id="om_2")
        handled = await runner._handle_active_session_busy_message(
            event2, "feishu:oc_chat_1"
        )
        assert handled is True

        for _ in range(20):
            await asyncio.sleep(0)
            if len(fake.submitted) >= 2:
                break

        # Both turns hold runtime submissions before the first completed.
        assert len(fake.submitted) == 2
        assert not fake.handles[0].future.done()
        turn_ids = [p["client_turn_id"] for p in fake.submitted]
        assert turn_ids[0] != turn_ids[1]
        assert turn_ids[0].endswith("om_1")
        assert turn_ids[1].endswith("om_2")
        assert fake.submitted[1]["busy_mode"] == "fifo"
        # No gateway-side pending-event growth.
        assert gateway_queued == []

        # Resolve both turns; the follow-up delivers its own reply.
        fake.handles[0].future.set_result(
            {"state": "completed", "final_text": "first reply",
             "already_delivered": False, "error": ""}
        )
        fake.handles[1].future.set_result(
            {"state": "completed", "final_text": "second reply",
             "already_delivered": False, "error": ""}
        )
        result1 = await asyncio.wait_for(turn1, timeout=2)
        assert result1["final_response"] == "first reply"

        for _ in range(20):
            await asyncio.sleep(0)
            if adapter._send_with_retry.await_count:
                break
        assert adapter._send_with_retry.await_count == 1
        sent_kwargs = adapter._send_with_retry.await_args.kwargs
        assert sent_kwargs["content"] == "second reply"
        assert sent_kwargs["chat_id"] == "oc_chat_1"

    @pytest.mark.asyncio
    async def test_busy_internal_event_not_forwarded_to_runtime(self, monkeypatch):
        """Internal synthetic events keep the pre-existing busy fallthrough
        (base adapter queues them silently) — never a delegated turn."""
        from unittest.mock import AsyncMock, MagicMock

        from gateway.platforms.base import MessageEvent

        _set_runtime_env(monkeypatch)
        runner = _make_runner(_delegate_cfg())
        runner._draining = False
        fake = _FakeRuntimeClient(resolve_immediately=False)
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)
        monkeypatch.setattr(runner, "_is_user_authorized", lambda source: True)

        adapter = MagicMock()
        adapter._send_with_retry = AsyncMock()
        runner.adapters = {Platform.FEISHU: adapter}

        event = MessageEvent(
            text="[background task finished]",
            source=_make_source(),
            internal=True,
        )
        handled = await runner._handle_active_session_busy_message(
            event, "feishu:oc_chat_1"
        )

        # Falls through to the base adapter's silent internal-event queueing.
        assert handled is False
        assert fake.submitted == []

    @pytest.mark.asyncio
    async def test_disabled_platform_does_not_delegate(self, monkeypatch):
        _set_runtime_env(monkeypatch)
        runner = _make_runner(_delegate_cfg(platforms=["feishu"]))
        fake = _FakeRuntimeClient()
        monkeypatch.setattr(runner, "_get_runtime_client", lambda: fake)

        try:
            await runner._run_agent(
                message="hi",
                context_prompt="",
                history=[],
                source=_make_source(Platform.TELEGRAM),
                session_id="sess-1",
                session_key="telegram:123",
            )
        except Exception:
            pass  # Expected — bare runner can't run the real inline path.

        assert fake.submitted == []


class TestServerShapedWireContract:
    """Wire-contract guard: the fake responses above must stay shaped like a
    REAL serve response — runtime_generation is `uuid4().hex` (an opaque
    string, usually non-numeric). Regression for the int() coercion that made
    every real submit fail with ValueError while numeric test fixtures
    stayed green."""

    @pytest.mark.asyncio
    async def test_submit_accepts_uuid_hex_runtime_generation(self, monkeypatch):
        client = GatewayRuntimeClient("ws://127.0.0.1:1/api/ws", "tok")
        fake_ws = _FakeWS()
        client._ws = fake_ws

        async def _noop():
            return None

        monkeypatch.setattr(client, "ensure_connected", _noop)
        submit_task = asyncio.create_task(
            client.submit_turn({"client_turn_id": "c-1", "message": "hi"})
        )
        await asyncio.sleep(0)
        frame = fake_ws.sent[0]
        client.handle_frame({
            "jsonrpc": "2.0",
            "id": frame["id"],
            "result": _submit_result(),
        })
        handle = await submit_task
        assert handle.runtime_generation == "5363e55bba1741cfb7cab44b09b2b22c"
        assert isinstance(handle.runtime_generation, str)
        # int(uuid_hex) raises — the client must never coerce.
        with pytest.raises(ValueError):
            int(handle.runtime_generation)


class TestDelegatedStopInterrupt:
    """/stop on a delegated route must send a DIRECTED session.interrupt to
    serve (turn id + runtime generation) — there is no local AIAgent to
    interrupt, and without this the model/tools keep running (§10.2)."""

    @pytest.mark.asyncio
    async def test_interrupt_delegated_runtime_turns_sends_directed_interrupt(self):
        from types import SimpleNamespace

        from gateway.run import GatewayRunner

        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        handle = SimpleNamespace(
            turn_id="t-9",
            runtime_session_id="rt-9",
            runtime_generation="5363e55bba1741cfb7cab44b09b2b22c",
            future=pending,
        )
        requests = []

        class _FakeClient:
            async def request(self, method, params, timeout=30):
                requests.append((method, params))
                return {"status": "interrupted"}

        runner = SimpleNamespace(
            _runtime_turn_handles={"sess-key": {"t-9": handle}},
            _get_runtime_client=lambda: _FakeClient(),
        )
        await GatewayRunner._interrupt_delegated_runtime_turns(runner, "sess-key")

        assert requests == [(
            "session.interrupt",
            {
                "session_id": "rt-9",
                "turn_id": "t-9",
                "runtime_generation": "5363e55bba1741cfb7cab44b09b2b22c",
            },
        )]

        # Resolved handles are skipped — no stray interrupts after completion.
        requests.clear()
        pending.set_result({"state": "completed"})
        await GatewayRunner._interrupt_delegated_runtime_turns(runner, "sess-key")
        assert requests == []
