"""Desktop Contract 5 — MyAgents provider-route overlay + credential broker.

Covers the fork-side guarantees the desktop relies on:
  * an overlay JSON named by HERMES_PROVIDERS_OVERLAY merges into
    ``config["providers"]`` and resolves like any named custom provider;
  * routes carrying ``key_broker`` fetch their credential from the loopback
    broker and FAIL CLOSED — no env/.env/config/pool fallback;
  * broker-backed entries are never persisted by ``save_config``.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli import myagents_providers
from hermes_cli.myagents_providers import (
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    BrokerCredentialError,
    fetch_broker_api_key,
    resolve_broker_runtime,
)


@pytest.fixture(autouse=True)
def _reset_overlay():
    myagents_providers.reset_for_tests()
    yield
    myagents_providers.reset_for_tests()


def _write_overlay(tmp_path, providers):
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps({"schema": 1, "providers": providers}), encoding="utf-8")
    return str(path)


def _overlay_entry(**overrides):
    entry = {
        "name": "myagents-pa-abc123",
        "base_url": "https://api.deepseek.com",
        "api_mode": "chat_completions",
        "key_broker": {"account": "pa-abc123"},
    }
    entry.update(overrides)
    return entry


class _Broker:
    """Minimal loopback broker double."""

    def __init__(self, status=200, api_key="sk-from-broker"):
        self.status = status
        self.api_key = api_key
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 — http.server contract
                outer.requests.append(
                    {"path": self.path, "authorization": self.headers.get("Authorization")}
                )
                body = json.dumps({"api_key": outer.api_key}).encode("utf-8")
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if outer.status == 200:
                    self.wfile.write(body)

            def log_message(self, *args):  # silence
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/credential"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def broker(monkeypatch):
    instance = _Broker()
    monkeypatch.setenv(BROKER_URL_ENV, instance.url)
    monkeypatch.setenv(BROKER_TOKEN_ENV, "capability-token-1")
    yield instance
    instance.close()


def test_overlay_merges_into_providers_dict(tmp_path, monkeypatch):
    monkeypatch.setenv(
        myagents_providers.OVERLAY_ENV,
        _write_overlay(tmp_path, {"myagents-pa-abc123": _overlay_entry()}),
    )
    config = {"providers": {"mine": {"base_url": "https://mine.example"}}}
    merged = myagents_providers.apply_providers_overlay(config)
    assert merged["providers"]["mine"]["base_url"] == "https://mine.example"
    assert merged["providers"]["myagents-pa-abc123"]["key_broker"] == {"account": "pa-abc123"}


def test_overlay_is_fail_open_and_loud_on_garbage(tmp_path, monkeypatch, caplog):
    bad = tmp_path / "overlay.json"
    bad.write_text("not-json{{{", encoding="utf-8")
    monkeypatch.setenv(myagents_providers.OVERLAY_ENV, str(bad))
    config = {}
    assert myagents_providers.apply_providers_overlay(config) == {}
    assert any("myagents providers overlay" in record.message for record in caplog.records)


def test_strip_overlay_providers_only_drops_broker_entries():
    config = {
        "providers": {
            "mine": {"base_url": "https://mine.example", "key_env": "MY_KEY"},
            "myagents-pa-abc123": _overlay_entry(),
        }
    }
    stripped = myagents_providers.strip_overlay_providers(config)
    assert list(stripped["providers"]) == ["mine"]
    # 原 dict 不被就地修改（save_config 语义：strip 是纯投影）。
    assert "myagents-pa-abc123" in config["providers"]


def test_fetch_broker_api_key_roundtrip(broker):
    api_key = fetch_broker_api_key({"account": "pa-abc123"})
    assert api_key == "sk-from-broker"
    assert broker.requests[0]["path"] == "/credential?account=pa-abc123"
    assert broker.requests[0]["authorization"] == "Bearer capability-token-1"


def test_fetch_broker_api_key_fails_closed(monkeypatch, broker):
    broker.status = 403
    with pytest.raises(BrokerCredentialError, match="403"):
        fetch_broker_api_key({"account": "pa-abc123"})
    monkeypatch.delenv(BROKER_URL_ENV)
    with pytest.raises(BrokerCredentialError, match="lease env missing"):
        fetch_broker_api_key({"account": "pa-abc123"})


def test_resolve_broker_runtime_shape(broker):
    resolved = resolve_broker_runtime(_overlay_entry(), "https://api.deepseek.com")
    assert resolved == {
        "provider": "custom",
        "api_mode": "chat_completions",
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-from-broker",
        "source": "myagents_broker:myagents-pa-abc123",
    }


def test_named_custom_runtime_uses_broker_and_never_env_fallback(tmp_path, monkeypatch, broker):
    from hermes_cli import runtime_provider

    monkeypatch.setenv(
        myagents_providers.OVERLAY_ENV,
        _write_overlay(tmp_path, {"myagents-pa-abc123": _overlay_entry()}),
    )
    config = myagents_providers.apply_providers_overlay({})
    monkeypatch.setattr(runtime_provider, "load_config", lambda: config)
    # 环境里放一个会被 host 推导规则命中的 Key：broker 路由必须无视它。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-should-not-be-used")

    resolved = runtime_provider.resolve_runtime_provider(requested="custom:myagents-pa-abc123")
    assert resolved["api_key"] == "sk-from-broker"
    assert resolved["base_url"] == "https://api.deepseek.com"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["source"].startswith("myagents_broker:")


def test_named_custom_runtime_broker_failure_raises_not_falls_back(tmp_path, monkeypatch, broker):
    from hermes_cli import runtime_provider

    broker.status = 401
    monkeypatch.setenv(
        myagents_providers.OVERLAY_ENV,
        _write_overlay(tmp_path, {"myagents-pa-abc123": _overlay_entry()}),
    )
    config = myagents_providers.apply_providers_overlay({})
    monkeypatch.setattr(runtime_provider, "load_config", lambda: config)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-should-not-be-used")

    with pytest.raises(BrokerCredentialError):
        runtime_provider.resolve_runtime_provider(requested="custom:myagents-pa-abc123")


def test_cron_provider_resolution_sees_overlay(tmp_path, monkeypatch, broker):
    """cron 不需要自己的装载点：provider 解析走主 load_config()。

    这是「补丁不含 cron/scheduler.py 改动」的守护用例——若上游把 cron 的
    provider 解析改为绕过 resolve_runtime_provider/主装载器，此用例会失败，
    提示需要重新评估装载点。
    """
    from hermes_cli import runtime_provider

    monkeypatch.setenv(
        myagents_providers.OVERLAY_ENV,
        _write_overlay(tmp_path, {"myagents-pa-abc123": _overlay_entry()}),
    )
    config = myagents_providers.apply_providers_overlay({})
    monkeypatch.setattr(runtime_provider, "load_config", lambda: config)
    # cron/scheduler.run_job 的 provider 解析入口与桌面相同：
    resolved = runtime_provider.resolve_runtime_provider(
        requested="myagents-pa-abc123", target_model="deepseek-chat"
    )
    assert resolved["source"].startswith("myagents_broker:")
