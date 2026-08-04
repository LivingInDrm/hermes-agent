"""Desktop Contract 5: managed provider routes use a fail-closed key broker."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli.myagents_providers import (
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    BrokerCredentialError,
    fetch_broker_api_key,
    resolve_broker_runtime,
)


class _Broker:
    def __init__(self, status: int = 200, api_key: str = "sk-from-broker"):
        self.status = status
        self.api_key = api_key
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server contract
                outer.requests.append(
                    (self.path, self.headers.get("Authorization"))
                )
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if outer.status == 200:
                    self.wfile.write(json.dumps({"api_key": outer.api_key}).encode())

            def log_message(self, *_args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/credential?generation=7"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def broker(monkeypatch):
    instance = _Broker()
    monkeypatch.setenv(BROKER_URL_ENV, instance.url)
    monkeypatch.setenv(BROKER_TOKEN_ENV, "capability-token-1")
    yield instance
    instance.close()


@pytest.fixture
def managed_route(tmp_path, monkeypatch):
    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / "config.yaml").write_text(
        """\
providers:
  myagents-pa-abc123:
    name: myagents-pa-abc123
    base_url: https://api.deepseek.com
    api_mode: chat_completions
    default_model: deepseek-chat
    max_output_tokens: 8192
    extra_headers:
      X-Route: desktop
    key_broker:
      account: pa-abc123
""",
        encoding="utf-8",
    )
    import hermes_cli.config as config
    from hermes_cli import managed_scope

    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    yield home
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()


def test_broker_roundtrip_preserves_existing_query(broker) -> None:
    assert fetch_broker_api_key({"account": "pa abc"}) == "sk-from-broker"
    assert broker.requests == [
        ("/credential?generation=7&account=pa+abc", "Bearer capability-token-1")
    ]


def test_broker_failures_are_closed_and_secret_free(monkeypatch, broker) -> None:
    broker.status = 403
    with pytest.raises(BrokerCredentialError, match="HTTP 403") as caught:
        fetch_broker_api_key({"account": "pa-abc123"})
    assert "capability-token-1" not in str(caught.value)

    monkeypatch.delenv(BROKER_URL_ENV)
    with pytest.raises(BrokerCredentialError, match="lease env missing"):
        fetch_broker_api_key({"account": "pa-abc123"})


def test_runtime_shape_keeps_route_options(broker) -> None:
    resolved = resolve_broker_runtime(
        {
            "name": "route",
            "api_mode": "anthropic_messages",
            "model": "model-1",
            "max_output_tokens": 4096,
            "extra_headers": {"X-Route": "desktop"},
            "key_broker": {"account": "pa-abc123"},
        },
        "https://example.test",
    )
    assert resolved["api_key"] == "sk-from-broker"
    assert resolved["model"] == "model-1"
    assert resolved["max_output_tokens"] == 4096
    assert resolved["extra_headers"] == {"X-Route": "desktop"}


def test_managed_route_resolves_via_broker_only(
    managed_route, monkeypatch, broker
) -> None:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ambient-must-not-win")
    resolved = resolve_runtime_provider(requested="custom:myagents-pa-abc123")

    assert resolved["api_key"] == "sk-from-broker"
    assert resolved["base_url"] == "https://api.deepseek.com"
    assert resolved["model"] == "deepseek-chat"
    assert resolved["max_output_tokens"] == 8192
    assert resolved["extra_headers"] == {"X-Route": "desktop"}
    assert resolved["source"] == "myagents_broker:myagents-pa-abc123"


def test_managed_route_does_not_fallback_when_broker_refuses(
    managed_route, monkeypatch, broker
) -> None:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    broker.status = 401
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ambient-must-not-win")
    with pytest.raises(BrokerCredentialError):
        resolve_runtime_provider(requested="custom:myagents-pa-abc123")

