"""MiniMax TTS region, endpoint, and credential selection tests."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.tts_tool import (
    DEFAULT_MINIMAX_BASE_URL,
    DEFAULT_MINIMAX_CN_BASE_URL,
    _generate_minimax_tts,
    _resolve_minimax_tts_runtime,
    check_tts_requirements,
)
from hermes_cli.myagents_providers import BrokerCredentialError


GLOBAL_CREDENTIAL_SENTINEL = "FAKE_GLOBAL_CREDENTIAL"
CN_CREDENTIAL_SENTINEL = "FAKE_CN_CREDENTIAL"


@pytest.fixture(autouse=True)
def _fake_minimax_credentials(monkeypatch):
    values = {}
    monkeypatch.setattr(
        "tools.tts_tool.get_env_value",
        lambda name, default=None: values.get(name, default),
    )
    return values


@pytest.mark.parametrize(
    ("config", "credentials", "expected"),
    [
        pytest.param(
            {},
            {"MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL},
            (
                "global",
                DEFAULT_MINIMAX_BASE_URL,
                "MINIMAX_API_KEY",
                GLOBAL_CREDENTIAL_SENTINEL,
            ),
            id="global-only",
        ),
        pytest.param(
            {},
            {"MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL},
            (
                "cn",
                DEFAULT_MINIMAX_CN_BASE_URL,
                "MINIMAX_CN_API_KEY",
                CN_CREDENTIAL_SENTINEL,
            ),
            id="china-only",
        ),
        pytest.param(
            {},
            {
                "MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL,
                "MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL,
            },
            (
                "global",
                DEFAULT_MINIMAX_BASE_URL,
                "MINIMAX_API_KEY",
                GLOBAL_CREDENTIAL_SENTINEL,
            ),
            id="both-default-to-global",
        ),
        pytest.param(
            {"minimax": {"region": "global"}},
            {
                "MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL,
                "MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL,
            },
            (
                "global",
                DEFAULT_MINIMAX_BASE_URL,
                "MINIMAX_API_KEY",
                GLOBAL_CREDENTIAL_SENTINEL,
            ),
            id="explicit-global",
        ),
        pytest.param(
            {"minimax": {"region": "cn"}},
            {
                "MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL,
                "MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL,
            },
            (
                "cn",
                DEFAULT_MINIMAX_CN_BASE_URL,
                "MINIMAX_CN_API_KEY",
                CN_CREDENTIAL_SENTINEL,
            ),
            id="explicit-china",
        ),
    ],
)
def test_runtime_selection_matrix(
    _fake_minimax_credentials,
    config,
    credentials,
    expected,
):
    _fake_minimax_credentials.update(credentials)

    runtime = _resolve_minimax_tts_runtime(config)

    assert (
        runtime.region,
        runtime.endpoint,
        runtime.credential_source,
        runtime.api_key,
    ) == expected


@pytest.mark.parametrize(
    ("region", "credentials", "missing_source"),
    [
        pytest.param(
            "global",
            {"MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL},
            "MINIMAX_API_KEY",
            id="global-does-not-borrow-china-key",
        ),
        pytest.param(
            "cn",
            {"MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL},
            "MINIMAX_CN_API_KEY",
            id="china-does-not-borrow-global-key",
        ),
    ],
)
def test_explicit_region_requires_matching_credential(
    _fake_minimax_credentials,
    region,
    credentials,
    missing_source,
):
    _fake_minimax_credentials.update(credentials)

    with pytest.raises(ValueError, match=missing_source):
        _resolve_minimax_tts_runtime({"minimax": {"region": region}})


@pytest.mark.parametrize(
    ("config", "credentials", "expected"),
    [
        pytest.param(
            {"provider": "minimax"},
            {"MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL},
            True,
            id="china-only-available",
        ),
        pytest.param(
            {"provider": "minimax", "minimax": {"region": "cn"}},
            {"MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL},
            False,
            id="selected-region-missing",
        ),
        pytest.param(
            {"provider": "minimax", "minimax": {"region": "invalid"}},
            {
                "MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL,
                "MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL,
            },
            False,
            id="invalid-region",
        ),
    ],
)
def test_availability_uses_atomic_runtime(
    monkeypatch,
    _fake_minimax_credentials,
    config,
    credentials,
    expected,
):
    _fake_minimax_credentials.update(credentials)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)

    assert check_tts_requirements() is expected


def test_runtime_repr_excludes_raw_credential(_fake_minimax_credentials):
    _fake_minimax_credentials["MINIMAX_API_KEY"] = GLOBAL_CREDENTIAL_SENTINEL

    runtime = _resolve_minimax_tts_runtime({})

    assert GLOBAL_CREDENTIAL_SENTINEL not in repr(runtime)


@pytest.mark.parametrize(
    ("region", "expected_endpoint", "expected_env"),
    [
        ("global", DEFAULT_MINIMAX_BASE_URL, "MINIMAX_API_KEY"),
        ("cn", DEFAULT_MINIMAX_CN_BASE_URL, "MINIMAX_CN_API_KEY"),
    ],
)
def test_broker_runtime_binds_region_endpoint_and_voice_purpose(
    monkeypatch, region, expected_endpoint, expected_env
):
    calls = []

    def resolve(env_var, provider_id, key_broker=None):
        calls.append((env_var, provider_id, key_broker))
        return "VOICE_BROKER_CREDENTIAL"

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", resolve)
    runtime = _resolve_minimax_tts_runtime({
        "minimax": {"region": region, "key_broker": {"account": "pa-voice"}}
    })

    assert runtime.endpoint == expected_endpoint
    assert runtime.credential_source == "MYAGENTS_RUNTIME_BROKER"
    assert calls == [(expected_env, "minimax", {"account": "pa-voice"})]
    assert "VOICE_BROKER_CREDENTIAL" not in repr(runtime)


def test_broker_failure_never_falls_back_to_environment(monkeypatch):
    calls = []

    def denied(_env_var, _provider_id, key_broker=None):
        calls.append(key_broker)
        raise BrokerCredentialError("credential broker refused (HTTP 403)")

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", denied)
    monkeypatch.setenv("MINIMAX_API_KEY", "AMBIENT_KEY_MUST_NOT_WIN")
    with pytest.raises(BrokerCredentialError, match="HTTP 403"):
        _resolve_minimax_tts_runtime({
            "minimax": {"region": "global", "key_broker": {"account": "pa-voice"}}
        })
    assert calls == [{"account": "pa-voice"}]


def test_t2a_payload_forwards_controlled_language_boost(monkeypatch, tmp_path):
    response = MagicMock()
    response.json.return_value = {
        "base_resp": {"status_code": 0},
        "data": {"audio": "0102"},
    }
    response.content = b'{"ok":true}'
    post = MagicMock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr(
        "tools.tts_tool._read_tts_response_json",
        lambda _response, label: {
            "base_resp": {"status_code": 0},
            "data": {"audio": "0102"},
        },
    )
    monkeypatch.setattr(
        "tools.tts_tool._resolve_minimax_tts_runtime",
        lambda _config: type("Runtime", (), {
            "endpoint": DEFAULT_MINIMAX_BASE_URL,
            "api_key": "VOICE_BROKER_CREDENTIAL",
        })(),
    )
    output = tmp_path / "voice.mp3"

    _generate_minimax_tts("中英 mixed", str(output), {
        "minimax": {
            "model": "speech-2.8-hd",
            "voice_id": "Chinese (Mandarin)_Warm_Girl",
            "language_boost": "auto",
        }
    })

    assert post.call_args.kwargs["json"]["language_boost"] == "auto"
    assert post.call_args.kwargs["json"]["model"] == "speech-2.8-hd"
    assert output.read_bytes() == b"\x01\x02"


def test_minimax_retries_one_transient_connection_failure(monkeypatch, tmp_path):
    response = MagicMock()
    post = MagicMock(side_effect=[
        requests.exceptions.ConnectionError("remote disconnected"),
        response,
    ])
    sleep = MagicMock()
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("tools.tts_tool.time.sleep", sleep)
    monkeypatch.setattr(
        "tools.tts_tool._read_tts_response_json",
        lambda _response, label: {
            "base_resp": {"status_code": 0},
            "data": {"audio": "0102"},
        },
    )
    monkeypatch.setattr(
        "tools.tts_tool._resolve_minimax_tts_runtime",
        lambda _config: type("Runtime", (), {
            "endpoint": DEFAULT_MINIMAX_BASE_URL,
            "api_key": "VOICE_BROKER_CREDENTIAL",
        })(),
    )
    output = tmp_path / "voice.mp3"

    _generate_minimax_tts("hello", str(output), {})

    assert post.call_count == 2
    sleep.assert_called_once_with(0.5)
    assert output.read_bytes() == b"\x01\x02"


def test_minimax_retries_documented_transient_api_error(monkeypatch, tmp_path):
    post = MagicMock(return_value=MagicMock())
    responses = iter([
        {"base_resp": {"status_code": 1001, "status_msg": "request timeout"}},
        {"base_resp": {"status_code": 0}, "data": {"audio": "0102"}},
    ])
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("tools.tts_tool.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "tools.tts_tool._read_tts_response_json",
        lambda _response, label: next(responses),
    )
    monkeypatch.setattr(
        "tools.tts_tool._resolve_minimax_tts_runtime",
        lambda _config: type("Runtime", (), {
            "endpoint": DEFAULT_MINIMAX_BASE_URL,
            "api_key": "VOICE_BROKER_CREDENTIAL",
        })(),
    )

    _generate_minimax_tts("hello", str(tmp_path / "voice.mp3"), {})

    assert post.call_count == 2


def test_minimax_does_not_retry_permanent_api_error(monkeypatch, tmp_path):
    post = MagicMock(return_value=MagicMock())
    sleep = MagicMock()
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("tools.tts_tool.time.sleep", sleep)
    monkeypatch.setattr(
        "tools.tts_tool._read_tts_response_json",
        lambda _response, label: {
            "base_resp": {"status_code": 2013, "status_msg": "invalid params"},
        },
    )
    monkeypatch.setattr(
        "tools.tts_tool._resolve_minimax_tts_runtime",
        lambda _config: type("Runtime", (), {
            "endpoint": DEFAULT_MINIMAX_BASE_URL,
            "api_key": "VOICE_BROKER_CREDENTIAL",
        })(),
    )

    with pytest.raises(RuntimeError, match="code 2013"):
        _generate_minimax_tts("hello", str(tmp_path / "voice.mp3"), {})

    post.assert_called_once()
    sleep.assert_not_called()


def test_invalid_language_boost_is_rejected_before_http(monkeypatch, tmp_path):
    post = MagicMock()
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr(
        "tools.tts_tool._resolve_minimax_tts_runtime",
        lambda _config: type("Runtime", (), {
            "endpoint": DEFAULT_MINIMAX_BASE_URL,
            "api_key": "VOICE_BROKER_CREDENTIAL",
        })(),
    )

    with pytest.raises(ValueError, match="language_boost"):
        _generate_minimax_tts("text", str(tmp_path / "voice.mp3"), {
            "minimax": {"language_boost": "not-a-language"}
        })
    post.assert_not_called()
