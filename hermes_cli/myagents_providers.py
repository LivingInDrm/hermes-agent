"""MyAgents credential-broker runtime support (Desktop Contract 5).

This downstream leaf intentionally imports only the Python standard library.
Hermes' upstream managed scope owns provider configuration; this module only
turns a non-sensitive ``key_broker`` marker into an in-memory credential.

Broker-backed routes fail closed. They never fall back to environment keys,
profile config, or credential pools, because doing so could silently change the
billing account selected by the desktop. Capability tokens and credentials must
never appear in logs or exception messages.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


BROKER_URL_ENV = "MYAGENTS_RUNTIME_BROKER_URL"
BROKER_TOKEN_ENV = "MYAGENTS_RUNTIME_BROKER_TOKEN"

_BROKER_TIMEOUT_SECONDS = 5.0
_BROKER_RESPONSE_LIMIT = 64 * 1024


class BrokerCredentialError(RuntimeError):
    """A broker credential is unavailable; messages contain categories only."""


def _credential_url(base_url: str, account: str, purpose: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("account", account))
    query.append(("purpose", purpose))
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )


def fetch_broker_api_key(key_broker: dict, purpose: str = "model") -> str:
    """Fetch the current credential for a broker-backed route; fail closed."""
    account = str((key_broker or {}).get("account") or "").strip()
    if not account:
        raise BrokerCredentialError("credential broker: route has no account id")
    if purpose not in {"model", "voice"}:
        raise BrokerCredentialError("credential broker: invalid credential purpose")

    url = os.environ.get(BROKER_URL_ENV, "").strip()
    token = os.environ.get(BROKER_TOKEN_ENV, "").strip()
    if not url or not token:
        raise BrokerCredentialError(
            "credential broker: lease env missing; backend was not spawned by the desktop"
        )

    request = urllib.request.Request(
        _credential_url(url, account, purpose),
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=_BROKER_TIMEOUT_SECONDS) as response:
            raw = response.read(_BROKER_RESPONSE_LIMIT + 1)
        if len(raw) > _BROKER_RESPONSE_LIMIT:
            raise BrokerCredentialError("credential broker response too large")
        body = json.loads(raw.decode("utf-8"))
    except BrokerCredentialError:
        raise
    except urllib.error.HTTPError as exc:
        raise BrokerCredentialError(f"credential broker refused (HTTP {exc.code})") from None
    except Exception as exc:  # network, timeout, decoding, or invalid JSON
        raise BrokerCredentialError(
            f"credential broker unreachable: {type(exc).__name__}"
        ) from None

    api_key = str(body.get("api_key") or "").strip() if isinstance(body, dict) else ""
    if not api_key:
        raise BrokerCredentialError("credential broker returned no credential")
    return api_key


def resolve_broker_runtime(
    custom_provider: Dict[str, Any],
    base_url: str,
    requested_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Hermes runtime using only the desktop broker credential."""
    result: Dict[str, Any] = {
        "provider": "custom",
        "api_mode": custom_provider.get("api_mode") or "chat_completions",
        "base_url": base_url,
        "api_key": fetch_broker_api_key(custom_provider.get("key_broker") or {}, purpose="model"),
        "source": f"myagents_broker:{custom_provider.get('name', requested_provider or '')}",
    }
    if custom_provider.get("model"):
        result["model"] = custom_provider["model"]
    if isinstance(custom_provider.get("max_output_tokens"), int):
        result["max_output_tokens"] = custom_provider["max_output_tokens"]
    if custom_provider.get("extra_headers"):
        result["extra_headers"] = dict(custom_provider["extra_headers"])
    return result
