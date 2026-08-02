"""MyAgents desktop model routes: provider overlay + credential broker client.

This is a DOWNSTREAM LEAF MODULE (Desktop Contract 5). It is self-contained on
purpose: stdlib imports only, no imports from any hermes module, so it tracks
no upstream API and carries zero rebase conflict surface. The only upstream
coupling is three small call sites (``hermes_cli/config.py`` load/save hooks
and the ``key_broker`` branch in ``hermes_cli/runtime_provider.py``).

How it works
============

The MyAgents desktop supervisor manages Provider accounts and credentials at
the software level (one account shared by every profile). Per backend
generation it projects the enabled accounts into a small JSON file and spawns
``hermes serve`` with:

  * ``HERMES_PROVIDERS_OVERLAY`` — path to the overlay JSON. Entries merge
    into ``config["providers"]`` (the new-style user provider dict), so the
    runtime resolver, the picker and per-session ``/model … --provider …
    --session`` switching all see them with no second code path.
  * ``MYAGENTS_MODEL_BROKER_URL`` / ``MYAGENTS_MODEL_BROKER_TOKEN`` — a
    loopback credential broker plus a capability token scoped to this backend
    generation. Routes carry ``key_broker: {account: <opaque id>}`` instead of
    a key; the credential is fetched at resolution time and exists only in
    this process's memory.

Invariants
==========

  * The overlay is NON-SENSITIVE: no keys, no tokens — only base_url,
    api_mode and the opaque account id.
  * Overlay path and content never change within one process lifetime (the
    desktop writes a fresh file per generation before spawning), so a single
    module-level read is correct and the config cache signature needs no
    extension.
  * Broker-backed routes FAIL CLOSED. ``resolve_broker_runtime`` never falls
    back to env vars, ``.env``, config keys or the credential pool — any
    fallback would silently switch the billing account behind the user's
    explicit selection. A broker failure raises ``BrokerCredentialError``.
  * Broker-backed entries are never persisted: ``strip_overlay_providers`` is
    called by ``save_config`` so a transient projection can't leak into the
    durable profile config.
  * Overlay parse errors are fail-open with a LOUD log: a broken overlay must
    not brick startup — the routes are simply absent and the desktop surfaces
    the corresponding models as blocked.
  * Neither the capability token nor any fetched credential may ever be
    logged. Error messages carry categories only.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

OVERLAY_ENV = "HERMES_PROVIDERS_OVERLAY"
BROKER_URL_ENV = "MYAGENTS_MODEL_BROKER_URL"
BROKER_TOKEN_ENV = "MYAGENTS_MODEL_BROKER_TOKEN"

_BROKER_TIMEOUT_SECONDS = 5.0

_LOCK = threading.Lock()
_LOADED = False
_PROVIDERS: Dict[str, Dict[str, Any]] = {}


class BrokerCredentialError(RuntimeError):
    """Broker-backed credential unavailable. Message carries categories only."""


# ---------------------------------------------------------------------------
# Overlay: load once per process, merge into config["providers"].
# ---------------------------------------------------------------------------

def _load_once() -> Dict[str, Dict[str, Any]]:
    global _LOADED, _PROVIDERS
    if _LOADED:
        return _PROVIDERS
    with _LOCK:
        if _LOADED:
            return _PROVIDERS
        providers: Dict[str, Dict[str, Any]] = {}
        path = os.environ.get(OVERLAY_ENV, "").strip()
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("providers") if isinstance(data, dict) else None
                if data.get("schema") == 1 and isinstance(entries, dict):
                    for name, entry in entries.items():
                        if (
                            isinstance(name, str)
                            and name.strip()
                            and isinstance(entry, dict)
                            and str(entry.get("base_url") or "").strip()
                        ):
                            providers[name.strip()] = dict(entry)
                else:
                    logger.warning(
                        "myagents providers overlay: %s has an unsupported shape — "
                        "IGNORING. Desktop-managed model routes are ABSENT this run.",
                        path,
                    )
            except Exception as exc:  # noqa: BLE001 — fail-open, but LOUD
                logger.warning(
                    "myagents providers overlay: failed to load %s: %s — IGNORING. "
                    "Desktop-managed model routes are ABSENT this run.",
                    path,
                    exc,
                )
        _PROVIDERS = providers
        _LOADED = True
        return _PROVIDERS


def overlay_providers() -> Dict[str, Dict[str, Any]]:
    """A defensive copy of the overlay's provider entries (may be empty)."""
    return {name: dict(entry) for name, entry in _load_once().items()}


def apply_providers_overlay(config: dict) -> dict:
    """Merge desktop-projected provider routes into ``config["providers"]``.

    Overlay entries win over same-named user entries; the desktop prefixes
    route names with ``myagents-`` so collisions do not occur in practice.
    Mutates and returns ``config`` (callers pass a dict they own). Must never
    raise — an overlay problem degrades to "routes absent", not a crash.
    """
    try:
        providers = _load_once()
        if not providers:
            return config
        existing = config.get("providers")
        merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        for name, entry in providers.items():
            merged[name] = dict(entry)
        config["providers"] = merged
        return config
    except Exception:  # noqa: BLE001 — overlay must never break config loading
        logger.warning("myagents providers overlay: failed to apply", exc_info=True)
        return config


def strip_overlay_providers(config: dict) -> dict:
    """Drop broker-backed provider entries before persisting config.

    Overlay routes are per-generation projections; persisting one would leave
    a stale entry whose broker lease is gone (fails closed on every use) and
    would shadow the user's own providers. Anything carrying ``key_broker``
    is by definition transient — strip it regardless of whether it came from
    the currently-loaded overlay. Returns a shallow-adjusted copy; the input
    dict is not mutated.
    """
    providers = config.get("providers")
    if not isinstance(providers, dict):
        return config
    kept = {
        name: entry
        for name, entry in providers.items()
        if not (isinstance(entry, dict) and entry.get("key_broker"))
    }
    if len(kept) != len(providers):
        config = dict(config)
        if kept:
            config["providers"] = kept
        else:
            config.pop("providers", None)
    return config


def reset_for_tests() -> None:
    global _LOADED, _PROVIDERS
    with _LOCK:
        _LOADED = False
        _PROVIDERS = {}


# ---------------------------------------------------------------------------
# Credential broker client + runtime resolution for broker-backed routes.
# ---------------------------------------------------------------------------

def fetch_broker_api_key(key_broker: dict) -> str:
    """Fetch the current credential for a broker-backed route. FAIL CLOSED."""
    account = str((key_broker or {}).get("account") or "").strip()
    if not account:
        raise BrokerCredentialError("credential broker: route has no account id")
    url = os.environ.get(BROKER_URL_ENV, "").strip()
    token = os.environ.get(BROKER_TOKEN_ENV, "").strip()
    if not url or not token:
        raise BrokerCredentialError(
            "credential broker: lease env missing — this backend was not spawned by the desktop"
        )
    request = urllib.request.Request(
        f"{url}?account={urllib.parse.quote(account, safe='')}",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=_BROKER_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 401 = lease revoked/rotated; 403 = account not leased / disabled /
        # disconnected. Either way the desktop is the place to fix it.
        raise BrokerCredentialError(f"credential broker refused (HTTP {exc.code})") from None
    except Exception as exc:  # noqa: BLE001 — network/timeout/JSON, no secrets in message
        raise BrokerCredentialError(f"credential broker unreachable: {type(exc).__name__}") from None
    api_key = str(body.get("api_key") or "").strip() if isinstance(body, dict) else ""
    if not api_key:
        raise BrokerCredentialError("credential broker returned no credential")
    return api_key


def resolve_broker_runtime(
    custom_provider: Dict[str, Any],
    base_url: str,
    requested_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the resolved-runtime dict for a broker-backed route.

    Lives here (not in ``runtime_provider.py``) so the upstream file only
    carries a tiny dispatch branch. Deliberately does NOT consult the api_key
    candidate chain, the credential pool or ``_host_derived_api_key`` — see
    the fail-closed invariant in the module docstring. The overlay always
    sets ``api_mode`` explicitly; ``chat_completions`` is only a last-resort
    default for hand-written entries.
    """
    api_key = fetch_broker_api_key(custom_provider.get("key_broker") or {})
    result: Dict[str, Any] = {
        "provider": "custom",
        "api_mode": custom_provider.get("api_mode") or "chat_completions",
        "base_url": base_url,
        "api_key": api_key,
        "source": f"myagents_broker:{custom_provider.get('name', requested_provider or '')}",
    }
    if custom_provider.get("model"):
        result["model"] = custom_provider["model"]
    if isinstance(custom_provider.get("max_output_tokens"), int):
        result["max_output_tokens"] = custom_provider["max_output_tokens"]
    if custom_provider.get("extra_headers"):
        result["extra_headers"] = dict(custom_provider["extra_headers"])
    return result
