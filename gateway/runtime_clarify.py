"""Clarify relay for runtime-delegated turns (shared session runtime).

A legacy gateway turn runs its agent locally: ``clarify_tool``'s callback
registers into :mod:`tools.clarify_gateway`, the adapter renders the button /
text prompt, and the next inbound message (or a button tap) resolves it. A
DELEGATED turn executes inside ``hermes serve``, whose blocking clarify bridge
emits a ``clarify.request`` event to the turn's delivery sink — this gateway.
Without a relay the question is invisible to everyone and the serve-side
clarify blocks until timeout (manual-acceptance regression: a 5-minute stall,
then the model answers on its own assumptions).

This module relays serve-side clarify prompts through the SAME
``tools.clarify_gateway`` registry the legacy path uses, so:

- the adapter renders the identical prompt UI (buttons or plain text),
- the inbound-message intercept in ``gateway.run`` resolves text answers,
- button callbacks resolve via ``resolve_gateway_clarify`` (clarify_id is the
  serve request_id),
- the resolution is forwarded to serve with ``clarify.respond``. The prompt is
  request_id-keyed on the serve side, so a Desktop answer can win the race —
  the RPC then reports "no pending request" and we simply drop ours.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Optional

module_logger = logging.getLogger(__name__)


def relay_clarify_request(
    *,
    payload: Dict[str, Any],
    session_key: str,
    adapter: Any,
    chat_id: str,
    metadata: Optional[Dict[str, Any]],
    loop: asyncio.AbstractEventLoop,
    client: Any,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Present a serve-raised clarify prompt on the origin channel.

    Runs on the gateway event loop (the runtime client's reader task). Never
    blocks: the adapter send is scheduled as a task and the answer wait runs
    on a daemon thread. Returns True when a relay was armed.
    """
    log = logger or module_logger
    request_id = str(payload.get("request_id") or "")
    question = str(payload.get("question") or "")
    raw_choices = payload.get("choices")
    choices = [str(c) for c in raw_choices] if isinstance(raw_choices, list) else None
    if not request_id or not question or adapter is None or not chat_id:
        return False

    from tools import clarify_gateway

    clarify_gateway.register(
        clarify_id=request_id,
        session_key=session_key or "",
        question=question,
        choices=choices,
    )

    # Pause typing so the prompt is not obscured (mirrors the legacy path).
    try:
        adapter.pause_typing_for_chat(chat_id)
    except Exception:
        pass

    async def _send() -> None:
        ok = False
        try:
            result = await adapter.send_clarify(
                chat_id=chat_id,
                question=question,
                choices=choices,
                clarify_id=request_id,
                session_key=session_key or "",
                metadata=metadata,
            )
            ok = bool(getattr(result, "success", False))
        except Exception as exc:
            log.warning("runtime clarify: prompt send failed: %s", exc)
        if not ok:
            # Could not deliver — release the pending entry with an empty
            # answer so the waiter exits and the next inbound message is NOT
            # swallowed as a clarify reply. Serve falls back to its own
            # timeout semantics.
            clarify_gateway.resolve_gateway_clarify(request_id, "")

    try:
        loop.create_task(_send())
    except Exception as exc:
        log.warning("runtime clarify: could not schedule prompt send: %s", exc)
        clarify_gateway.resolve_gateway_clarify(request_id, "")

    def _wait_and_forward() -> None:
        timeout = float(clarify_gateway.get_clarify_timeout())
        response = clarify_gateway.wait_for_response(request_id, timeout=timeout)
        if not response:
            # Timeout, delivery failure, or expired-elsewhere — serve owns
            # the fallback (its own clarify timeout / the winning answer).
            return
        future = asyncio.run_coroutine_threadsafe(
            client.respond_clarify(request_id, response), loop
        )
        try:
            future.result(timeout=15)
        except Exception as exc:
            # 4009 "no pending request" = someone else answered first.
            log.info("runtime clarify: respond not accepted (%s)", exc)

    threading.Thread(
        target=_wait_and_forward,
        daemon=True,
        name=f"runtime-clarify-{request_id}",
    ).start()
    return True


def relay_clarify_expire(
    payload: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Drop a pending relayed prompt: answered elsewhere or timed out on serve.

    Critical for the text-fallback intercept — without this the NEXT inbound
    channel message would be swallowed as the answer to a dead prompt.
    """
    request_id = str(payload.get("request_id") or "")
    if not request_id:
        return
    from tools import clarify_gateway

    if clarify_gateway.resolve_gateway_clarify(request_id, ""):
        (logger or module_logger).debug(
            "runtime clarify: prompt %s expired/resolved elsewhere", request_id
        )
