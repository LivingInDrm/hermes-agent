"""
WebSocket JSON-RPC client for gateway → ``hermes serve`` turn delegation.

Shared-session runtime, gateway-delegation milestone (design:
docs/design/channel-desktop-shared-session-runtime.md §5.2/§6.1/§7).  When
``gateway.runtime_delegate`` enables a platform+route, the messaging gateway
stops running a local AIAgent for normal inbound chat turns and submits the
prepared turn to the profile's ``hermes serve`` runtime over its ``/api/ws``
JSON-RPC endpoint (``turn.submit``).  While the turn is active, serve fans
the session's events back out over this same connection (the per-turn
delivery sink); the gateway collects the final assistant text and delivers
it to the platform through its normal delivery path.

Protocol summary (serve side is already implemented — see
hermes_cli/web_server.py and tui_gateway/):

- Connect: ``ws://<host>:<port>/api/ws?token=<service token>``.  The token
  is the gateway service token (env ``HERMES_DESKTOP_RUNTIME_TOKEN``); a WS
  presenting it gets authority "gateway-service".
- Requests are JSON-RPC 2.0 frames ``{"jsonrpc","id","method","params"}``
  answered by ``{"id", "result" | "error": {code, message}}``.
- Server-initiated events arrive as ``{"method": "event", "params":
  {"type", "session_id", "payload", "turn": {"turn_id", ...}}}`` — the
  ``turn`` sibling key attributes mid-turn events (message.delta /
  message.complete / session.info / turn.*) to the active turn.

Failure semantics (design §12.3): a disconnect fails pending REQUEST futures
with a retryable :class:`RuntimeConnectionError`, but an in-flight TURN whose
events were lost resolves with state ``"unknown"`` — this client never
auto-replays a turn.

The reader loop routes every inbound frame through the pure
:meth:`GatewayRuntimeClient.handle_frame` method so unit tests can feed
frames directly without a socket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

module_logger = logging.getLogger(__name__)

# JSON-RPC error codes surfaced by serve's turn.submit (contract 5).
RUNTIME_QUEUE_FULL_CODE = 4290      # per-session turn queue full — retryable busy
RUNTIME_SESSION_NOT_FOUND_CODE = 4007  # stored session id unknown to serve

# Terminal turn lifecycle events → outcome state.
_TURN_TERMINAL_STATES = {
    "turn.completed": "completed",
    "turn.interrupted": "interrupted",
    "turn.failed": "failed",
}


class RuntimeClientError(Exception):
    """Base error for the gateway runtime client."""


class RuntimeConnectionError(RuntimeClientError):
    """Transport-level failure (connect/send/disconnect). Retryable."""


class RuntimeRequestError(RuntimeClientError):
    """JSON-RPC error response from serve (carries the wire error code)."""

    def __init__(self, code: int, message: str):
        super().__init__(f"runtime rpc error {code}: {message}")
        self.code = code
        self.message = message


@dataclass
class RuntimeTurnHandle:
    """A submitted turn awaiting its outcome.

    ``future`` resolves with ``{"state": "completed" | "interrupted" |
    "failed" | "unknown", "final_text": str, "already_delivered": bool,
    "error": str}``.  ``final_text`` prefers the ``message.complete``
    payload text and falls back to the accumulated deltas.
    """

    turn_id: str
    sequence: int = 0
    stored_session_id: str = ""
    lineage_id: str = ""
    runtime_session_id: str = ""
    runtime_generation: str = ""
    queue_position: int = 0
    duplicate: bool = False
    status: str = ""
    future: Optional["asyncio.Future"] = None
    deltas: List[str] = field(default_factory=list)
    final_text: str = ""

    def resolve(
        self,
        state: str,
        *,
        already_delivered: bool = False,
        error: str = "",
    ) -> None:
        """Resolve the outcome future exactly once (later calls are no-ops)."""
        future = self.future
        if future is None or future.done():
            return
        future.set_result(
            {
                "state": state,
                "final_text": self.final_text or "".join(self.deltas),
                "already_delivered": already_delivered,
                "error": error,
            }
        )


class GatewayRuntimeClient:
    """One WS connection to a ``hermes serve`` runtime, shared across turns.

    Owns a single aiohttp ``ClientSession`` + WebSocket and a reader task.
    Connection is lazy (:meth:`ensure_connected`) and re-established on the
    next request after a disconnect.  All frame interpretation lives in
    :meth:`handle_frame` (pure, socket-free) so it can be unit tested by
    feeding frames directly.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        profile: str = "",
        logger: Optional[logging.Logger] = None,
        on_delta: Optional[Callable[["RuntimeTurnHandle", str], None]] = None,
        on_tip_updated: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_clarify_request: Optional[
            Callable[["RuntimeTurnHandle", Dict[str, Any]], None]
        ] = None,
        on_clarify_expire: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._url = url
        self._token = token
        self.profile = profile
        self._logger = logger or module_logger
        # Optional callbacks: on_delta(handle, text) fires per message.delta
        # chunk; on_tip_updated(payload) fires on session.tip.updated (the
        # runner uses it to keep gateway routing in sync with serve's tip).
        # on_clarify_request(handle, payload) fires when serve raises an
        # interactive clarify prompt inside a delegated turn — the runner
        # relays it to the origin channel and answers via respond_clarify
        # (without this the serve-side clarify blocks until timeout with the
        # question invisible to everyone). on_clarify_expire(payload) fires
        # when the prompt resolved elsewhere or timed out.
        self.on_delta = on_delta
        self.on_tip_updated = on_tip_updated
        self.on_clarify_request = on_clarify_request
        self.on_clarify_expire = on_clarify_expire

        self._session: Any = None
        self._ws: Any = None
        self._reader_task: Optional[asyncio.Task] = None
        self._connect_lock: Optional[asyncio.Lock] = None
        self._closed = False
        # Pending JSON-RPC request futures, keyed by request id.
        self._pending: Dict[str, asyncio.Future] = {}
        # In-flight turn handles, keyed by turn_id.
        self._turns: Dict[str, RuntimeTurnHandle] = {}

    # -- connection lifecycle -------------------------------------------

    @property
    def connected(self) -> bool:
        return self._ws is not None and not getattr(self._ws, "closed", False)

    async def ensure_connected(self) -> None:
        """Lazily connect and start the reader task (idempotent)."""
        if self._closed:
            raise RuntimeConnectionError("runtime client is closed")
        if self.connected:
            return
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            if self.connected:
                return
            try:
                import aiohttp
            except ImportError as exc:  # pragma: no cover — aiohttp is a gateway dep
                raise RuntimeConnectionError(
                    f"runtime delegation requires aiohttp: {exc}"
                ) from exc
            sep = "&" if "?" in self._url else "?"
            try:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(
                    f"{self._url}{sep}token={self._token}",
                    heartbeat=30.0,
                )
            except Exception as exc:
                raise RuntimeConnectionError(
                    f"failed to connect to serve runtime at {self._url}: {exc}"
                ) from exc
            self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        import aiohttp

        ws = self._ws
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        frame = json.loads(msg.data)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(frame, dict):
                        try:
                            self.handle_frame(frame)
                        except Exception:
                            self._logger.exception(
                                "runtime client: frame handling failed"
                            )
                elif msg.type in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.warning("runtime client: reader loop error: %s", exc)
        finally:
            self.connection_lost()

    def connection_lost(self, exc: Optional[BaseException] = None) -> None:
        """Handle a dropped connection.

        Pending request futures fail with a retryable
        :class:`RuntimeConnectionError`; in-flight turn handles resolve with
        state ``"unknown"`` — a turn whose events were lost has an unknown
        outcome and is never auto-replayed (design §12.3).
        """
        pending, self._pending = self._pending, {}
        turns, self._turns = self._turns, {}
        self._ws = None
        error = RuntimeConnectionError(
            str(exc) if exc else "runtime connection lost"
        )
        for future in pending.values():
            if not future.done():
                future.set_exception(error)
        for handle in turns.values():
            handle.resolve("unknown", error=str(error))

    async def close(self) -> None:
        """Tear down the WS, reader task, and HTTP session."""
        self._closed = True
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        task = self._reader_task
        self._reader_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        # Idempotent: the reader task's finally usually already ran this.
        self.connection_lost()

    # -- JSON-RPC --------------------------------------------------------

    async def request(
        self,
        method: str,
        params: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Any:
        """Send one JSON-RPC request over the WS and await its response."""
        await self.ensure_connected()
        request_id = uuid.uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        frame = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            await self._ws.send_json(frame)
        except Exception as exc:
            self._pending.pop(request_id, None)
            raise RuntimeConnectionError(f"failed to send {method}: {exc}") from exc
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise RuntimeConnectionError(
                f"{method} timed out after {timeout:.0f}s"
            ) from None

    async def respond_clarify(self, request_id: str, answer: str) -> Any:
        """Forward a channel user's clarify answer to serve (request_id-keyed).

        First responder wins on the serve side (`clarify.respond` resolves the
        pending prompt once); a stale answer gets a 4009 error — callers treat
        that as "someone else already answered", not a failure.
        """
        return await self.request(
            "clarify.respond", {"request_id": request_id, "answer": answer}
        )

    async def submit_turn(
        self,
        params: Dict[str, Any],
        timeout: float = 60.0,
    ) -> RuntimeTurnHandle:
        """Call ``turn.submit`` and register a :class:`RuntimeTurnHandle`.

        On ``{duplicate: true}`` for a turn this client is already tracking,
        the existing handle is returned (re-attach) so both callers await
        the same outcome; otherwise a fresh handle is registered — serve
        replays nothing, the events simply attribute to that turn_id.
        """
        submit_params = dict(params)
        if not submit_params.get("profile") and self.profile:
            submit_params["profile"] = self.profile
        result = await self.request("turn.submit", submit_params, timeout=timeout)
        if not isinstance(result, dict):
            raise RuntimeClientError(
                f"turn.submit returned unexpected result: {result!r}"
            )
        turn_id = str(result.get("turn_id") or "")
        if not turn_id:
            raise RuntimeClientError("turn.submit result is missing turn_id")

        existing = self._turns.get(turn_id)
        if result.get("duplicate") and existing is not None:
            return existing

        handle = RuntimeTurnHandle(
            turn_id=turn_id,
            sequence=int(result.get("sequence") or 0),
            stored_session_id=str(result.get("stored_session_id") or ""),
            lineage_id=str(result.get("lineage_id") or ""),
            runtime_session_id=str(result.get("runtime_session_id") or ""),
            # RUNTIME_GENERATION is an opaque uuid hex string on the serve
            # side — never coerce to int (a real response would raise).
            runtime_generation=str(result.get("runtime_generation") or ""),
            queue_position=int(result.get("queue_position") or 0),
            duplicate=bool(result.get("duplicate")),
            status=str(result.get("status") or ""),
            future=asyncio.get_running_loop().create_future(),
        )
        self._turns[turn_id] = handle
        return handle

    # -- frame dispatch (pure) --------------------------------------------

    def handle_frame(self, frame: Dict[str, Any]) -> None:
        """Dispatch one inbound frame (response or event).

        Pure with respect to the transport: tests feed frames here directly.
        """
        frame_id = frame.get("id")
        if frame_id is not None and ("result" in frame or "error" in frame):
            self._handle_response(str(frame_id), frame)
            return
        if frame.get("method") != "event":
            return
        params = frame.get("params")
        if not isinstance(params, dict):
            return
        self._handle_event(params)

    def _handle_response(self, frame_id: str, frame: Dict[str, Any]) -> None:
        future = self._pending.pop(frame_id, None)
        if future is None or future.done():
            return
        error = frame.get("error")
        if error is not None:
            if isinstance(error, dict):
                code = error.get("code")
                message = str(error.get("message") or "unknown error")
            else:
                code, message = None, str(error)
            future.set_exception(
                RuntimeRequestError(code if isinstance(code, int) else -1, message)
            )
        else:
            future.set_result(frame.get("result"))

    def _handle_event(self, params: Dict[str, Any]) -> None:
        event_type = str(params.get("type") or "")
        payload = params.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        turn_info = params.get("turn")
        if not isinstance(turn_info, dict):
            turn_info = {}

        if event_type == "session.tip.updated":
            if self.on_tip_updated is not None:
                try:
                    self.on_tip_updated(payload)
                except Exception:
                    self._logger.exception(
                        "runtime client: on_tip_updated callback failed"
                    )
            return

        if event_type == "clarify.expire":
            # request_id-keyed; delivered even when no local turn handle is
            # tracked anymore (the prompt may outlive the turn bookkeeping).
            if self.on_clarify_expire is not None:
                try:
                    self.on_clarify_expire(payload)
                except Exception:
                    self._logger.exception(
                        "runtime client: on_clarify_expire callback failed"
                    )
            return

        # Mid-turn events attribute via the `turn` sibling key; terminal
        # turn.* events also carry turn_id in their payload.
        turn_id = str(turn_info.get("turn_id") or payload.get("turn_id") or "")
        if not turn_id:
            return
        handle = self._turns.get(turn_id)
        if handle is None:
            return

        if event_type == "clarify.request":
            if self.on_clarify_request is not None:
                try:
                    self.on_clarify_request(handle, payload)
                except Exception:
                    self._logger.exception(
                        "runtime client: on_clarify_request callback failed"
                    )
            return

        if event_type == "message.delta":
            text = str(payload.get("text") or payload.get("delta") or "")
            if text:
                handle.deltas.append(text)
                if self.on_delta is not None:
                    try:
                        self.on_delta(handle, text)
                    except Exception:
                        self._logger.exception(
                            "runtime client: on_delta callback failed"
                        )
            return

        if event_type == "message.complete":
            text = payload.get("text")
            handle.final_text = str(text) if text else "".join(handle.deltas)
            return

        state = _TURN_TERMINAL_STATES.get(event_type)
        if state is not None:
            self._turns.pop(turn_id, None)
            handle.resolve(
                state,
                already_delivered=bool(payload.get("already_delivered")),
                error=str(payload.get("error") or ""),
            )
