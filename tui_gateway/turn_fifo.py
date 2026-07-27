"""Per-session turn FIFO and turn identity for the shared session runtime.

Design source: MyAgents docs/design/channel-desktop-shared-session-runtime.md
(§4 identity, §7 runtime protocol, §8 FIFO / generation-local dedup).

Everything in this module is **process-local**: queues, sequences, dedup
tables and sinks die with the serve process (``durability: "process"``).
Turns that never started are reported ``lost`` by clients observing a
``runtime_generation`` change; the active turn at crash time becomes
``unknown`` and is never replayed automatically.

This module is intentionally free of imports from ``tui_gateway.server`` so
its data structures stay unit-testable in isolation; the server wires the
dispatch/emit callbacks.
"""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# One value per serve process. Everything queue/dedup/sink-related is scoped
# to this generation; clients must discard state from older generations.
RUNTIME_GENERATION: str = uuid.uuid4().hex

# Local V1 keeps the Gateway's existing pending cap; deliberately NOT
# configurable (design §8.1).
TURN_QUEUE_MAX_PENDING = 32

TURN_STATES = frozenset({
    "queued",
    "running",
    "awaiting-approval",
    "awaiting-input",
    "completed",
    "interrupted",
    "failed",
    "unknown",
    "lost",
})

_TERMINAL_TURN_STATES = frozenset({
    "completed", "interrupted", "failed", "unknown", "lost",
})


def is_terminal_turn_state(state: str) -> bool:
    return state in _TERMINAL_TURN_STATES


class TurnQueueFullError(Exception):
    """Raised when a session's pending FIFO is at capacity.

    Explicit, retryable rejection — the queue never silently merges,
    overwrites or drops an input (design §8.1).
    """

    def __init__(self, depth: int) -> None:
        super().__init__(
            f"session turn queue is full ({depth}/{TURN_QUEUE_MAX_PENDING} pending); retry later"
        )
        self.depth = depth
        self.retryable = True


@dataclass
class TurnRecord:
    """One independent user input and its execution lifecycle."""

    turn_id: str
    client_turn_id: str
    sequence: int
    origin: str  # "desktop" | "channel"
    delivery_mode: str  # "desktop-only" | "origin-channel"
    text: Any
    profile: str
    lineage_id: str
    state: str = "queued"
    enqueued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Channel execution context (trusted, gateway-derived; never from renderer)
    trusted_source: Optional[dict] = None
    ephemeral_prompt: Optional[str] = None
    auto_skills: Optional[list] = None
    execution_hints: Optional[dict] = None
    delivery_sink_id: Optional[str] = None
    attachments: Optional[list] = None
    # Legacy single-transport compatibility: the transport that submitted the
    # turn streams the drained turn's events until multi-subscriber routing
    # (M3) removes per-turn transport pinning.
    transport: Any = None

    def envelope(self, *, stored_session_id: str, runtime_session_id: str) -> dict:
        """Event envelope fields required on every turn event (design §7.5)."""
        return {
            "profile": self.profile,
            "lineage_id": self.lineage_id,
            "stored_session_id": stored_session_id,
            "runtime_session_id": runtime_session_id,
            "runtime_generation": RUNTIME_GENERATION,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "origin": self.origin,
            "delivery_mode": self.delivery_mode,
        }


class SessionTurnState:
    """Actor-side turn bookkeeping attached to one live runtime session.

    The live session record (``_sessions[sid]``) *is* the Runtime Actor's
    embodiment: Hermes already guarantees at most one live session per
    canonical lineage (resume dedup under ``_session_resume_lock`` +
    ``resolve_resume_session_id`` tip canonicalization). This object adds the
    independent turn identities, the bounded FIFO and the accept-order
    sequence the shared runtime requires. All mutation happens under the
    owning session's ``history_lock``.
    """

    def __init__(self) -> None:
        self.active: Optional[TurnRecord] = None
        self.queue: deque[TurnRecord] = deque()
        self._sequence = itertools.count(1)

    # -- accept order -----------------------------------------------------

    def next_sequence(self) -> int:
        return next(self._sequence)

    # -- queue ------------------------------------------------------------

    def enqueue(self, record: TurnRecord) -> int:
        """Append a queued turn; returns its 1-based queue position hint."""
        if len(self.queue) >= TURN_QUEUE_MAX_PENDING:
            raise TurnQueueFullError(len(self.queue))
        record.state = "queued"
        self.queue.append(record)
        return len(self.queue)

    def promote_next(self) -> Optional[TurnRecord]:
        """Pop the next queued turn and make it the active one."""
        if self.active is not None and not is_terminal_turn_state(self.active.state):
            return None
        if not self.queue:
            return None
        record = self.queue.popleft()
        record.state = "running"
        record.started_at = time.time()
        self.active = record
        return record

    def activate(self, record: TurnRecord) -> None:
        """Directly mark a record active (idle-session fast path)."""
        record.state = "running"
        record.started_at = time.time()
        self.active = record

    def finish_active(self, state: str) -> Optional[TurnRecord]:
        record = self.active
        if record is None:
            return None
        record.state = state
        record.finished_at = time.time()
        return record

    def drop_queued(self, state: str = "interrupted") -> List[TurnRecord]:
        """Clear all pending turns (legacy session.interrupt semantics)."""
        dropped = list(self.queue)
        self.queue.clear()
        for record in dropped:
            record.state = state
            record.finished_at = time.time()
        return dropped

    def queue_depth(self) -> int:
        return len(self.queue)

    def queued_snapshot(self) -> List[dict]:
        return [
            {
                "turn_id": record.turn_id,
                "sequence": record.sequence,
                "origin": record.origin,
                "delivery_mode": record.delivery_mode,
                "user": _display_text(record.text),
                "enqueued_at": record.enqueued_at,
            }
            for record in self.queue
        ]

    def active_snapshot(self) -> Optional[dict]:
        record = self.active
        if record is None or is_terminal_turn_state(record.state):
            return None
        return {
            "turn_id": record.turn_id,
            "sequence": record.sequence,
            "origin": record.origin,
            "delivery_mode": record.delivery_mode,
            "state": record.state,
            "started_at": record.started_at,
        }


def _display_text(text: Any) -> str:
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts: List[str] = []
        for item in text:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return "" if text is None else str(text)


class GenerationDedupTable:
    """Generation-local ``clientTurnId`` dedup (design §8.3).

    Key: (profile, conversation identity, client authority, client_turn_id).
    The conversation identity is the canonical lineage id for existing
    sessions, or the trusted ``source + source_session_key`` before a
    channel-create resolves. The table lives and dies with the process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[tuple, TurnRecord] = {}

    @staticmethod
    def key(profile: str, conversation_identity: str, authority: str, client_turn_id: str) -> tuple:
        return (profile, conversation_identity, authority, client_turn_id)

    def claim(self, key: tuple, record: TurnRecord) -> Optional[TurnRecord]:
        """Register ``record`` for ``key``; returns the existing record on duplicate."""
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                return existing
            self._entries[key] = record
            return None

    def release(self, key: tuple) -> None:
        """Forget a claim whose submit failed validation after registration."""
        with self._lock:
            self._entries.pop(key, None)


def make_turn_id() -> str:
    return f"turn-{uuid.uuid4().hex[:20]}"


def resolve_compression_lineage_root(db: Any, session_id: str) -> str:
    """Return the **verified** compression-lineage root for ``session_id``.

    Unlike ``SessionDB.get_conversation_root`` (which follows every
    ``parent_session_id`` edge, including branch/delegate children), this walk
    only crosses edges that are genuine compression continuations:

    - the parent ended with ``end_reason='compression'``;
    - the current row is not an explicit branch (``_branched_from``) or
      delegate (``_delegate_from``) child.

    A branch/delegate child is its own lineage root — collapsing it into the
    source conversation would merge two Actors (design §5.1).
    """
    if not session_id:
        return session_id
    current = session_id
    seen = {current}
    for _ in range(100):
        row = None
        try:
            row = db.get_session(current)
        except Exception:
            return current
        if not row:
            return current
        model_config = row.get("model_config")
        if isinstance(model_config, str) and model_config.strip():
            try:
                import json

                model_config = json.loads(model_config)
            except Exception:
                model_config = {}
        if isinstance(model_config, dict) and (
            model_config.get("_branched_from") or model_config.get("_delegate_from")
        ):
            return current
        parent_id = row.get("parent_session_id")
        if not parent_id or parent_id in seen:
            return current
        try:
            parent = db.get_session(parent_id)
        except Exception:
            return current
        if not parent or parent.get("end_reason") != "compression":
            return current
        current = parent_id
        seen.add(current)
    return current
