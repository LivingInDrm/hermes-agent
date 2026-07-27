"""turn_fifo pure data structures: lineage canonicalization + FIFO invariants."""

import json
import types

from tui_gateway.turn_fifo import (
    TURN_QUEUE_MAX_PENDING,
    GenerationDedupTable,
    SessionTurnState,
    TurnQueueFullError,
    TurnRecord,
    resolve_compression_lineage_root,
)


def _record(seq, text="x", client="", origin="desktop"):
    return TurnRecord(
        turn_id=f"turn-{seq}", client_turn_id=client, sequence=seq,
        origin=origin, delivery_mode="desktop-only", text=text,
        profile="p", lineage_id="lineage",
    )


class _FakeDB:
    """Rows: id → {parent_session_id, end_reason, model_config}."""

    def __init__(self, rows):
        self.rows = rows

    def get_session(self, session_id):
        return self.rows.get(session_id)


# ── lineage root resolution (design §5.1) ──────────────────────────────────

def test_compression_chain_resolves_to_root():
    db = _FakeDB({
        "a": {"id": "a", "parent_session_id": None, "end_reason": "compression"},
        "b": {"id": "b", "parent_session_id": "a", "end_reason": "compression"},
        "c": {"id": "c", "parent_session_id": "b", "end_reason": None},
    })
    assert resolve_compression_lineage_root(db, "c") == "a"
    assert resolve_compression_lineage_root(db, "b") == "a"
    assert resolve_compression_lineage_root(db, "a") == "a"


def test_branch_child_is_its_own_root():
    """A branch is a NEW lineage — collapsing it into its source would merge
    two Actors."""
    db = _FakeDB({
        "src": {"id": "src", "parent_session_id": None, "end_reason": "compression"},
        "branch": {
            "id": "branch", "parent_session_id": "src", "end_reason": None,
            "model_config": json.dumps({"_branched_from": "src"}),
        },
        "delegate": {
            "id": "delegate", "parent_session_id": "src", "end_reason": None,
            "model_config": json.dumps({"_delegate_from": "src"}),
        },
    })
    assert resolve_compression_lineage_root(db, "branch") == "branch"
    assert resolve_compression_lineage_root(db, "delegate") == "delegate"


def test_non_compression_parent_edge_is_not_crossed():
    db = _FakeDB({
        "parent": {"id": "parent", "parent_session_id": None, "end_reason": "ws_disconnect"},
        "child": {"id": "child", "parent_session_id": "parent", "end_reason": None},
    })
    assert resolve_compression_lineage_root(db, "child") == "child"


def test_unknown_or_cyclic_rows_fall_back_to_input():
    assert resolve_compression_lineage_root(_FakeDB({}), "ghost") == "ghost"
    cyclic = _FakeDB({
        "a": {"id": "a", "parent_session_id": "b", "end_reason": "compression"},
        "b": {"id": "b", "parent_session_id": "a", "end_reason": "compression"},
    })
    # Bounded walk; never spins.
    assert resolve_compression_lineage_root(cyclic, "a") in ("a", "b")


# ── FIFO invariants ────────────────────────────────────────────────────────

def test_fifo_orders_and_caps():
    state = SessionTurnState()
    for index in range(TURN_QUEUE_MAX_PENDING):
        state.enqueue(_record(index))
    assert state.queue_depth() == TURN_QUEUE_MAX_PENDING
    try:
        state.enqueue(_record(99))
        raise AssertionError("expected TurnQueueFullError")
    except TurnQueueFullError as exc:
        assert exc.retryable is True

    first = state.promote_next()
    assert first.sequence == 0
    assert first.state == "running"
    assert state.active is first
    # No promotion while a turn is active.
    assert state.promote_next() is None
    state.finish_active("completed")
    assert state.promote_next().sequence == 1


def test_sequences_are_monotonic():
    state = SessionTurnState()
    values = [state.next_sequence() for _ in range(5)]
    assert values == sorted(values)
    assert len(set(values)) == 5


def test_dedup_claim_and_release():
    table = GenerationDedupTable()
    key = GenerationDedupTable.key("p", "lineage", "desktop", "client-1")
    first = _record(1, client="client-1")
    assert table.claim(key, first) is None
    assert table.claim(key, _record(2, client="client-1")) is first
    table.release(key)
    replacement = _record(3, client="client-1")
    assert table.claim(key, replacement) is None


def test_envelope_carries_required_fields():
    record = _record(7)
    envelope = record.envelope(stored_session_id="tip", runtime_session_id="sid")
    assert envelope == {
        "profile": "p",
        "lineage_id": "lineage",
        "stored_session_id": "tip",
        "runtime_session_id": "sid",
        "runtime_generation": envelope["runtime_generation"],
        "turn_id": "turn-7",
        "sequence": 7,
        "origin": "desktop",
        "delivery_mode": "desktop-only",
    }
    assert envelope["runtime_generation"]
