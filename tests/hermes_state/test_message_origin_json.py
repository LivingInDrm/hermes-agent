"""messages.origin_json chat provenance (channel-per-user-session design §5.3).

Under merged per-user sessions one session receives input from many chat
windows, so "where was this said" is stamped on the turn's user row:
``append_message(origin_json=...)`` persists it, ``get_messages`` (the REST
``/api/sessions/{id}/messages`` projection — SELECT *) exposes it, and the
conversation replay + rewrite paths (``get_messages_as_conversation`` →
``replace_messages``) carry it through history rewrites.
"""
import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


ORIGIN = json.dumps(
    {"chat_type": "group", "chat_id": "oc_g", "chat_name": "产品群", "thread_id": ""},
    ensure_ascii=False,
)


def test_append_and_rest_projection_roundtrip(db):
    db.create_session("s1", source="gateway")
    db.append_message("s1", role="user", content="干活", origin_json=ORIGIN)
    db.append_message("s1", role="assistant", content="收到")

    rows = db.get_messages("s1")
    assert rows[0]["origin_json"] == ORIGIN
    # Assistant rows are never stamped — display projects the user row's
    # origin forward across its turn.
    assert rows[1]["origin_json"] is None


def test_default_is_null_for_desktop_rows(db):
    db.create_session("s1", source="desktop")
    db.append_message("s1", role="user", content="hi")
    assert db.get_messages("s1")[0]["origin_json"] is None


def test_survives_replace_messages_rewrite(db):
    """Truncate/undo/compaction rewrite via replace_messages keeps provenance."""
    db.create_session("s1", source="gateway")
    db.append_message("s1", role="user", content="干活", origin_json=ORIGIN)
    db.append_message("s1", role="assistant", content="收到")

    conversation = db.get_messages_as_conversation("s1")
    assert conversation[0]["origin_json"] == ORIGIN

    db.replace_messages("s1", conversation)
    rows = db.get_messages("s1")
    assert rows[0]["origin_json"] == ORIGIN
    assert rows[1]["origin_json"] is None


def test_legacy_db_gains_column_on_reopen(tmp_path):
    """Declarative column reconciliation adds origin_json to existing DBs."""
    import sqlite3

    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s1", source="cli")
    db.close()

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE messages DROP COLUMN origin_json")
    conn.commit()
    conn.close()

    reopened = SessionDB(path)
    reopened.create_session("s2", source="gateway")
    reopened.append_message("s2", role="user", content="x", origin_json=ORIGIN)
    assert reopened.get_messages("s2")[0]["origin_json"] == ORIGIN
