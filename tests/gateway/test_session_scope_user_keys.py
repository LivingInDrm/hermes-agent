"""Per-user session scope key derivation (channel-per-user-session design).

``session_scope: user`` collapses dm/group/thread into one key per
``(platform, participant)``: ``agent:<ns>:<platform>:user:<participant_id>``.
Scope resolves from the startup-bridged ``HERMES_SESSION_SCOPE`` env mapping
so every ``build_session_key`` call site in the process agrees; sources with
no participant identity fail explicitly instead of falling back to a shared
per-chat bucket.
"""

import json

import pytest

from gateway.config import Platform
from gateway.session import (
    MissingParticipantIdentity,
    SESSION_SCOPE_CHAT,
    SESSION_SCOPE_USER,
    SessionSource,
    build_session_key,
    is_shared_multi_user_session,
    session_scope_for,
)


def _source(
    chat_type="dm",
    chat_id="oc_chat_a",
    user_id="ou_open_id",
    user_id_alt=None,
    thread_id=None,
    platform=Platform.FEISHU,
):
    return SessionSource(
        platform=platform,
        chat_type=chat_type,
        chat_id=chat_id,
        user_id=user_id,
        user_id_alt=user_id_alt,
        thread_id=thread_id,
    )


class TestUserScopeKeyDerivation:
    def test_dm_group_thread_converge_to_one_key(self):
        """DM / group / group-topic from the same participant share one key."""
        dm = _source(chat_type="dm", chat_id="oc_dm")
        group_a = _source(chat_type="group", chat_id="oc_group_a")
        group_b = _source(chat_type="group", chat_id="oc_group_b")
        topic = _source(
            chat_type="group", chat_id="oc_group_b", thread_id="omt_topic"
        )

        keys = {
            build_session_key(s, session_scope=SESSION_SCOPE_USER)
            for s in (dm, group_a, group_b, topic)
        }
        assert keys == {"agent:main:feishu:user:ou_open_id"}

    def test_different_participants_stay_isolated(self):
        alice = _source(user_id="ou_alice")
        bob = _source(user_id="ou_bob")
        assert build_session_key(
            alice, session_scope=SESSION_SCOPE_USER
        ) != build_session_key(bob, session_scope=SESSION_SCOPE_USER)

    def test_user_id_alt_preferred_over_user_id(self):
        """Feishu union_id (user_id_alt) wins so keys survive app changes."""
        source = _source(user_id="ou_open_id", user_id_alt="on_union_id")
        key = build_session_key(source, session_scope=SESSION_SCOPE_USER)
        assert key == "agent:main:feishu:user:on_union_id"

    def test_named_profile_namespace_preserved(self):
        source = _source()
        key = build_session_key(
            source, profile="coder", session_scope=SESSION_SCOPE_USER
        )
        assert key == "agent:coder:feishu:user:ou_open_id"

    def test_positional_layout_platform_slot_holds(self):
        """parts[2] == platform for both scopes (positional parsers)."""
        key = build_session_key(_source(), session_scope=SESSION_SCOPE_USER)
        assert key.split(":")[2] == "feishu"

    def test_missing_participant_fails_explicitly(self):
        """No user_id/user_id_alt → raise, never a shared per-chat bucket."""
        anonymous = _source(user_id=None, user_id_alt=None)
        with pytest.raises(MissingParticipantIdentity):
            build_session_key(anonymous, session_scope=SESSION_SCOPE_USER)

    def test_chat_scope_unchanged(self):
        """Explicit chat scope keeps the legacy per-chat key shape."""
        source = _source(chat_type="group", chat_id="oc_group_a")
        key = build_session_key(source, session_scope=SESSION_SCOPE_CHAT)
        assert key == "agent:main:feishu:group:oc_group_a:ou_open_id"


class TestScopeResolutionFromEnv:
    def test_default_is_chat(self, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_SCOPE", raising=False)
        assert session_scope_for("feishu") == SESSION_SCOPE_CHAT

    def test_env_mapping_selects_user_scope_per_platform(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_SCOPE", json.dumps({"feishu": "user"}))
        assert session_scope_for("feishu") == SESSION_SCOPE_USER
        assert session_scope_for("telegram") == SESSION_SCOPE_CHAT

    def test_malformed_env_falls_back_to_chat(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_SCOPE", "not json")
        assert session_scope_for("feishu") == SESSION_SCOPE_CHAT
        monkeypatch.setenv("HERMES_SESSION_SCOPE", json.dumps({"feishu": "bogus"}))
        assert session_scope_for("feishu") == SESSION_SCOPE_CHAT

    def test_build_session_key_resolves_scope_from_env(self, monkeypatch):
        """Call sites that pass no explicit scope pick up the platform config."""
        monkeypatch.setenv("HERMES_SESSION_SCOPE", json.dumps({"feishu": "user"}))
        dm = _source(chat_type="dm", chat_id="oc_dm")
        group = _source(chat_type="group", chat_id="oc_group")
        assert build_session_key(dm) == build_session_key(group)
        # Other platforms on the same process keep chat-scope keys.
        tg = _source(platform=Platform.TELEGRAM, chat_type="dm", chat_id="42")
        assert build_session_key(tg) == "agent:main:telegram:dm:42"


class TestSharedMultiUserUnderUserScope:
    def test_group_never_shared_under_user_scope(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_SCOPE", json.dumps({"feishu": "user"}))
        group = _source(chat_type="group", chat_id="oc_group")
        assert not is_shared_multi_user_session(
            group, group_sessions_per_user=False
        )

    def test_chat_scope_sharing_rules_unchanged(self, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_SCOPE", raising=False)
        group = _source(chat_type="group", chat_id="oc_group")
        assert is_shared_multi_user_session(group, group_sessions_per_user=False)
        assert not is_shared_multi_user_session(group, group_sessions_per_user=True)


class TestDelegatedRouteSkipsGatewayDbRow:
    """Delegated routes: serve owns persistence — no gateway sqlite row
    (regression: orphaned empty duplicate surfaced as a second desktop task)."""

    def _store(self, tmp_path, delegated):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore

        class _Db:
            def __init__(self):
                self.created = []

            def create_session(self, **kwargs):
                self.created.append(kwargs)

            def record_gateway_session_peer(self, *args, **kwargs):
                pass

        store = SessionStore(tmp_path, GatewayConfig())
        store._db = _Db()
        store.runtime_delegated_probe = lambda source: delegated
        return store

    def test_delegated_source_creates_entry_but_no_db_row(self, tmp_path):
        store = self._store(tmp_path, delegated=True)
        entry = store.get_or_create_session(_source())
        assert entry.session_key
        assert store._db.created == []

    def test_non_delegated_source_still_creates_db_row(self, tmp_path):
        store = self._store(tmp_path, delegated=False)
        store.get_or_create_session(_source())
        assert len(store._db.created) == 1


class TestUserScopeDisplayName:
    """User-scope peer stamp: display identity is the person; never fall back
    to chat_name (Feishu DM chat_name degrades to the raw oc_ chat_id)."""

    def _capture(self, tmp_path):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore

        class _Db:
            def __init__(self):
                self.stamps = []

            def record_gateway_session_peer(self, *args, **kwargs):
                self.stamps.append(kwargs)

        store = SessionStore(tmp_path, GatewayConfig())
        store._db = _Db()
        return store

    def test_user_scope_prefers_user_name_and_never_chat_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_SCOPE", json.dumps({"feishu": "user"}))
        store = self._capture(tmp_path)
        named = SessionSource(
            platform=Platform.FEISHU, chat_id="oc_x", chat_name="oc_x",
            chat_type="dm", user_id="ou_a", user_name="刘晓春",
        )
        store._record_gateway_session_peer("sid-1", "key-1", named)
        assert store._db.stamps[-1]["display_name"] == "刘晓春"

        anonymous = SessionSource(
            platform=Platform.FEISHU, chat_id="oc_x", chat_name="oc_x",
            chat_type="dm", user_id="ou_a",
        )
        store._record_gateway_session_peer("sid-2", "key-2", anonymous)
        assert store._db.stamps[-1]["display_name"] is None

    def test_chat_scope_keeps_chat_name_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_SCOPE", raising=False)
        store = self._capture(tmp_path)
        source = SessionSource(
            platform=Platform.FEISHU, chat_id="oc_x", chat_name="产品群",
            chat_type="group", user_id="ou_a",
        )
        store._record_gateway_session_peer("sid-3", "key-3", source)
        assert store._db.stamps[-1]["display_name"] == "产品群"
