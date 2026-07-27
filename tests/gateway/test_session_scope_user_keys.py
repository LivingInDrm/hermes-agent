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
