"""User-scoped session keys: one conversation per person, wherever they say it.

The default remains per-chat and must stay byte-identical; ``session_scope``
is opt-in per platform.
"""

from types import SimpleNamespace

import pytest

from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import Platform


def _source(chat_type, chat_id=None, thread_id=None, user_id=None, user_id_alt=None):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_type=chat_type,
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=user_id,
        user_id_alt=user_id_alt,
    )


class TestUserScope:
    def test_same_user_across_dm_group_and_thread_shares_one_key(self):
        dm = _source("dm", chat_id="oc_dm", user_id="ou_a", user_id_alt="on_a")
        group = _source("group", chat_id="oc_group", user_id="ou_a", user_id_alt="on_a")
        thread = _source(
            "group", chat_id="oc_group", thread_id="omt_1", user_id="ou_a", user_id_alt="on_a"
        )

        keys = {build_session_key(s, session_scope="user") for s in (dm, group, thread)}
        assert len(keys) == 1, keys
        # The alternate (developer-scoped) identity wins — it is the stable one.
        assert keys.pop().endswith(":user:on_a")

    def test_different_users_never_share_a_key(self):
        a = _source("group", chat_id="oc_group", user_id="ou_a", user_id_alt="on_a")
        b = _source("group", chat_id="oc_group", user_id="ou_b", user_id_alt="on_b")

        assert build_session_key(a, session_scope="user") != build_session_key(
            b, session_scope="user"
        )

    def test_falls_back_to_user_id_when_alt_is_absent(self):
        source = _source("dm", chat_id="oc_dm", user_id="ou_a")
        assert build_session_key(source, session_scope="user").endswith(":user:ou_a")

    def test_missing_identity_fails_closed(self):
        # Degrading to a chat key here would merge *different* people into one
        # session — a cross-user history leak. Refuse instead.
        source = _source("group", chat_id="oc_group")
        with pytest.raises(ValueError, match="stable user identity"):
            build_session_key(source, session_scope="user")

    def test_default_scope_is_unchanged(self):
        source = _source("group", chat_id="oc_group", user_id="ou_a", user_id_alt="on_a")
        baseline = build_session_key(source)
        assert build_session_key(source, session_scope="chat") == baseline
        assert build_session_key(source, session_scope="user") != baseline
        # The per-chat key still carries the chat, exactly as before.
        assert "oc_group" in baseline

    def test_unknown_scope_value_does_not_widen_the_session(self):
        source = _source("group", chat_id="oc_group", user_id="ou_a", user_id_alt="on_a")
        # Only the literal "user" opts in; anything else keeps per-chat rules.
        assert build_session_key(source, session_scope="nonsense") == build_session_key(source)


class TestScopeResolution:
    """Where the setting is read from — the part that silently no-ops if wrong."""

    def _store(self, platforms):
        from gateway.session import SessionStore

        store = SessionStore.__new__(SessionStore)
        store.config = SimpleNamespace(platforms=platforms)
        return store

    def test_reads_config_keyed_by_platform_enum(self):
        # config.platforms is keyed by the Platform enum. Looking it up with
        # .value silently misses every time: the setting shows up in
        # config.yaml and does nothing, and every chat gets its own session.
        from gateway.config import PlatformConfig

        cfg = PlatformConfig(enabled=True)
        cfg.extra["session_scope"] = "user"
        store = self._store({Platform.FEISHU: cfg})
        assert store._session_scope_for(_source("dm", chat_id="oc_x", user_id="ou_a")) == "user"

    def test_defaults_to_chat_when_unset(self):
        from gateway.config import PlatformConfig

        store = self._store({Platform.FEISHU: PlatformConfig(enabled=True)})
        assert store._session_scope_for(_source("dm", chat_id="oc_x")) == "chat"

    def test_unknown_platform_defaults_to_chat(self):
        store = self._store({})
        assert store._session_scope_for(_source("dm", chat_id="oc_x")) == "chat"
