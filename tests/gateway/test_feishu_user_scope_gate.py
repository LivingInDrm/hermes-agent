"""Feishu inbound fail-explicit gate under session_scope=user.

The single inbound funnel (`_dispatch_inbound_event`) must drop messages
whose source carries no participant identity instead of letting them fall
into a shared per-chat bucket (channel-per-user-session design §3.2). With
chat scope (the default) the same messages pass through untouched.
"""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_adapter():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    return FeishuAdapter(PlatformConfig())


def _event(user_id=None, user_id_alt=None):
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id=user_id,
        user_id_alt=user_id_alt,
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.COMMAND,  # bypasses text batching → guard path
        source=source,
        raw_message=None,
        message_id="om_1",
    )


class TestUserScopeInboundGate(unittest.TestCase):
    def _dispatch(self, event):
        adapter = _make_adapter()
        with patch.object(
            adapter, "_handle_message_with_guards", new=AsyncMock()
        ) as guards:
            asyncio.run(adapter._dispatch_inbound_event(event))
        return guards

    @patch.dict(os.environ, {"HERMES_SESSION_SCOPE": json.dumps({"feishu": "user"})})
    def test_missing_participant_identity_is_dropped(self):
        guards = self._dispatch(_event(user_id=None, user_id_alt=None))
        guards.assert_not_awaited()

    @patch.dict(os.environ, {"HERMES_SESSION_SCOPE": json.dumps({"feishu": "user"})})
    def test_identified_participant_passes(self):
        guards = self._dispatch(_event(user_id="ou_alice"))
        guards.assert_awaited_once()

    def test_chat_scope_default_passes_unidentified(self):
        os.environ.pop("HERMES_SESSION_SCOPE", None)
        guards = self._dispatch(_event(user_id=None, user_id_alt=None))
        guards.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
