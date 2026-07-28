"""Client-supplied display metadata on prompt.submit.

The point of the design is what clients *cannot* send: there is no
``display_kind`` parameter, so no client can classify its own message as
``hidden`` and drop it out of every transcript projection.
"""

import pytest

from tui_gateway.server import _validated_display_metadata


class TestValidation:
    def test_absent_metadata_is_none(self):
        # Default behaviour must be completely unchanged when the param is absent.
        assert _validated_display_metadata(None) is None

    def test_accepts_a_flat_object(self):
        value = {"platform": "feishu", "chat_type": "group", "chat_name": "产品群"}
        assert _validated_display_metadata(value) == value

    def test_accepts_bounded_nesting(self):
        value = {"origin": {"chat": {"name": "产品群"}}}
        assert _validated_display_metadata(value) == value

    @pytest.mark.parametrize("bad", ["a string", 42, ["a", "list"], True])
    def test_rejects_non_objects(self, bad):
        with pytest.raises(ValueError, match="must be a JSON object"):
            _validated_display_metadata(bad)

    def test_rejects_too_many_keys(self):
        with pytest.raises(ValueError, match="at most"):
            _validated_display_metadata({f"k{i}": i for i in range(17)})

    def test_rejects_excessive_nesting(self):
        with pytest.raises(ValueError, match="nests deeper"):
            _validated_display_metadata({"a": {"b": {"c": {"d": "too deep"}}}})

    def test_rejects_oversized_payloads(self):
        with pytest.raises(ValueError, match="exceeds"):
            _validated_display_metadata({"blob": "x" * 4096})

    def test_rejects_non_scalar_values(self):
        with pytest.raises(ValueError, match="JSON scalars"):
            _validated_display_metadata({"when": object()})

    def test_rejects_non_string_keys(self):
        with pytest.raises(ValueError, match="keys must be strings"):
            _validated_display_metadata({"outer": {1: "x"}})


class TestNoKindForgery:
    def test_metadata_carrying_a_display_kind_key_is_just_data(self):
        # A client may put the *string* "display_kind" in its metadata; that is
        # inert data on the message, not the row's classification. Nothing in
        # the RPC lets a client set the actual display_kind field.
        value = {"display_kind": "hidden"}
        assert _validated_display_metadata(value) == value

    def test_prompt_submit_has_no_display_kind_parameter(self):
        import inspect

        from tui_gateway import server

        source = inspect.getsource(server)
        handler_start = source.index('@method("prompt.submit")')
        handler = source[handler_start:handler_start + 4000]
        assert 'params.get("display_metadata")' in handler
        # The handler must never read a client-provided kind.
        assert 'params.get("display_kind")' not in handler
