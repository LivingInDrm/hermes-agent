"""Client display metadata accepted by ``prompt.submit``."""

import inspect

import pytest

from hermes_cli.myagents_prompt_metadata import validate_display_metadata


class TestValidation:
    def test_absent_metadata_is_none(self):
        assert validate_display_metadata(None) is None

    def test_accepts_a_flat_object(self):
        value = {"platform": "feishu", "chat_type": "group", "chat_name": "产品群"}
        assert validate_display_metadata(value) == value

    def test_accepts_bounded_nesting(self):
        value = {"origin": {"chat": {"name": "产品群"}}}
        assert validate_display_metadata(value) == value

    @pytest.mark.parametrize("bad", ["a string", 42, ["a", "list"], True])
    def test_rejects_non_objects(self, bad):
        with pytest.raises(ValueError, match="must be a JSON object"):
            validate_display_metadata(bad)

    def test_rejects_too_many_keys(self):
        with pytest.raises(ValueError, match="at most"):
            validate_display_metadata({f"k{i}": i for i in range(17)})

    def test_rejects_excessive_nesting(self):
        with pytest.raises(ValueError, match="nests deeper"):
            validate_display_metadata({"a": {"b": {"c": {"d": "too deep"}}}})

    def test_rejects_oversized_payloads(self):
        with pytest.raises(ValueError, match="exceeds"):
            validate_display_metadata({"blob": "x" * 4096})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
    def test_rejects_non_json_scalars(self, bad):
        with pytest.raises(ValueError, match="JSON"):
            validate_display_metadata({"value": bad})

    def test_rejects_non_string_keys(self):
        with pytest.raises(ValueError, match="keys must be strings"):
            validate_display_metadata({"outer": {1: "x"}})


class TestNoKindForgery:
    def test_display_kind_inside_metadata_is_inert_data(self):
        value = {"display_kind": "hidden"}
        assert validate_display_metadata(value) == value

    def test_prompt_submit_never_reads_a_client_kind(self):
        from tui_gateway import methods_prompt

        source = inspect.getsource(methods_prompt)
        handler_start = source.index('@method("prompt.submit")')
        handler = source[handler_start : handler_start + 14_000]
        assert 'params.get("display_metadata")' in handler
        assert 'params.get("display_kind")' not in handler
        assert '"annotated" if client_display_metadata is not None' in handler
