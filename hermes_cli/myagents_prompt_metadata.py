"""Validate MyAgents client display metadata without importing Hermes internals.

Clients may annotate an ordinary user message, but they never choose its
``display_kind``.  The JSON-RPC handler selects the non-hidden ``annotated``
kind after this bounded validator accepts the metadata.
"""

from __future__ import annotations

import json
from typing import Any


MAX_KEYS = 16
MAX_DEPTH = 3
MAX_BYTES = 2048


def validate_display_metadata(raw: Any) -> dict | None:
    """Return a valid metadata object, or raise ``ValueError`` for bad input."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("display_metadata must be a JSON object")
    if len(raw) > MAX_KEYS:
        raise ValueError(f"display_metadata accepts at most {MAX_KEYS} keys")

    def check(value: Any, depth: int) -> None:
        if isinstance(value, dict):
            if depth > MAX_DEPTH:
                raise ValueError(
                    f"display_metadata nests deeper than {MAX_DEPTH} levels"
                )
            if len(value) > MAX_KEYS:
                raise ValueError(
                    f"display_metadata accepts at most {MAX_KEYS} keys"
                )
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("display_metadata keys must be strings")
                check(item, depth + 1)
            return
        if isinstance(value, list):
            if depth > MAX_DEPTH:
                raise ValueError(
                    f"display_metadata nests deeper than {MAX_DEPTH} levels"
                )
            for item in value:
                check(item, depth + 1)
            return
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError("display_metadata values must be JSON scalars")

    check(raw, 1)
    try:
        encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("display_metadata must be JSON-serialisable") from exc
    if len(encoded.encode("utf-8")) > MAX_BYTES:
        raise ValueError(f"display_metadata exceeds {MAX_BYTES} bytes")
    return raw
