"""
Lightweight fallback implementation of attridict.
Provides attribute-style access to dictionaries.
"""
from __future__ import annotations

from typing import Any, Mapping


class AttriDict(dict):
    def __init__(self, initial: Mapping[str, Any] | None = None, **kwargs: Any):
        data = dict(initial or {})
        data.update(kwargs)
        super().__init__(data)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def copy(self) -> "AttriDict":
        return AttriDict(self)

    def to_dict(self) -> dict:
        return dict(self)


def attridict(data: Mapping[str, Any] | None = None, **kwargs: Any) -> AttriDict:
    return AttriDict(data, **kwargs)


__all__ = ["AttriDict", "attridict"]
