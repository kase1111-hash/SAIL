"""
SAIL Stats Protocol

Provides a standard interface for components that expose statistics.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HasStats(Protocol):
    """Protocol for components that expose statistics."""

    def get_stats(self) -> dict[str, Any]: ...
