"""
Event emitter mixin.

Replaces 15 copy-pasted callback registration + emission patterns
across the codebase with a single reusable mixin.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EventEmitter(Generic[T]):
    """Mixin that provides event callback registration and emission.

    Every class that uses this mixin must call ``self.__init_emitter__()``
    inside its own ``__init__``.  This avoids MRO surprises when the class
    also inherits from another base.

    Usage::

        class MyManager(EventEmitter[MyEvent]):
            def __init__(self):
                self.__init_emitter__()
                # ... other init ...

            def do_work(self):
                self._emit(MyEvent(event_type="done"))
    """

    def __init_emitter__(self) -> None:
        """Initialize the event callback list.  Call from ``__init__``."""
        self._event_callbacks: list[Callable[[T], None]] = []

    # -- public API --

    def on_event(self, callback: Callable[[T], None]) -> None:
        """Register a callback invoked on every emitted event."""
        self._event_callbacks.append(callback)

    # Alias used by several subsystems
    register_event_callback = on_event

    # -- protected API --

    def _emit(self, event: T) -> None:
        """Send *event* to all registered callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
