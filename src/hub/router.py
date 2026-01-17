"""
SAIL Request Router

Routes requests from nodes to appropriate handlers, manages request
prioritization, and handles load balancing across the hub.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Strategy for routing requests."""

    DIRECT = "direct"  # Route directly to handler
    QUEUE = "queue"  # Queue for processing
    PRIORITY = "priority"  # Priority-based queue
    LOAD_BALANCED = "load_balanced"  # Distribute load


class RequestPriority(int, Enum):
    """Priority levels for requests."""

    CRITICAL = 0  # Crisis/emergency
    HIGH = 1  # Intervention triggers
    NORMAL = 2  # Regular queries
    LOW = 3  # Background sync/telemetry


@dataclass
class RoutedRequest:
    """A request in the routing system."""

    request_id: str
    request_type: str
    source_node_id: str
    payload: dict[str, Any]
    priority: RequestPriority = RequestPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    timeout_seconds: float = 30.0
    retries: int = 0
    max_retries: int = 3
    user_id: str | None = None
    callback: Callable | None = None

    def is_expired(self) -> bool:
        """Check if request has expired."""
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.timeout_seconds


@dataclass
class RouteConfig:
    """Configuration for a route."""

    request_type: str
    handler: Callable
    priority: RequestPriority = RequestPriority.NORMAL
    timeout_seconds: float = 30.0
    rate_limit: int | None = None  # Requests per second
    require_auth: bool = True
    allowed_node_types: list[str] | None = None


@dataclass
class RateLimitState:
    """Rate limiting state for a route."""

    tokens: float
    last_update: float
    max_tokens: float
    refill_rate: float  # Tokens per second


class RequestRouter:
    """
    Routes requests from nodes to handlers.

    Supports multiple routing strategies, rate limiting,
    priority queues, and request timeout management.
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.PRIORITY,
        max_queue_size: int = 1000,
        worker_count: int = 4,
    ) -> None:
        """
        Initialize request router.

        Args:
            strategy: Routing strategy to use
            max_queue_size: Maximum queue size
            worker_count: Number of worker tasks
        """
        self._strategy = strategy
        self._max_queue_size = max_queue_size
        self._worker_count = worker_count

        # Route configuration
        self._routes: dict[str, RouteConfig] = {}

        # Request queues by priority
        self._queues: dict[RequestPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size)
            for priority in RequestPriority
        }

        # Rate limiting
        self._rate_limits: dict[str, RateLimitState] = {}

        # Workers
        self._workers: list[asyncio.Task] = []
        self._running = False

        # Statistics
        self._stats = {
            "total_routed": 0,
            "successful": 0,
            "failed": 0,
            "timed_out": 0,
            "rate_limited": 0,
            "queue_full": 0,
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int),
        }

        # Pending responses
        self._pending: dict[str, asyncio.Future] = {}

    async def start(self) -> None:
        """Start the router workers."""
        if self._running:
            return

        self._running = True

        # Start worker tasks
        for i in range(self._worker_count):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)

        logger.info(f"Request router started with {self._worker_count} workers")

    async def stop(self) -> None:
        """Stop the router workers."""
        if not self._running:
            return

        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

        self._workers.clear()

        # Cancel pending requests
        for future in self._pending.values():
            if not future.done():
                future.cancel()

        self._pending.clear()
        logger.info("Request router stopped")

    def register_route(
        self,
        request_type: str,
        handler: Callable[[RoutedRequest], Awaitable[dict[str, Any]]],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_seconds: float = 30.0,
        rate_limit: int | None = None,
        require_auth: bool = True,
        allowed_node_types: list[str] | None = None,
    ) -> None:
        """
        Register a route for a request type.

        Args:
            request_type: Type of request to handle
            handler: Async handler function
            priority: Default priority for this route
            timeout_seconds: Request timeout
            rate_limit: Optional rate limit (requests/second)
            require_auth: Whether auth is required
            allowed_node_types: Restrict to specific node types
        """
        self._routes[request_type] = RouteConfig(
            request_type=request_type,
            handler=handler,
            priority=priority,
            timeout_seconds=timeout_seconds,
            rate_limit=rate_limit,
            require_auth=require_auth,
            allowed_node_types=allowed_node_types,
        )

        if rate_limit:
            self._rate_limits[request_type] = RateLimitState(
                tokens=float(rate_limit),
                last_update=time.monotonic(),
                max_tokens=float(rate_limit),
                refill_rate=float(rate_limit),
            )

        logger.debug(f"Registered route: {request_type}")

    def unregister_route(self, request_type: str) -> None:
        """Unregister a route."""
        self._routes.pop(request_type, None)
        self._rate_limits.pop(request_type, None)

    async def route(
        self,
        request_id: str,
        request_type: str,
        source_node_id: str,
        payload: dict[str, Any],
        priority: RequestPriority | None = None,
        timeout_seconds: float | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Route a request to its handler.

        Args:
            request_id: Unique request identifier
            request_type: Type of request
            source_node_id: Source node
            payload: Request payload
            priority: Optional priority override
            timeout_seconds: Optional timeout override
            user_id: Optional user identifier

        Returns:
            Response from handler
        """
        self._stats["total_routed"] += 1
        self._stats["by_type"][request_type] += 1

        # Get route config
        route = self._routes.get(request_type)
        if not route:
            self._stats["failed"] += 1
            raise ValueError(f"No route for request type: {request_type}")

        # Use route defaults if not overridden
        actual_priority = priority if priority is not None else route.priority
        actual_timeout = timeout_seconds if timeout_seconds is not None else route.timeout_seconds

        self._stats["by_priority"][actual_priority.name] += 1

        # Check rate limit
        if not self._check_rate_limit(request_type):
            self._stats["rate_limited"] += 1
            raise RateLimitError(f"Rate limit exceeded for {request_type}")

        # Create request
        request = RoutedRequest(
            request_id=request_id,
            request_type=request_type,
            source_node_id=source_node_id,
            payload=payload,
            priority=actual_priority,
            timeout_seconds=actual_timeout,
            user_id=user_id,
        )

        # Route based on strategy
        if self._strategy == RoutingStrategy.DIRECT:
            return await self._route_direct(request, route)

        elif self._strategy in (RoutingStrategy.QUEUE, RoutingStrategy.PRIORITY):
            return await self._route_queued(request, route)

        elif self._strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._route_load_balanced(request, route)

        else:
            raise ValueError(f"Unknown routing strategy: {self._strategy}")

    async def _route_direct(
        self,
        request: RoutedRequest,
        route: RouteConfig,
    ) -> dict[str, Any]:
        """Route directly to handler."""
        try:
            result = await asyncio.wait_for(
                route.handler(request),
                timeout=request.timeout_seconds,
            )
            self._stats["successful"] += 1
            return result

        except TimeoutError:
            self._stats["timed_out"] += 1
            raise TimeoutError(f"Request {request.request_id} timed out")

        except Exception:
            self._stats["failed"] += 1
            raise

    async def _route_queued(
        self,
        request: RoutedRequest,
        route: RouteConfig,
    ) -> dict[str, Any]:
        """Route through priority queue."""
        queue = self._queues[request.priority]

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending[request.request_id] = future

        try:
            # Add to queue
            try:
                queue.put_nowait((request, route, future))
            except asyncio.QueueFull:
                self._stats["queue_full"] += 1
                raise QueueFullError(f"Queue full for priority {request.priority.name}")

            # Wait for response
            result = await asyncio.wait_for(future, timeout=request.timeout_seconds)
            return result

        except TimeoutError:
            self._stats["timed_out"] += 1
            raise TimeoutError(f"Request {request.request_id} timed out")

        finally:
            self._pending.pop(request.request_id, None)

    async def _route_load_balanced(
        self,
        request: RoutedRequest,
        route: RouteConfig,
    ) -> dict[str, Any]:
        """Route with load balancing."""
        # For now, use queued routing
        # In production, would distribute across multiple handlers
        return await self._route_queued(request, route)

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing queued requests."""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Check queues in priority order
                for priority in RequestPriority:
                    queue = self._queues[priority]

                    if not queue.empty():
                        request, route, future = await asyncio.wait_for(
                            queue.get(),
                            timeout=0.1,
                        )

                        if request.is_expired():
                            if not future.done():
                                future.set_exception(
                                    TimeoutError(f"Request {request.request_id} expired")
                                )
                            continue

                        try:
                            result = await route.handler(request)
                            if not future.done():
                                future.set_result(result)
                            self._stats["successful"] += 1

                        except Exception as e:
                            if not future.done():
                                future.set_exception(e)
                            self._stats["failed"] += 1

                        break  # Process one request per iteration

                else:
                    # No requests in any queue
                    await asyncio.sleep(0.01)

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.debug(f"Worker {worker_id} stopped")

    def _check_rate_limit(self, request_type: str) -> bool:
        """Check if request is within rate limit."""
        if request_type not in self._rate_limits:
            return True

        state = self._rate_limits[request_type]
        now = time.monotonic()
        elapsed = now - state.last_update

        # Refill tokens
        state.tokens = min(
            state.max_tokens,
            state.tokens + elapsed * state.refill_rate,
        )
        state.last_update = now

        # Check availability
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            return True

        return False

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            **self._stats,
            "strategy": self._strategy.value,
            "registered_routes": list(self._routes.keys()),
            "queue_sizes": {
                priority.name: self._queues[priority].qsize()
                for priority in RequestPriority
            },
            "pending_requests": len(self._pending),
            "workers_running": len(self._workers),
        }

    def get_queue_depth(self, priority: RequestPriority | None = None) -> int:
        """Get queue depth for a priority or total."""
        if priority:
            return self._queues[priority].qsize()
        return sum(q.qsize() for q in self._queues.values())

    @property
    def is_running(self) -> bool:
        """Check if router is running."""
        return self._running


class RateLimitError(Exception):
    """Rate limit exceeded."""

    pass


class QueueFullError(Exception):
    """Request queue is full."""

    pass
