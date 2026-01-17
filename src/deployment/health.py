"""
SAIL Health Monitoring

Health monitoring and dashboard for the distributed SAIL system.
Tracks node health, system metrics, and alerts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.hub.base import NodeState, NodeType

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """A single health metric."""

    name: str
    value: float
    unit: str = ""
    min_healthy: float | None = None
    max_healthy: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_healthy(self) -> bool:
        """Check if metric is within healthy range."""
        if self.min_healthy is not None and self.value < self.min_healthy:
            return False
        if self.max_healthy is not None and self.value > self.max_healthy:
            return False
        return True


@dataclass
class HealthAlert:
    """A health alert."""

    alert_id: str
    severity: AlertSeverity
    source: str
    message: str
    metric_name: str | None = None
    metric_value: float | None = None
    threshold: float | None = None
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    resolved: bool = False
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "source": self.source,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class NodeHealth:
    """Health status for a single node."""

    node_id: str
    node_type: NodeType
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    connection_state: NodeState = NodeState.DISCONNECTED
    last_heartbeat: datetime | None = None
    latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    error_count: int = 0
    metrics: dict[str, HealthMetric] = field(default_factory=dict)
    alerts: list[HealthAlert] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "status": self.status.value,
            "connection_state": self.connection_state.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "latency_ms": self.latency_ms,
            "uptime_seconds": self.uptime_seconds,
            "error_count": self.error_count,
            "metrics": {k: {"value": v.value, "unit": v.unit} for k, v in self.metrics.items()},
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "last_check": self.last_check.isoformat(),
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus = HealthStatus.UNKNOWN
    hub_status: HealthStatus = HealthStatus.UNKNOWN
    nodes: dict[str, NodeHealth] = field(default_factory=dict)
    total_nodes: int = 0
    healthy_nodes: int = 0
    degraded_nodes: int = 0
    unhealthy_nodes: int = 0
    active_alerts: list[HealthAlert] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "hub_status": self.hub_status.value,
            "total_nodes": self.total_nodes,
            "healthy_nodes": self.healthy_nodes,
            "degraded_nodes": self.degraded_nodes,
            "unhealthy_nodes": self.unhealthy_nodes,
            "active_alerts": len(self.active_alerts),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "last_update": self.last_update.isoformat(),
        }


class HealthConfig(BaseModel):
    """Health monitoring configuration."""

    # Check intervals
    health_check_interval_seconds: int = Field(default=30, description="Health check interval")
    heartbeat_timeout_seconds: int = Field(default=90, description="Heartbeat timeout")

    # Thresholds
    latency_warning_ms: float = Field(default=200.0, description="Latency warning threshold")
    latency_critical_ms: float = Field(default=500.0, description="Latency critical threshold")
    error_rate_warning: float = Field(default=0.05, description="Error rate warning (5%)")
    error_rate_critical: float = Field(default=0.15, description="Error rate critical (15%)")

    # Alerting
    alert_cooldown_seconds: int = Field(default=300, description="Alert cooldown")
    max_alerts_per_node: int = Field(default=10, description="Max alerts per node")

    # Retention
    metric_retention_hours: int = Field(default=24, description="Metric retention period")
    alert_retention_days: int = Field(default=7, description="Alert retention period")


class HealthMonitor:
    """
    Health monitoring system.

    Monitors health of hub and all satellite nodes,
    generates alerts, and provides dashboard data.
    """

    def __init__(self, config: HealthConfig | None = None) -> None:
        """Initialize health monitor."""
        self._config = config or HealthConfig()

        # Health state
        self._system_health = SystemHealth()
        self._node_health: dict[str, NodeHealth] = {}

        # Alert management
        self._alerts: list[HealthAlert] = []
        self._alert_cooldowns: dict[str, datetime] = {}

        # Callbacks
        self._alert_callbacks: list[Callable[[HealthAlert], None]] = []
        self._status_callbacks: list[Callable[[SystemHealth], None]] = []

        # Background task
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        # Statistics
        self._stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "alerts_resolved": 0,
        }

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Health monitor stopped")

    def register_node(
        self,
        node_id: str,
        node_type: NodeType,
        name: str,
    ) -> NodeHealth:
        """Register a node for health monitoring."""
        health = NodeHealth(
            node_id=node_id,
            node_type=node_type,
            name=name,
        )
        self._node_health[node_id] = health
        self._system_health.nodes[node_id] = health
        self._update_system_health()

        logger.info(f"Registered node for health monitoring: {node_id}")
        return health

    def unregister_node(self, node_id: str) -> None:
        """Unregister a node from health monitoring."""
        self._node_health.pop(node_id, None)
        self._system_health.nodes.pop(node_id, None)
        self._update_system_health()

    def update_node_health(
        self,
        node_id: str,
        connection_state: NodeState | None = None,
        latency_ms: float | None = None,
        heartbeat: bool = False,
        error: str | None = None,
    ) -> NodeHealth | None:
        """
        Update health data for a node.

        Args:
            node_id: Node identifier
            connection_state: Current connection state
            latency_ms: Network latency
            heartbeat: Whether this is a heartbeat update
            error: Error message if any

        Returns:
            Updated node health
        """
        health = self._node_health.get(node_id)
        if not health:
            return None

        if connection_state:
            health.connection_state = connection_state

        if latency_ms is not None:
            health.latency_ms = latency_ms
            health.metrics["latency"] = HealthMetric(
                name="latency",
                value=latency_ms,
                unit="ms",
                max_healthy=self._config.latency_warning_ms,
            )

        if heartbeat:
            health.last_heartbeat = datetime.now()

        if error:
            health.error_count += 1
            self._check_error_alert(health, error)

        # Recalculate status
        health.status = self._calculate_node_status(health)
        health.last_check = datetime.now()

        self._update_system_health()
        return health

    def record_metric(
        self,
        node_id: str,
        name: str,
        value: float,
        unit: str = "",
        min_healthy: float | None = None,
        max_healthy: float | None = None,
    ) -> None:
        """Record a health metric for a node."""
        health = self._node_health.get(node_id)
        if not health:
            return

        metric = HealthMetric(
            name=name,
            value=value,
            unit=unit,
            min_healthy=min_healthy,
            max_healthy=max_healthy,
        )

        health.metrics[name] = metric

        # Check for alerts
        if not metric.is_healthy:
            self._generate_alert(
                source=node_id,
                severity=AlertSeverity.WARNING,
                message=f"Metric {name} out of healthy range: {value}{unit}",
                metric_name=name,
                metric_value=value,
                threshold=max_healthy or min_healthy,
            )

    def _calculate_node_status(self, health: NodeHealth) -> HealthStatus:
        """Calculate overall health status for a node."""
        # Check connection
        if health.connection_state == NodeState.DISCONNECTED:
            return HealthStatus.UNHEALTHY
        if health.connection_state == NodeState.ERROR:
            return HealthStatus.CRITICAL

        # Check heartbeat
        if health.last_heartbeat:
            elapsed = (datetime.now() - health.last_heartbeat).total_seconds()
            if elapsed > self._config.heartbeat_timeout_seconds:
                return HealthStatus.UNHEALTHY

        # Check latency
        if health.latency_ms > self._config.latency_critical_ms:
            return HealthStatus.CRITICAL
        if health.latency_ms > self._config.latency_warning_ms:
            return HealthStatus.DEGRADED

        # Check metrics
        unhealthy_metrics = sum(1 for m in health.metrics.values() if not m.is_healthy)
        if unhealthy_metrics > 2:
            return HealthStatus.UNHEALTHY
        if unhealthy_metrics > 0:
            return HealthStatus.DEGRADED

        # Check active alerts
        active_alerts = [a for a in health.alerts if not a.resolved]
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            return HealthStatus.CRITICAL
        if active_alerts:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _update_system_health(self) -> None:
        """Update overall system health."""
        self._system_health.total_nodes = len(self._node_health)
        self._system_health.healthy_nodes = sum(
            1 for h in self._node_health.values() if h.status == HealthStatus.HEALTHY
        )
        self._system_health.degraded_nodes = sum(
            1 for h in self._node_health.values() if h.status == HealthStatus.DEGRADED
        )
        self._system_health.unhealthy_nodes = sum(
            1 for h in self._node_health.values()
            if h.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL)
        )

        # Calculate system status
        if self._system_health.unhealthy_nodes > 0:
            self._system_health.status = HealthStatus.UNHEALTHY
        elif self._system_health.degraded_nodes > 0:
            self._system_health.status = HealthStatus.DEGRADED
        elif self._system_health.healthy_nodes == self._system_health.total_nodes:
            self._system_health.status = HealthStatus.HEALTHY
        else:
            self._system_health.status = HealthStatus.UNKNOWN

        # Update active alerts
        self._system_health.active_alerts = [a for a in self._alerts if not a.resolved]
        self._system_health.last_update = datetime.now()

        # Notify callbacks
        for callback in self._status_callbacks:
            try:
                callback(self._system_health)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def _generate_alert(
        self,
        source: str,
        severity: AlertSeverity,
        message: str,
        metric_name: str | None = None,
        metric_value: float | None = None,
        threshold: float | None = None,
    ) -> HealthAlert | None:
        """Generate a health alert."""
        import uuid

        # Check cooldown
        cooldown_key = f"{source}:{metric_name or message}"
        if cooldown_key in self._alert_cooldowns:
            if datetime.now() < self._alert_cooldowns[cooldown_key]:
                return None

        alert = HealthAlert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            source=source,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
        )

        self._alerts.append(alert)
        self._alert_cooldowns[cooldown_key] = datetime.now() + timedelta(
            seconds=self._config.alert_cooldown_seconds
        )

        # Add to node if applicable
        if source in self._node_health:
            node_alerts = self._node_health[source].alerts
            node_alerts.append(alert)
            # Trim old alerts
            while len(node_alerts) > self._config.max_alerts_per_node:
                node_alerts.pop(0)

        self._stats["alerts_generated"] += 1

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.warning(f"Health alert: [{severity.value}] {source}: {message}")
        return alert

    def _check_error_alert(self, health: NodeHealth, error: str) -> None:
        """Check if error warrants an alert."""
        # Simple threshold - alert after 3 errors
        if health.error_count % 3 == 0:
            self._generate_alert(
                source=health.node_id,
                severity=AlertSeverity.ERROR,
                message=f"Multiple errors detected: {error}",
            )

    async def _monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)
                self._stats["total_checks"] += 1

                now = datetime.now()

                # Check each node
                for node_id, health in self._node_health.items():
                    # Check for stale heartbeat
                    if health.last_heartbeat:
                        elapsed = (now - health.last_heartbeat).total_seconds()
                        if elapsed > self._config.heartbeat_timeout_seconds:
                            if health.status != HealthStatus.UNHEALTHY:
                                self._generate_alert(
                                    source=node_id,
                                    severity=AlertSeverity.WARNING,
                                    message="Node heartbeat timeout",
                                )
                                health.status = HealthStatus.UNHEALTHY

                # Clean up old alerts
                retention = timedelta(days=self._config.alert_retention_days)
                self._alerts = [
                    a for a in self._alerts
                    if not a.resolved or (now - a.created_at) < retention
                ]

                self._update_system_health()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self._stats["alerts_resolved"] += 1
                self._update_system_health()
                return True
        return False

    def get_system_health(self) -> SystemHealth:
        """Get current system health."""
        return self._system_health

    def get_node_health(self, node_id: str) -> NodeHealth | None:
        """Get health for a specific node."""
        return self._node_health.get(node_id)

    def get_active_alerts(self) -> list[HealthAlert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self._alerts if not a.resolved]

    def on_alert(self, callback: Callable[[HealthAlert], None]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def on_status_change(self, callback: Callable[[SystemHealth], None]) -> None:
        """Register status change callback."""
        self._status_callbacks.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get health monitor statistics."""
        return {
            **self._stats,
            "monitored_nodes": len(self._node_health),
            "total_alerts": len(self._alerts),
            "active_alerts": len(self.get_active_alerts()),
        }

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data for health dashboard."""
        return {
            "system": self._system_health.to_dict(),
            "nodes": [h.to_dict() for h in self._node_health.values()],
            "alerts": [a.to_dict() for a in self.get_active_alerts()],
            "stats": self.get_stats(),
        }
