"""
SAIL Hub Module

Central hub coordination for satellite node deployment and management.
The hub acts as the brain of the distributed SAIL system, coordinating
context synchronization, request routing, and node health monitoring.
"""

from src.hub.base import (
    HubConfig,
    HubState,
    Node,
    NodeCapability,
    NodeInfo,
    NodeState,
    NodeType,
)
from src.hub.context_sync import ContextSync, SyncConflict, SyncStrategy
from src.hub.coordinator import HubCoordinator
from src.hub.router import RequestRouter, RoutingStrategy

__all__ = [
    # Context sync
    "ContextSync",
    # Base types
    "HubConfig",
    # Coordinator
    "HubCoordinator",
    "HubState",
    "Node",
    "NodeCapability",
    "NodeInfo",
    "NodeState",
    "NodeType",
    # Router
    "RequestRouter",
    "RoutingStrategy",
    "SyncConflict",
    "SyncStrategy",
]
