"""
SAIL Nodes Module

Satellite node implementations for the distributed SAIL system.
Includes room nodes, mobile nodes, and vehicle nodes.
"""

from src.nodes.base import (
    BaseNode,
    NodeConfig,
    NodeEvent,
    NodeMessage,
    MessageType,
)
from src.nodes.protocol import (
    Protocol,
    ProtocolConfig,
    SecureChannel,
    MessageFrame,
)
from src.nodes.room import RoomNode, RoomNodeConfig
from src.nodes.mobile import MobileNode, MobileNodeConfig
from src.nodes.vehicle import VehicleNode, VehicleNodeConfig

__all__ = [
    # Base
    "BaseNode",
    "NodeConfig",
    "NodeEvent",
    "NodeMessage",
    "MessageType",
    # Protocol
    "Protocol",
    "ProtocolConfig",
    "SecureChannel",
    "MessageFrame",
    # Room
    "RoomNode",
    "RoomNodeConfig",
    # Mobile
    "MobileNode",
    "MobileNodeConfig",
    # Vehicle
    "VehicleNode",
    "VehicleNodeConfig",
]
