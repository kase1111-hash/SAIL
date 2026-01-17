"""
SAIL Deployment Module

Node deployment automation, OTA updates, and health monitoring
for the distributed SAIL system.
"""

from src.deployment.provisioning import (
    NodeProvisioner,
    ProvisioningConfig,
    ProvisioningResult,
)
from src.deployment.ota import (
    OTAManager,
    OTAConfig,
    FirmwareUpdate,
    UpdateChannel,
    UpdateState,
)
from src.deployment.health import (
    HealthMonitor,
    HealthConfig,
    HealthStatus,
    NodeHealth,
    SystemHealth,
)

__all__ = [
    # Provisioning
    "NodeProvisioner",
    "ProvisioningConfig",
    "ProvisioningResult",
    # OTA
    "OTAManager",
    "OTAConfig",
    "FirmwareUpdate",
    "UpdateChannel",
    "UpdateState",
    # Health
    "HealthMonitor",
    "HealthConfig",
    "HealthStatus",
    "NodeHealth",
    "SystemHealth",
]
