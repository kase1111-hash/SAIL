"""
SAIL Deployment Module

Node deployment automation, OTA updates, and health monitoring
for the distributed SAIL system.
"""

from src.deployment.health import (
    HealthConfig,
    HealthMonitor,
    HealthStatus,
    NodeHealth,
    SystemHealth,
)
from src.deployment.ota import (
    FirmwareUpdate,
    OTAConfig,
    OTAManager,
    UpdateChannel,
    UpdateState,
)
from src.deployment.provisioning import (
    NodeProvisioner,
    ProvisioningConfig,
    ProvisioningResult,
)

__all__ = [
    "FirmwareUpdate",
    "HealthConfig",
    # Health
    "HealthMonitor",
    "HealthStatus",
    "NodeHealth",
    # Provisioning
    "NodeProvisioner",
    "OTAConfig",
    # OTA
    "OTAManager",
    "ProvisioningConfig",
    "ProvisioningResult",
    "SystemHealth",
    "UpdateChannel",
    "UpdateState",
]
