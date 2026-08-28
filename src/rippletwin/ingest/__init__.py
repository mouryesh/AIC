"""Getting real plant data into the twin.

This package is the boundary between RippleTwin and a production line. Until it
existed, the inference path was typed to ``SimResult`` -- the simulator's output
-- which meant a real plant could not run this software at all, no matter how
good the method was. That is the classic prototype trap, and the reason so many
industrial analytics projects die between the demo and the pilot.

Everything here answers one question: *what does a plant have to hand us, and
what do we do when it does not have all of it?*
"""

from .plant_data import (
    OPTIONAL_TELEMETRY,
    REQUIRED_TELEMETRY,
    PlantData,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "PlantData",
    "ValidationReport",
    "ValidationIssue",
    "REQUIRED_TELEMETRY",
    "OPTIONAL_TELEMETRY",
]
