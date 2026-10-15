"""Versioned canonical 128D profile for the dual-arm Franka setup."""

from .base import DatasetSchema
from .canonical_128d import CANONICAL_128D_SCHEMA

FRANKA_DUAL_SCHEMA_ID = 'franka_dual_unified'
FRANKA_DUAL_SCHEMA_VERSION = '2.0.0'
FRANKA_DUAL_LAYOUT_ID = 'franka_dual_128_v1'

FRANKA_DUAL_128D_SCHEMA = DatasetSchema(
    schema_id=FRANKA_DUAL_SCHEMA_ID,
    schema_version=FRANKA_DUAL_SCHEMA_VERSION,
    layout_id=FRANKA_DUAL_LAYOUT_ID,
    features=CANONICAL_128D_SCHEMA.features,
)


def franka_qpose_indices() -> tuple[int, ...]:
    """Return the versioned franka qpose view indices."""

    return (*range(0, 7), 28, *range(7, 14), 29)
