"""Versioned canonical 128D profile for the dual-arm ReBot B601."""

from .base import DatasetSchema
from .canonical_128d import CANONICAL_128D_SCHEMA

REBOT_DUAL_SCHEMA_ID = 'rebot_dual_unified'
REBOT_DUAL_SCHEMA_VERSION = '2.0.0'
REBOT_DUAL_LAYOUT_ID = 'rebot_dual_128_v1'

REBOT_DUAL_128D_SCHEMA = DatasetSchema(
    schema_id=REBOT_DUAL_SCHEMA_ID,
    schema_version=REBOT_DUAL_SCHEMA_VERSION,
    layout_id=REBOT_DUAL_LAYOUT_ID,
    features=CANONICAL_128D_SCHEMA.features,
)
