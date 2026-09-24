from dataclasses import dataclass

from .base import DatasetSchema, FeatureSpec

SCHEMA_ID = 'agilex_aloha_unified'
SCHEMA_VERSION = '2.0.0'
LAYOUT_ID = 'lingbot_vla2_dexhand32x2_128_v1'
UNIFIED_DIM = 128
CAMERA_ORDER = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')

PARQUET_COLUMNS = (
    'observation.unified_128d',
    'observation.unified_128d_mask',
    'observation.ego2cam',
    'observation.ego2cam_valid',
    'target.unified_128d',
    'target.unified_128d_mask',
    'target.source',
    'timestamp',
    'frame_index',
    'episode_index',
    'index',
    'subtask_index',
)

METADATA_FILES = (
    'meta/info.json',
    'meta/tasks.jsonl',
    'meta/subtasks.jsonl',
    'meta/episodes.jsonl',
    'meta/episodes_stats.jsonl',
    'meta/fluxvla_dataset_version.json',
    'meta/artifacts.sha256.json',
)

TARGET_SOURCE = {
    'policy': 0,
    'operator': 1,
    'blended_policy_operator': 2,
    'safety_override': 3,
    'scripted': 4,
    'unknown': 5,
}


@dataclass(frozen=True)
class LayoutRange:
    name: str
    start: int
    end: int
    unit: str | None = None

    @property
    def size(self) -> int:
        return self.end - self.start


LAYOUT_RANGES = (
    LayoutRange('arm_joint_position', 0, 14, 'rad'),
    LayoutRange('end_effector_pose', 14, 28, 'mixed'),
    LayoutRange('gripper_position', 28, 30, 'm'),
    LayoutRange('left_dexterous_hand', 30, 62),
    LayoutRange('right_dexterous_hand', 62, 94),
    LayoutRange('waist_position', 94, 98),
    LayoutRange('head_position', 98, 100),
    LayoutRange('mobile_base_signal', 100, 103),
    LayoutRange('left_leg_joint_position', 103, 109, 'rad'),
    LayoutRange('right_leg_joint_position', 109, 115, 'rad'),
    LayoutRange('mobile_base_rotation_6d', 115, 121),
    LayoutRange('reserved', 121, 128),
)

CANONICAL_128D_SCHEMA = DatasetSchema(
    schema_id=SCHEMA_ID,
    schema_version=SCHEMA_VERSION,
    layout_id=LAYOUT_ID,
    features=(
        FeatureSpec('observation.unified_128d', 'float32', (UNIFIED_DIM, )),
        FeatureSpec('observation.unified_128d_mask', 'bool', (UNIFIED_DIM, )),
        FeatureSpec('observation.ego2cam', 'float32', (3, 4, 4)),
        FeatureSpec('observation.ego2cam_valid', 'bool', (3, )),
        FeatureSpec('target.unified_128d', 'float32', (UNIFIED_DIM, )),
        FeatureSpec('target.unified_128d_mask', 'bool', (UNIFIED_DIM, )),
        FeatureSpec('target.source', 'int8', ()),
        FeatureSpec('timestamp', 'float32', ()),
        FeatureSpec('frame_index', 'int64', ()),
        FeatureSpec('episode_index', 'int64', ()),
        FeatureSpec('index', 'int64', ()),
        FeatureSpec('subtask_index', 'int64', ()),
    ),
)


def aloha_qpose_indices() -> tuple[int, ...]:
    """Return the versioned 14D ALOHA qpose view indices."""

    return (*range(0, 6), 28, *range(7, 13), 29)
