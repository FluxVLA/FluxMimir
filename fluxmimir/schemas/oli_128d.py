"""Versioned canonical 128D profile for the LimX Oli humanoid."""

from .base import DatasetSchema
from .canonical_128d import canonical_features

OLI_SCHEMA_ID = 'oli_unified'
OLI_SCHEMA_VERSION = '1.0.0'
OLI_LAYOUT_ID = 'oli_128_v1'
OLI_CAMERA_ORDER = ('head', 'left_wrist')
OLI_STATE_DIM = 33
OLI_ACTION_DIM = 42
OLI_STATE_JOINT_NAMES = tuple(
    'left_hip_pitch_joint left_hip_roll_joint left_hip_yaw_joint '
    'left_knee_joint left_ankle_pitch_joint left_ankle_roll_joint '
    'right_hip_pitch_joint right_hip_roll_joint right_hip_yaw_joint '
    'right_knee_joint right_ankle_pitch_joint right_ankle_roll_joint '
    'waist_yaw_joint waist_roll_joint waist_pitch_joint '
    'head_yaw_joint head_pitch_joint left_shoulder_pitch_joint '
    'left_shoulder_roll_joint left_shoulder_yaw_joint left_elbow_joint '
    'left_wrist_yaw_joint left_wrist_pitch_joint left_wrist_roll_joint '
    'right_shoulder_pitch_joint right_shoulder_roll_joint '
    'right_shoulder_yaw_joint right_elbow_joint right_wrist_yaw_joint '
    'right_wrist_pitch_joint right_wrist_roll_joint left_hand_closed '
    'right_hand_closed'.split())

OLI_128D_SCHEMA = DatasetSchema(
    schema_id=OLI_SCHEMA_ID,
    schema_version=OLI_SCHEMA_VERSION,
    layout_id=OLI_LAYOUT_ID,
    features=canonical_features(OLI_CAMERA_ORDER),
)


def oli_qpose_indices() -> tuple[int, ...]:
    """Return the versioned oli arm/gripper view indices."""

    return (*range(0, 7), 28, *range(7, 14), 29)
