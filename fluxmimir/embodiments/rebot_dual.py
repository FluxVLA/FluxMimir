from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fluxmimir.conversion.structures import EpisodeRecord
from fluxmimir.engines.utils import EMBODIMENTS
from fluxmimir.schemas import (CAMERA_ORDER, REBOT_DUAL_LAYOUT_ID,
                               REBOT_DUAL_SCHEMA_ID, REBOT_DUAL_SCHEMA_VERSION,
                               TARGET_SOURCE, UNIFIED_DIM)
from fluxmimir.structures import ValidationIssue, ValidationReport

from .base import BaseEmbodimentAdapter

REBOT_DUAL_EMBODIMENT_ID = 'rebot_b601_dual'
REBOT_DUAL_EMBODIMENT_DESCRIPTION = 'ReBot B601 dual-arm follower robot'
REBOT_DUAL_JOINT_NAMES = (
    'left_shoulder_pan.pos',
    'left_shoulder_lift.pos',
    'left_elbow_flex.pos',
    'left_wrist_flex.pos',
    'left_wrist_yaw.pos',
    'left_wrist_roll.pos',
    'left_gripper.pos',
    'right_shoulder_pan.pos',
    'right_shoulder_lift.pos',
    'right_elbow_flex.pos',
    'right_wrist_flex.pos',
    'right_wrist_yaw.pos',
    'right_wrist_roll.pos',
    'right_gripper.pos',
)


def _map_joint_state(values: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[0], UNIFIED_DIM), dtype=np.float32)
    result[:, 0:6] = values[:, 0:6]
    result[:, 7:13] = values[:, 7:13]
    result[:, 28] = values[:, 6]
    result[:, 29] = values[:, 13]
    return result


def _joint_state_mask(length: int) -> np.ndarray:
    mask = np.zeros((length, UNIFIED_DIM), dtype=np.bool_)
    mask[:, 0:6] = True
    mask[:, 7:13] = True
    mask[:, 28:30] = True
    return mask


@EMBODIMENTS.register_module()
class ReBotDualAdapter(BaseEmbodimentAdapter):
    """Map ReBot feedback and ROS command targets into canonical 128D.

    The gripper values are motor angles in radians. Targets are the raw values
    observed on the two ``target_joint_state`` topics, before any follower
    driver soft-limit clipping.
    """

    schema_id = REBOT_DUAL_SCHEMA_ID
    schema_version = REBOT_DUAL_SCHEMA_VERSION
    layout_id = REBOT_DUAL_LAYOUT_ID
    embodiment_id = REBOT_DUAL_EMBODIMENT_ID
    embodiment_description = REBOT_DUAL_EMBODIMENT_DESCRIPTION
    robot_type = 'bi_rebot_b601_follower_ros'
    camera_order = CAMERA_ORDER
    gripper_unit = 'rad'
    layout_profile = 'rebot'
    active_ranges = ((0, 6), (7, 13), (28, 30))
    qpose_reconstruction = ('unified[0:6] + unified[28] + '
                            'unified[7:13] + unified[29]')
    source_format = 'LeRobot v3 Parquet and MP4'
    conversion_description = 'LeRobot v3 to ReBot canonical unified 128D'
    video_materialization = 'copied_source_mp4'
    training_readiness_blockers = (
        'calibrated ego2cam is unavailable',
        'hardware-clock provenance for timestamps is unverified',
        'frame-level subtasks are unavailable',
    )
    unavailable_annotation_source = 'unavailable_in_source'

    def __init__(
        self,
        action_semantics: str = 'issued_joint_position_target',
        action_source: str = 'operator',
        nominal_fps: int = 30,
        missing_dense_value: float = 0.0,
        gripper_min_rad: float = -4.712389,
        gripper_max_rad: float = 0.0,
    ) -> None:
        if action_semantics != 'issued_joint_position_target':
            raise ValueError('ReBot action semantics must be explicitly '
                             'confirmed as issued_joint_position_target')
        if action_source != 'operator':
            raise ValueError(
                'this ReBot profile is for operator teleoperation')
        if nominal_fps <= 0:
            raise ValueError('nominal_fps must be positive')
        self.action_semantics = action_semantics
        self.action_source = action_source
        self.nominal_fps = int(nominal_fps)
        self.missing_dense_value = float(missing_dense_value)
        self.gripper_min_rad = float(gripper_min_rad)
        self.gripper_max_rad = float(gripper_max_rad)
        self.target_provenance = {
            'source': 'left and right target_joint_state ROS topics',
            'recorder_read': 'rebot_ros_record.py:471',
            'left_right_concat': 'rebot_ros_record.py:528',
            'semantic': 'pre-driver-soft-limit joint-position target',
            'control_mode': 'operator teleoperation',
        }

    def preflight(self, episode: EpisodeRecord) -> ValidationReport:
        issues: list[ValidationIssue] = []
        state = np.asarray(episode.fields.get('observations/qpos'))
        action = np.asarray(episode.fields.get('action'))
        expected_shape = (episode.length, len(REBOT_DUAL_JOINT_NAMES))
        if state.shape != expected_shape:
            issues.append(
                ValidationIssue(
                    'invalid_state_shape',
                    f'state shape {state.shape} != {expected_shape}',
                    location=episode.ref.source_uri))
        elif not np.isfinite(state).all():
            issues.append(
                ValidationIssue(
                    'nonfinite_state',
                    'state contains NaN/Inf',
                    location=episode.ref.source_uri))
        if action.shape != expected_shape:
            issues.append(
                ValidationIssue(
                    'invalid_action_shape', f'action shape {action.shape} != '
                    f'{expected_shape}',
                    location=episode.ref.source_uri))
        elif not np.isfinite(action).all():
            issues.append(
                ValidationIssue(
                    'nonfinite_action',
                    'action contains NaN/Inf',
                    location=episode.ref.source_uri))
        else:
            gripper = action[:, (6, 13)]
            tolerance = 1e-5
            if (np.any(gripper < self.gripper_min_rad - tolerance)
                    or np.any(gripper > self.gripper_max_rad + tolerance)):
                issues.append(
                    ValidationIssue(
                        'gripper_target_range',
                        f'gripper target outside [{self.gripper_min_rad}, '
                        f'{self.gripper_max_rad}] rad',
                        location=episode.ref.source_uri))

        names = tuple(episode.metadata.get('joint_names', ()))
        if names != REBOT_DUAL_JOINT_NAMES:
            issues.append(
                ValidationIssue(
                    'joint_order',
                    f'joint names/order differ: {names}',
                    location=episode.ref.source_uri))
        missing = sorted(set(self.camera_order) - set(episode.media))
        if missing:
            issues.append(
                ValidationIssue(
                    'missing_cameras',
                    f'missing cameras: {missing}',
                    location=episode.ref.source_uri))
        return ValidationReport(issues=tuple(issues))

    def map_observation(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        values = np.asarray(
            episode.fields['observations/qpos'], dtype=np.float32)
        return {
            'observation.unified_128d':
            _map_joint_state(values),
            'observation.unified_128d_mask':
            _joint_state_mask(episode.length),
            'observation.ego2cam':
            np.full((episode.length, len(self.camera_order), 4, 4),
                    self.missing_dense_value,
                    dtype=np.float32),
            'observation.ego2cam_valid':
            np.zeros((episode.length, len(self.camera_order)), dtype=np.bool_),
        }

    def map_target(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        action = np.asarray(episode.fields['action'], dtype=np.float32)
        return {
            'target.unified_128d':
            _map_joint_state(action),
            'target.unified_128d_mask':
            _joint_state_mask(episode.length),
            'target.source':
            np.full(
                episode.length,
                TARGET_SOURCE[self.action_source],
                dtype=np.int8),
        }
