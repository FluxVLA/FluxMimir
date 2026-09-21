from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fluxmimir.conversion.structures import EpisodeRecord
from fluxmimir.engines.utils import EMBODIMENTS
from fluxmimir.schemas import (CAMERA_ORDER, FRANKA_DUAL_LAYOUT_ID,
                               FRANKA_DUAL_SCHEMA_ID,
                               FRANKA_DUAL_SCHEMA_VERSION, TARGET_SOURCE,
                               UNIFIED_DIM)
from fluxmimir.structures import (IssueSeverity, ValidationIssue,
                                  ValidationReport)

from .base import BaseEmbodimentAdapter

FRANKA_DUAL_EMBODIMENT_ID = 'franka_dual'
FRANKA_DUAL_EMBODIMENT_DESCRIPTION = (
    'Dual Franka Panda arms with parallel grippers')
FRANKA_DUAL_JOINT_NAMES = (
    'left_panda_joint1',
    'left_panda_joint2',
    'left_panda_joint3',
    'left_panda_joint4',
    'left_panda_joint5',
    'left_panda_joint6',
    'left_panda_joint7',
    'left_gripper',
    'right_panda_joint1',
    'right_panda_joint2',
    'right_panda_joint3',
    'right_panda_joint4',
    'right_panda_joint5',
    'right_panda_joint6',
    'right_panda_joint7',
    'right_gripper',
)


def _map_state(values: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[0], UNIFIED_DIM), dtype=np.float32)
    result[:, 0:7] = values[:, 0:7]
    result[:, 7:14] = values[:, 8:15]
    result[:, 28] = values[:, 7]
    result[:, 29] = values[:, 15]
    return result


def _state_mask(length: int) -> np.ndarray:
    mask = np.zeros((length, UNIFIED_DIM), dtype=np.bool_)
    mask[:, 0:14] = True
    mask[:, 28:30] = True
    return mask


def _map_pose(values: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[0], UNIFIED_DIM), dtype=np.float32)
    result[:, 14:21] = values[:, 0:7]
    result[:, 21:28] = values[:, 8:15]
    return result


@EMBODIMENTS.register_module()
class FrankaDualAdapter(BaseEmbodimentAdapter):
    """Map dual Franka feedback into the canonical 107D layout."""

    schema_id = FRANKA_DUAL_SCHEMA_ID
    schema_version = FRANKA_DUAL_SCHEMA_VERSION
    layout_id = FRANKA_DUAL_LAYOUT_ID
    embodiment_id = FRANKA_DUAL_EMBODIMENT_ID
    embodiment_description = FRANKA_DUAL_EMBODIMENT_DESCRIPTION
    robot_type = 'franka_dual'
    camera_order = CAMERA_ORDER
    gripper_unit = 'm'
    layout_profile = 'franka'
    active_ranges = ((0, 14), (28, 30))
    qpose_reconstruction = 'unified[0:14] + unified[28:30]'
    source_format = 'LeRobot v2.1 franka_dual episodes'
    conversion_description = 'streaming LeRobot v2.1 to canonical unified 107D'
    video_materialization = 'direct_streaming_encode'
    training_readiness_blockers = (
        'raw action and target provenance are unavailable in this export',
        'calibrated ego2cam is unavailable',
        'hardware timestamps are unavailable for sampled source data',
        'frame-level subtasks are unavailable',
    )
    unavailable_annotation_source = 'unavailable_in_lerobot_v2'

    def __init__(
        self,
        action_semantics: str = 'unknown',
        action_source: str = 'operator',
        nominal_fps: int = 30,
        missing_dense_value: float = 0.0,
        gripper_min_m: float = 0.0,
        gripper_max_m: float = 0.081,
    ) -> None:
        if action_semantics not in ('unknown', 'issued_joint_position_target'):
            raise ValueError(
                f'unsupported action semantics: {action_semantics}')
        if action_source not in TARGET_SOURCE:
            raise ValueError(f'unknown target source: {action_source}')
        if nominal_fps <= 0:
            raise ValueError('nominal_fps must be positive')
        if gripper_min_m >= gripper_max_m:
            raise ValueError('gripper range must be increasing')
        self.action_semantics = action_semantics
        self.action_source = action_source
        self.nominal_fps = int(nominal_fps)
        self.missing_dense_value = float(missing_dense_value)
        self.gripper_min_m = float(gripper_min_m)
        self.gripper_max_m = float(gripper_max_m)

    def preflight(self, episode: EpisodeRecord) -> ValidationReport:
        issues: list[ValidationIssue] = []
        state = np.asarray(episode.fields.get('observations/qpos'))
        expected_shape = (episode.length, len(FRANKA_DUAL_JOINT_NAMES))
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
        else:
            gripper = state[:, (7, 15)]
            tolerance = 1e-5
            if (np.any(gripper < self.gripper_min_m - tolerance)
                    or np.any(gripper > self.gripper_max_m + tolerance)):
                issues.append(
                    ValidationIssue(
                        'gripper_unit_range',
                        f'gripper state outside [{self.gripper_min_m}, '
                        f'{self.gripper_max_m}] m',
                        location=episode.ref.source_uri))

        eepose = episode.fields.get('observations/eepose')
        if eepose is not None:
            eepose = np.asarray(eepose)
            if eepose.shape != expected_shape:
                issues.append(
                    ValidationIssue(
                        'invalid_eepose_shape',
                        f'eepose shape {eepose.shape} != {expected_shape}',
                        location=episode.ref.source_uri))
            elif not np.isfinite(eepose).all():
                issues.append(
                    ValidationIssue(
                        'nonfinite_eepose',
                        'eepose contains NaN/Inf',
                        location=episode.ref.source_uri))

        names = tuple(episode.metadata.get('joint_names', ()))
        if names != FRANKA_DUAL_JOINT_NAMES:
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
        if self.action_semantics == 'unknown' and 'action' in episode.fields:
            issues.append(
                ValidationIssue(
                    'target_semantics_unknown',
                    'raw /action is provenance only; target remains an '
                    'invalid placeholder',
                    severity=IssueSeverity.WARNING,
                    location=episode.ref.source_uri))
        if 'timestamp' not in episode.fields:
            issues.append(
                ValidationIssue(
                    'synthetic_timestamp',
                    f'using frame_index/{self.nominal_fps}',
                    severity=IssueSeverity.WARNING,
                    location=episode.ref.source_uri))
        return ValidationReport(issues=tuple(issues))

    def map_observation(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        state = np.asarray(
            episode.fields['observations/qpos'], dtype=np.float32)
        unified = _map_state(state)
        mask = _state_mask(episode.length)

        eepose = episode.fields.get('observations/eepose')
        if eepose is not None:
            eepose = np.asarray(eepose, dtype=np.float32)
            if eepose.shape != (episode.length, len(FRANKA_DUAL_JOINT_NAMES)):
                raise ValueError(f'bad eepose shape: {eepose.shape}')
            if not np.isfinite(eepose).all():
                raise ValueError('eepose contains NaN/Inf')
            unified[:, 14:28] = _map_pose(eepose)[:, 14:28]
            mask[:, 14:28] = True

        return {
            'observation.unified_107d':
            unified,
            'observation.unified_107d_mask':
            mask,
            'observation.ego2cam':
            np.full(
                (episode.length, len(self.camera_order), 4, 4),
                self.missing_dense_value,
                dtype=np.float32,
            ),
            'observation.ego2cam_valid':
            np.zeros((episode.length, len(self.camera_order)), dtype=np.bool_),
        }

    def map_target(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        values = np.full((episode.length, UNIFIED_DIM),
                         self.missing_dense_value,
                         dtype=np.float32)
        mask = np.zeros_like(values, dtype=np.bool_)
        source = TARGET_SOURCE['unknown']
        if self.action_semantics == 'issued_joint_position_target':
            if 'action' not in episode.fields:
                raise ValueError(
                    'confirmed action semantics but /action is absent')
            action = np.asarray(episode.fields['action'], dtype=np.float32)
            values = _map_state(action)
            mask = _state_mask(episode.length)
            source = TARGET_SOURCE[self.action_source]
        return {
            'target.unified_107d': values,
            'target.unified_107d_mask': mask,
            'target.source': np.full(episode.length, source, dtype=np.int8),
        }
