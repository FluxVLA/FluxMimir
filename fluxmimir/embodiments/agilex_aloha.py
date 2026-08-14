from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fluxmimir.conversion.structures import EpisodeRecord
from fluxmimir.engines.utils import EMBODIMENTS
from fluxmimir.schemas import (LAYOUT_ID, SCHEMA_ID, SCHEMA_VERSION,
                               TARGET_SOURCE, UNIFIED_DIM)
from fluxmimir.structures import (IssueSeverity, ValidationIssue,
                                  ValidationReport)

from .base import BaseEmbodimentAdapter

EMBODIMENT_ID = 'agilex_aloha'
EMBODIMENT_DESCRIPTION = 'AgileX ALOHA dual-arm robot'

SOURCE_CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')

REQUIRED_CONFIRMATIONS = (
    'joint_order',
    'action_semantics',
    'timestamp_source',
    'nominal_fps',
    'depth_unit',
    'camera_calibration',
)


def _resize_16d(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != 16:
        return values
    resized = np.zeros((*values.shape[:-1], 14), dtype=np.float32)
    resized[..., :6] = values[..., :6]
    resized[..., 6] = (values[..., 6] - values[..., 7]) * (0.1 / 0.07)
    resized[..., 7:13] = values[..., 8:14]
    resized[..., 13] = (values[..., 14] - values[..., 15]) * (0.1 / 0.07)
    return resized


def _map_qpose(values: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[0], UNIFIED_DIM), dtype=np.float32)
    result[:, 0:6] = values[:, 0:6]
    result[:, 7:13] = values[:, 7:13]
    result[:, 28] = values[:, 6]
    result[:, 29] = values[:, 13]
    return result


def _qpose_mask(length: int) -> np.ndarray:
    mask = np.zeros((length, UNIFIED_DIM), dtype=np.bool_)
    mask[:, 0:6] = True
    mask[:, 7:13] = True
    mask[:, 28:30] = True
    return mask


@EMBODIMENTS.register_module()
class AgileXAlohaAdapter(BaseEmbodimentAdapter):
    """Map physically decoded ALOHA tensors into canonical 107D.

    ``action_semantics`` deliberately defaults to ``unknown``. Merely finding
    an HDF dataset named ``action`` is not sufficient evidence that it is the
    issued controller target expected by supervised policies.
    """

    embodiment_id = EMBODIMENT_ID
    embodiment_description = EMBODIMENT_DESCRIPTION
    robot_type = 'aloha'
    gripper_unit = 'm'
    layout_profile = 'aloha'
    source_format = 'AgileX ALOHA episode HDF5'
    conversion_description = 'streaming HDF5 to canonical unified 107D'
    video_materialization = 'direct_streaming_encode'
    training_readiness_blockers = (
        'raw action semantics/provenance are unconfirmed',
        'calibrated ego2cam is unavailable',
        'hardware timestamps are unavailable for sampled source data',
        'episode-level task and frame subtasks are unavailable',
    )
    unavailable_annotation_source = 'unavailable_in_raw_hdf'

    def __init__(
        self,
        schema_id: str = SCHEMA_ID,
        schema_version: str = SCHEMA_VERSION,
        layout_id: str = LAYOUT_ID,
        require_confirmed: list[str]
        | tuple[str, ...] = REQUIRED_CONFIRMATIONS,
        missing_dense_value: float = 0.0,
        unknown_target_source: int = TARGET_SOURCE['unknown'],
        joint_order: str = 'left_arm_gripper_right_arm_gripper',
        action_semantics: str = 'unknown',
        action_source: str = 'operator',
        nominal_fps: int = 30,
    ) -> None:
        if (schema_id, schema_version,
                layout_id) != (SCHEMA_ID, SCHEMA_VERSION, LAYOUT_ID):
            raise ValueError('AgileXAlohaAdapter schema/layout mismatch')
        if joint_order != 'left_arm_gripper_right_arm_gripper':
            raise ValueError(f'unsupported ALOHA joint order: {joint_order}')
        if nominal_fps <= 0:
            raise ValueError('nominal_fps must be positive')
        if action_source not in TARGET_SOURCE:
            raise ValueError(f'unknown target source: {action_source}')
        self.schema_id = schema_id
        self.schema_version = schema_version
        self.layout_id = layout_id
        self.require_confirmed = tuple(require_confirmed)
        self.missing_dense_value = float(missing_dense_value)
        self.unknown_target_source = int(unknown_target_source)
        self.joint_order = joint_order
        self.action_semantics = action_semantics
        self.action_source = action_source
        self.nominal_fps = int(nominal_fps)

    def preflight(self, episode: EpisodeRecord) -> ValidationReport:
        issues: list[ValidationIssue] = []
        qpos = np.asarray(episode.fields.get('observations/qpos'))
        if qpos.ndim != 2 or qpos.shape[0] != episode.length:
            issues.append(
                ValidationIssue(
                    'invalid_qpos_shape',
                    f'qpos shape is {qpos.shape}',
                    location=episode.ref.source_uri))
        elif qpos.shape[1] not in (14, 16):
            issues.append(
                ValidationIssue(
                    'unsupported_qpos_width',
                    f'qpos width is {qpos.shape[1]}',
                    location=episode.ref.source_uri))
        elif not np.isfinite(qpos).all():
            issues.append(
                ValidationIssue(
                    'nonfinite_qpos',
                    'qpos contains NaN/Inf',
                    location=episode.ref.source_uri))
        if self.action_semantics == 'unknown' and 'action' in episode.fields:
            issues.append(
                ValidationIssue(
                    'target_semantics_unknown',
                    'raw /action is provenance only; target remains an '
                    'invalid placeholder',
                    severity=IssueSeverity.WARNING,
                    location=episode.ref.source_uri,
                ))
        if 'timestamp' not in episode.fields:
            issues.append(
                ValidationIssue(
                    'synthetic_timestamp',
                    f'using frame_index/{self.nominal_fps}',
                    severity=IssueSeverity.WARNING,
                    location=episode.ref.source_uri,
                ))
        missing_cameras = sorted(set(SOURCE_CAMERAS) - set(episode.media))
        if missing_cameras:
            issues.append(
                ValidationIssue(
                    'missing_cameras',
                    f'missing cameras: {missing_cameras}',
                    location=episode.ref.source_uri))
        return ValidationReport(issues=tuple(issues))

    def map_observation(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        qpos = _resize_16d(
            np.asarray(episode.fields['observations/qpos'], dtype=np.float32))
        unified = _map_qpose(qpos)
        mask = _qpose_mask(episode.length)

        eepose = episode.fields.get('observations/eepose')
        if eepose is not None:
            eepose = np.asarray(eepose, dtype=np.float32)
            if eepose.shape != (episode.length, 14):
                raise ValueError(f'bad eepose shape: {eepose.shape}')
            if not np.isfinite(eepose).all():
                raise ValueError('eepose contains NaN/Inf')
            unified[:, 14:28] = eepose
            mask[:, 14:28] = True

        return {
            'observation.unified_107d':
            unified,
            'observation.unified_107d_mask':
            mask,
            'observation.ego2cam':
            np.full(
                (episode.length, len(SOURCE_CAMERAS), 4, 4),
                self.missing_dense_value,
                dtype=np.float32,
            ),
            'observation.ego2cam_valid':
            np.zeros((episode.length, len(SOURCE_CAMERAS)), dtype=np.bool_),
        }

    def map_target(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        values = np.full((episode.length, UNIFIED_DIM),
                         self.missing_dense_value,
                         dtype=np.float32)
        mask = np.zeros_like(values, dtype=np.bool_)
        source = self.unknown_target_source
        if self.action_semantics == 'issued_joint_position_target':
            if 'action' not in episode.fields:
                raise ValueError(
                    'confirmed action semantics but /action is absent')
            action = _resize_16d(
                np.asarray(episode.fields['action'], dtype=np.float32))
            if action.shape != (episode.length, 14):
                raise ValueError(f'bad action shape: {action.shape}')
            if not np.isfinite(action).all():
                raise ValueError('action contains NaN/Inf')
            values = _map_qpose(action)
            mask = _qpose_mask(episode.length)
            source = TARGET_SOURCE[self.action_source]
        return {
            'target.unified_107d': values,
            'target.unified_107d_mask': mask,
            'target.source': np.full(episode.length, source, dtype=np.int8),
        }
