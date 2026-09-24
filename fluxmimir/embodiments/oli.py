from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fluxmimir.conversion.structures import EpisodeRecord
from fluxmimir.engines.utils import EMBODIMENTS
from fluxmimir.schemas import (OLI_ACTION_DIM, OLI_CAMERA_ORDER, OLI_LAYOUT_ID,
                               OLI_SCHEMA_ID, OLI_SCHEMA_VERSION,
                               OLI_STATE_DIM, OLI_STATE_JOINT_NAMES,
                               TARGET_SOURCE, UNIFIED_DIM)
from fluxmimir.structures import (IssueSeverity, ValidationIssue,
                                  ValidationReport)

from .base import BaseEmbodimentAdapter

OLI_EMBODIMENT_ID = 'oli'
OLI_EMBODIMENT_DESCRIPTION = 'LimX Oli humanoid'


@dataclass(frozen=True)
class SourceBlock:
    """One named unified range and its source state/action slices."""

    name: str
    unified: tuple[int, int]
    state: tuple[int, int] | None = None
    action: tuple[int, int] | None = None


OLI_BLOCKS = (
    SourceBlock('arms', (0, 14), state=(17, 31), action=(17, 31)),
    SourceBlock('gripper', (28, 30), state=(31, 33), action=(40, 42)),
    SourceBlock('waist', (94, 97), state=(12, 15), action=(12, 15)),
    SourceBlock('head', (98, 100), state=(15, 17), action=(15, 17)),
    SourceBlock('base_position', (100, 103), action=(31, 34)),
    SourceBlock('left_leg', (103, 109), state=(0, 6), action=(0, 6)),
    SourceBlock('right_leg', (109, 115), state=(6, 12), action=(6, 12)),
    SourceBlock('base_rotation_6d', (115, 121), action=(34, 40)),
)


def _mapped(source: np.ndarray, length: int, attribute: str,
            missing_value: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((length, UNIFIED_DIM), missing_value, dtype=np.float32)
    mask = np.zeros_like(values, dtype=np.bool_)
    for block in OLI_BLOCKS:
        source_range = getattr(block, attribute)
        if source_range is None:
            continue
        start, end = block.unified
        low, high = source_range
        if end - start != high - low:
            raise ValueError(f'{block.name}: block width mismatch')
        values[:, start:end] = source[:, low:high]
        mask[:, start:end] = True
    return values, mask


@EMBODIMENTS.register_module()
class OliAdapter(BaseEmbodimentAdapter):
    """Map LimX Oli feedback into the canonical 128D layout.

    ``action_semantics`` defaults to ``unknown``: the source column is
    named ``joint_cmd_*``, but issued target provenance must be declared
    explicitly by the conversion config.
    """

    schema_id = OLI_SCHEMA_ID
    schema_version = OLI_SCHEMA_VERSION
    layout_id = OLI_LAYOUT_ID
    embodiment_id = OLI_EMBODIMENT_ID
    embodiment_description = OLI_EMBODIMENT_DESCRIPTION
    robot_type = 'oli'
    camera_order = OLI_CAMERA_ORDER
    gripper_unit = 'm'
    layout_profile = 'oli'
    active_ranges = ((0, 14), (28, 30), (94, 98), (98, 100), (100, 103),
                     (103, 115), (115, 121))
    qpose_reconstruction = ('unified[0:14] + unified[28:30] + '
                            'unified[94:100] + unified[103:121]')
    source_format = 'LeRobot v2.1 Oli episodes'
    conversion_description = ('streaming LeRobot v2.1 to canonical '
                              'unified 128D')
    video_materialization = 'copied_source_mp4'
    training_readiness_blockers = (
        'issued target provenance is unconfirmed by the source',
        'calibrated ego2cam and wrist calibration are unavailable',
        'frame-level subtasks are unavailable',
    )
    unavailable_annotation_source = 'unavailable_in_lerobot_v2'

    def __init__(
        self,
        action_semantics: str = 'unknown',
        action_source: str = 'operator',
        nominal_fps: int = 30,
        missing_dense_value: float = 0.0,
    ) -> None:
        if action_semantics not in ('unknown', 'issued_joint_position_target'):
            raise ValueError(
                f'unsupported action semantics: {action_semantics}')
        if action_source not in TARGET_SOURCE:
            raise ValueError(f'unknown target source: {action_source}')
        if nominal_fps <= 0:
            raise ValueError('nominal_fps must be positive')
        self.action_semantics = action_semantics
        self.action_source = action_source
        self.nominal_fps = int(nominal_fps)
        self.missing_dense_value = float(missing_dense_value)

    def preflight(self, episode: EpisodeRecord) -> ValidationReport:
        issues: list[ValidationIssue] = []
        location = episode.ref.source_uri

        def add(code: str,
                message: str,
                severity: IssueSeverity = IssueSeverity.ERROR) -> None:
            issues.append(
                ValidationIssue(
                    code, message, severity=severity, location=location))

        state = np.asarray(episode.fields.get('observations/qpos'))
        expected_state = (episode.length, OLI_STATE_DIM)
        if state.shape != expected_state:
            add('invalid_state_shape', f'state shape is {state.shape}')
        elif not np.isfinite(state).all():
            add('nonfinite_state', 'state contains NaN/Inf')
        action = episode.fields.get('action')
        if action is not None:
            action = np.asarray(action)
            expected_action = (episode.length, OLI_ACTION_DIM)
            if action.shape != expected_action:
                add('invalid_action_shape', f'action shape is {action.shape}')
            elif not np.isfinite(action).all():
                add('nonfinite_action', 'action contains NaN/Inf')
        names = tuple(episode.metadata.get('joint_names', ()))
        if names != OLI_STATE_JOINT_NAMES:
            add('joint_order', 'state joint order differs')
        missing = sorted(set(self.camera_order) - set(episode.media))
        if missing:
            add('missing_cameras', f'missing cameras: {missing}')
        if self.action_semantics == 'unknown' and action is not None:
            add(
                'target_semantics_unknown',
                'raw /action is provenance only; target remains an '
                'invalid placeholder', IssueSeverity.WARNING)
        if 'timestamp' not in episode.fields:
            add('synthetic_timestamp', f'using frame_index/{self.nominal_fps}',
                IssueSeverity.WARNING)
        return ValidationReport(issues=tuple(issues))

    def map_observation(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        state = np.asarray(
            episode.fields['observations/qpos'], dtype=np.float32)
        values, mask = _mapped(state, episode.length, 'state',
                               self.missing_dense_value)
        return {
            'observation.unified_128d':
            values,
            'observation.unified_128d_mask':
            mask,
            'observation.ego2cam':
            np.full((episode.length, len(self.camera_order), 4, 4),
                    self.missing_dense_value,
                    dtype=np.float32),
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
            if action.shape != (episode.length, OLI_ACTION_DIM):
                raise ValueError(f'bad action shape: {action.shape}')
            if not np.isfinite(action).all():
                raise ValueError('action contains NaN/Inf')
            values, mask = _mapped(action, episode.length, 'action',
                                   self.missing_dense_value)
            source = TARGET_SOURCE[self.action_source]
        return {
            'target.unified_128d': values,
            'target.unified_128d_mask': mask,
            'target.source': np.full(episode.length, source, dtype=np.int8),
        }
