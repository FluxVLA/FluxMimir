from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np

from fluxmimir.embodiments import BaseEmbodimentAdapter
from fluxmimir.engines.stages import BaseStage
from fluxmimir.engines.utils import STAGES
from fluxmimir.schemas import PARQUET_COLUMNS
from fluxmimir.structures import PipelineContext, StageResult

from .criteria import StaticFrameCriteria
from .detection import DETECTOR_VERSION, static_frame_mask, static_segments


@STAGES.register_module()
class FilterStaticFrames(BaseStage):
    """Drop episodes with a long static segment and audit each decision."""

    def __init__(self, embodiment: BaseEmbodimentAdapter, *, enabled: bool,
                 policy: str, static_frame_threshold: int,
                 signals: tuple[str, ...], groups: tuple[str, ...],
                 step_threshold: Mapping[str, float], audit_name: str) -> None:
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be a bool')
        if policy not in ('report', 'drop_episode'):
            raise ValueError(f'unsupported policy: {policy}')
        if not signals:
            raise ValueError('signals must not be empty')
        if len(set(signals)) != len(signals):
            raise ValueError(f'duplicate signals: {signals}')
        columns = tuple(f'{signal}.unified_107d' for signal in signals)
        unknown = sorted(signal for signal, column in zip(signals, columns)
                         if column not in PARQUET_COLUMNS
                         or f'{column}_mask' not in PARQUET_COLUMNS)
        if unknown:
            raise ValueError(f'unsupported signals: {unknown}')
        if not isinstance(audit_name, str) or not audit_name:
            raise ValueError('audit_name must not be empty')
        self.embodiment = embodiment
        self.enabled = enabled
        self.policy = policy
        self.columns = columns
        self.audit_name = audit_name
        self.criteria = StaticFrameCriteria(
            step_threshold=step_threshold,
            groups=groups,
            static_frame_threshold=static_frame_threshold)
        self.audit_stream: TextIO | None = None

    def setup(self, context: PipelineContext) -> None:
        if not self.enabled:
            return
        audit_path = Path(context.work_dir) / self.audit_name
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_stream = audit_path.open('a', encoding='utf-8')
        self._write({
            'record': 'header',
            'run_id': context.run_id,
            'detector_version': DETECTOR_VERSION,
            'policy': self.policy,
            'signals': list(self.columns),
            'schema_id': self.embodiment.schema_id,
            'layout_id': self.embodiment.layout_id,
            'criteria': self.criteria.record(),
            'criteria_sha256': self.criteria.fingerprint(),
        })

    def process(self, value: Any, context: PipelineContext) -> StageResult:
        if not self.enabled:
            return StageResult(value=value)
        if not isinstance(value, Mapping):
            raise TypeError('FilterStaticFrames expects a mapped dictionary')
        if '_statistics' in value:
            raise ValueError('FilterStaticFrames must run before '
                             'ComputeMaskedEpisodeStatistics')
        length = len(np.asarray(value['timestamp']))
        for column in self.columns:
            if len(np.asarray(value[column])) != length:
                raise ValueError(f'{column} and timestamp lengths differ')
        static = self._static_frames(value)
        segments = static_segments(static)
        longest = max((end - start + 1 for start, end in segments), default=0)
        hit = longest >= self.criteria.static_frame_threshold
        dropped = hit and self.policy == 'drop_episode'
        record = self._record(value, context, static, segments, longest, hit,
                              dropped)
        self._write(record)
        warnings: tuple[str, ...] = ()
        if dropped:
            warnings = (f'static frame filter dropped episode: longest '
                        f'static segment {longest} >= '
                        f'{self.criteria.static_frame_threshold}', )
        return StageResult(
            value=value,
            metrics={
                'static_frames': float(record['static_frames']),
                'static_ratio': float(record['static_ratio']),
                'max_static_window': float(record['max_static_window']),
                'segment_count': float(len(segments)),
            },
            warnings=warnings,
            provenance={
                'static_frame_filter': {
                    'policy': self.policy,
                    'schema_id': self.embodiment.schema_id,
                    'criteria_sha256': self.criteria.fingerprint(),
                    'detector_version': DETECTOR_VERSION,
                }
            },
            skipped=dropped,
        )

    def close(self) -> None:
        if self.audit_stream is not None:
            self.audit_stream.close()
            self.audit_stream = None

    def _static_frames(self, value: Mapping[str, Any]) -> np.ndarray:
        thresholds = self.criteria.thresholds()
        masks = []
        for column in self.columns:
            valid = np.asarray(value[f'{column}_mask'], dtype=np.bool_)
            if not valid.any():
                continue
            masks.append(
                static_frame_mask(
                    np.asarray(value[column], dtype=np.float64), valid,
                    thresholds))
        if not masks:
            raise ValueError('no configured signal has a valid dimension')
        return np.logical_and.reduce(masks)

    def _record(self, value: Mapping[str, Any], context: PipelineContext,
                static: np.ndarray, segments: tuple[tuple[int, int], ...],
                longest: int, hit: bool, dropped: bool) -> dict[str, Any]:
        timestamps = np.asarray(value['timestamp'], dtype=np.float64)
        total = int(static.sum())
        episode_index = (None if dropped else int(
            context.metadata['episode_index']))
        return {
            'record':
            'episode',
            'run_id':
            context.run_id,
            'source_episode_identity':
            value.get('_provenance', {}).get('source_relative_path'),
            'episode_index':
            episode_index,
            'source_length':
            int(len(timestamps)),
            'decision':
            'dropped' if dropped else 'kept',
            'static_hit':
            bool(hit),
            'static_frames':
            total,
            'static_ratio':
            float(total / len(static)) if len(static) else 0.0,
            'max_static_window':
            int(longest),
            'segments':
            [{
                'start': start,
                'end': end,
                'frames': end - start + 1,
                'seconds': float(timestamps[end] - timestamps[start]),
            } for start, end in segments
             if end - start + 1 >= self.criteria.static_frame_threshold],
        }

    def _write(self, record: Mapping[str, Any]) -> None:
        assert self.audit_stream is not None
        self.audit_stream.write(json.dumps(record, ensure_ascii=False) + '\n')
