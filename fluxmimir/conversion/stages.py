from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fluxmimir.embodiments import (AgileXAlohaAdapter, BaseEmbodimentAdapter,
                                   FrankaDualAdapter, OliAdapter,
                                   ReBotDualAdapter)
from fluxmimir.engines.stages import BaseStage
from fluxmimir.engines.utils import STAGES
from fluxmimir.schemas import UNIFIED_DIM
from fluxmimir.structures import PipelineContext, StageResult

from .base import BaseEpisodeMapper
from .structures import EpisodeRecord


def masked_statistics(values: np.ndarray,
                      valid_mask: np.ndarray | None = None) -> dict[str, list]:
    """Compute reference-compatible per-dimension masked statistics."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f'statistics values must be 2D, got {matrix.shape}')
    if valid_mask is None:
        valid = np.ones_like(matrix, dtype=np.bool_)
    else:
        valid = np.asarray(valid_mask, dtype=np.bool_)
        if valid.shape != matrix.shape:
            raise ValueError('statistics values/mask shape mismatch')
    if not np.isfinite(matrix[valid]).all():
        raise ValueError('statistics input contains valid NaN/Inf')

    names = ('min', 'max', 'mean', 'std', 'q01', 'q99')
    results = {
        name: np.zeros(matrix.shape[1], dtype=np.float64)
        for name in names
    }
    counts = valid.sum(axis=0, dtype=np.int64)
    for dimension, count in enumerate(counts):
        if count == 0:
            continue
        selected = matrix[valid[:, dimension], dimension]
        results['min'][dimension] = selected.min()
        results['max'][dimension] = selected.max()
        results['mean'][dimension] = selected.mean()
        results['std'][dimension] = selected.std()
        results['q01'][dimension] = np.quantile(selected, 0.01)
        results['q99'][dimension] = np.quantile(selected, 0.99)
    return {
        **{
            name: results[name].tolist()
            for name in names
        },
        'count': counts.tolist(),
    }


@STAGES.register_module()
class ProbeSourceSchema(BaseStage):

    def __init__(self, scope: str = 'all_episodes') -> None:
        self.scope = scope

    def process(self, episode: Any, context: PipelineContext) -> StageResult:
        del context
        if not isinstance(episode, EpisodeRecord):
            raise TypeError('ProbeSourceSchema expects EpisodeRecord')
        fingerprint = episode.metadata.get('schema_fingerprint')
        if not fingerprint:
            raise ValueError(
                f'{episode.ref.source_uri}: missing schema fingerprint')
        return StageResult(
            value=episode, provenance={'schema_fingerprint': fingerprint})


@STAGES.register_module()
class MapAlohaToCanonical128D(BaseEpisodeMapper):

    def __init__(self,
                 embodiment: BaseEmbodimentAdapter | None = None) -> None:
        self.embodiment = embodiment or AgileXAlohaAdapter()

    def map_episode(self, episode: EpisodeRecord,
                    context: PipelineContext) -> Mapping[str, Any]:
        report = self.embodiment.preflight(episode)
        if not report.ok:
            messages = '; '.join(f'{issue.code}: {issue.message}'
                                 for issue in report.issues)
            raise ValueError(f'{episode.ref.source_uri}: {messages}')

        output = dict(self.embodiment.map_observation(episode))
        output.update(self.embodiment.map_target(episode))
        length = episode.length
        frame_index = np.arange(length, dtype=np.int64)
        if 'timestamp' in episode.fields:
            timestamp = np.asarray(
                episode.fields['timestamp'], dtype=np.float32).reshape(-1)
            source_kind = episode.metadata.get('data_source', 'hdf')
            timestamp_source = f'source_{source_kind}'
        else:
            fps = int(getattr(self.embodiment, 'nominal_fps', 30))
            timestamp = (frame_index / fps).astype(np.float32)
            timestamp_source = f'synthetic_frame_index_over_{fps}_fps'
        if timestamp.shape != (length, ) or not np.isfinite(timestamp).all():
            raise ValueError('invalid timestamps')
        if length > 1 and np.any(np.diff(timestamp) <= 0):
            raise ValueError('timestamps must be strictly increasing')

        episode_index = int(context.metadata.get('episode_index', 0))
        global_start = int(context.metadata.get('global_index_start', 0))
        output.update({
            'timestamp':
            timestamp,
            'frame_index':
            frame_index,
            'episode_index':
            np.full(length, episode_index, dtype=np.int64),
            'index':
            np.arange(global_start, global_start + length, dtype=np.int64),
            'subtask_index':
            np.zeros(length, dtype=np.int64),
            '_provenance': {
                'timestamp_source':
                timestamp_source,
                'schema_fingerprint':
                episode.metadata['schema_fingerprint'],
                'source_relative_path':
                episode.metadata.get('relative_path'),
                'target_semantics':
                getattr(self.embodiment, 'action_semantics', 'unknown'),
            },
        })
        return output

    def process(self, episode: Any, context: PipelineContext) -> StageResult:
        if not isinstance(episode, EpisodeRecord):
            raise TypeError('MapAlohaToCanonical128D expects EpisodeRecord')
        report = self.embodiment.preflight(episode)
        if not report.ok:
            messages = '; '.join(f'{issue.code}: {issue.message}'
                                 for issue in report.issues)
            raise ValueError(f'{episode.ref.source_uri}: {messages}')
        value = self.map_episode(episode, context)
        warnings = tuple(f'{issue.code}: {issue.message}'
                         for issue in report.issues)
        return StageResult(
            value=value, warnings=warnings, provenance=value['_provenance'])


@STAGES.register_module()
class MapReBotToCanonical128D(MapAlohaToCanonical128D):
    """Map ReBot dual-arm records with the source-neutral 128D machinery."""

    def __init__(self,
                 embodiment: BaseEmbodimentAdapter | None = None) -> None:
        super().__init__(embodiment=embodiment or ReBotDualAdapter())


@STAGES.register_module()
class MapFrankaToCanonical128D(MapAlohaToCanonical128D):
    """Map dual Franka records with the source-neutral 128D machinery."""

    def __init__(self,
                 embodiment: BaseEmbodimentAdapter | None = None) -> None:
        super().__init__(embodiment=embodiment or FrankaDualAdapter())


@STAGES.register_module()
class MapOliToCanonical128D(MapAlohaToCanonical128D):
    """Map LimX Oli records with the source-neutral 128D machinery."""

    def __init__(self,
                 embodiment: BaseEmbodimentAdapter | None = None) -> None:
        super().__init__(embodiment=embodiment or OliAdapter())


@STAGES.register_module()
class EncodeEpisodeVideos(BaseStage):

    def __init__(self,
                 codec: str = 'h264',
                 fps: int = 30,
                 frame_count_must_match: bool = True,
                 enabled: bool = True,
                 crf: int = 23,
                 gop_size: int = 30,
                 mode: str = 'encode') -> None:
        if mode not in {'encode', 'copy'}:
            raise ValueError(f'unsupported video mode: {mode}')
        self.spec = {
            'enabled': bool(enabled),
            'codec': codec,
            'fps': int(fps),
            'frame_count_must_match': bool(frame_count_must_match),
            'crf': int(crf),
            'gop_size': int(gop_size),
            'mode': mode,
        }

    def process(self, episode: Any, context: PipelineContext) -> StageResult:
        del context
        if not isinstance(episode, dict):
            raise TypeError('EncodeEpisodeVideos expects mapped dictionary')
        value = dict(episode)
        value['_video_encoding'] = dict(self.spec)
        return StageResult(value=value)


@STAGES.register_module()
class ComputeMaskedEpisodeStatistics(BaseStage):

    def process(self, episode: Any, context: PipelineContext) -> StageResult:
        del context
        if not isinstance(episode, dict):
            raise TypeError('statistics stage expects mapped dictionary')
        value = dict(episode)
        for name in ('observation.unified_128d', 'target.unified_128d'):
            if np.asarray(value[name]).shape[1] != UNIFIED_DIM:
                raise ValueError(
                    f'{name} does not have {UNIFIED_DIM} dimensions')
        value['_statistics'] = {
            'observation.unified_128d':
            masked_statistics(value['observation.unified_128d'],
                              value['observation.unified_128d_mask']),
            'target.unified_128d':
            masked_statistics(value['target.unified_128d'],
                              value['target.unified_128d_mask']),
        }
        return StageResult(value=value)


@STAGES.register_module()
class BuildCanonicalMetadata(BaseStage):

    def process(self, episode: Any, context: PipelineContext) -> StageResult:
        del context
        if not isinstance(episode, dict):
            raise TypeError('metadata stage expects mapped dictionary')
        return StageResult(value=episode)
