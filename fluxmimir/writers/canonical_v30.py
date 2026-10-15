from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fluxmimir.conversion import EpisodeRecord
from fluxmimir.engines.utils import WRITERS
from fluxmimir.structures import ArtifactRecord, StageResult

from .canonical import (CanonicalDatasetWriter, _fixed_tensor,
                        _require_pyarrow, _write_json, _write_jsonl)

CODEBASE_VERSION = 'v3.0'
DATA_FILES_SIZE_IN_MB = 100
VIDEO_FILES_SIZE_IN_MB = 200
EPISODE_STATISTICS = ('min', 'max', 'mean', 'std', 'q01', 'q99')


def _floats(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


@WRITERS.register_module()
class CanonicalDatasetV30Writer(CanonicalDatasetWriter):
    """Write the canonical content with the LeRobot v3.0 file layout.

    Column content, masks, statistics and publish conventions are inherited
    from ``CanonicalDatasetWriter``; only paths and metadata forms change:
    one episode per ``file-XXX`` Parquet, video and episode metadata keyed by
    chunk, tasks as Parquet and one aggregate ``meta/stats.json``.
    """

    def _chunk_ordinal(self, episode_index: int) -> tuple[int, int]:
        return divmod(episode_index, self.chunks_size)

    def _data_path(self, episode_index: int) -> Path:
        chunk, ordinal = self._chunk_ordinal(episode_index)
        return (self.root / f'data/chunk-{chunk:03d}' /
                f'file-{ordinal:03d}.parquet')

    def _video_path(self, episode_index: int, camera: str) -> Path:
        chunk, ordinal = self._chunk_ordinal(episode_index)
        key = f'observation.images.{camera}'
        return (self.root / f'videos/{key}/chunk-{chunk:03d}' /
                f'file-{ordinal:03d}.mp4')

    def _episodes_path(self, episode_index: int) -> Path:
        chunk, ordinal = self._chunk_ordinal(episode_index)
        return (self.root / f'meta/episodes/chunk-{chunk:03d}' /
                f'file-{ordinal:03d}.parquet')

    def write_episode(self, episode: EpisodeRecord,
                      result: StageResult) -> tuple[ArtifactRecord, ...]:
        artifacts = super().write_episode(episode, result)
        timestamps = np.asarray(result.value['timestamp'], dtype=np.float64)
        self.episodes[-1]['timestamp_span'] = (float(timestamps[0]),
                                               float(timestamps[-1]))
        return artifacts

    def _build_info(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        info = super()._build_info(metadata)
        info['codebase_version'] = CODEBASE_VERSION
        info['data_files_size_in_mb'] = DATA_FILES_SIZE_IN_MB
        info['video_files_size_in_mb'] = VIDEO_FILES_SIZE_IN_MB
        info.pop('total_chunks', None)
        info.pop('total_videos', None)
        info['data_path'] = ('data/chunk-{chunk_index:03d}/'
                             'file-{file_index:03d}.parquet')
        info['video_path'] = ('videos/{video_key}/chunk-{chunk_index:03d}/'
                              'file-{file_index:03d}.mp4')
        info['annotation_contract'] = {
            'task_definitions_location': 'meta/tasks.parquet',
            'episode_task_assignment': ('meta/episodes/**/*.parquet'
                                        '[*].tasks'),
            'subtask_definitions_location': 'meta/subtasks.jsonl',
            'frame_subtask_assignment': 'data/**/*.parquet[*].subtask_index',
        }
        return info

    def _write_metadata(self, info: dict[str, Any], version: dict[str, Any],
                        tasks: list[dict[str, Any]],
                        subtasks: list[dict[str, Any]]) -> None:
        task_text = {row['task_index']: row['task'] for row in tasks}
        statistics = {
            record['episode_index']: record['stats']
            for record in self.episode_stats
        }
        starts: dict[int, int] = {}
        offset = 0
        for record in self.episodes:
            starts[record['episode_index']] = offset
            offset += record['length']
        _write_json(self.root / 'meta/info.json', info)
        _write_json(self.root / 'meta/fluxvla_dataset_version.json', version)
        _write_json(self.root / 'meta/stats.json', self._aggregate_stats())
        _write_jsonl(self.root / 'meta/subtasks.jsonl', subtasks)
        self._write_table(self.root / 'meta/tasks.parquet',
                          self._tasks_table(tasks))
        for record in self.episodes:
            episode_index = record['episode_index']
            self._write_table(
                self._episodes_path(episode_index),
                self._episodes_table(record, task_text, starts[episode_index],
                                     statistics[episode_index]))

    @staticmethod
    def _write_table(path: Path, table: Any) -> None:
        _, pq = _require_pyarrow()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            pq.write_table(table, temporary, compression='zstd')
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _tasks_table(tasks: list[dict[str, Any]]) -> Any:
        pa, _ = _require_pyarrow()
        provenance = pa.struct([('source', pa.string())])
        return pa.table({
            'task_index':
            pa.array([row['task_index'] for row in tasks], type=pa.int64()),
            'task_id':
            pa.array([row['task_id'] for row in tasks], type=pa.string()),
            'task':
            pa.array([row['task'] for row in tasks], type=pa.string()),
            'language':
            pa.array([row['language'] for row in tasks], type=pa.string()),
            'placeholder_status':
            pa.array([row['placeholder_status'] for row in tasks],
                     type=pa.string()),
            'annotation_provenance':
            pa.array([row['annotation_provenance'] for row in tasks],
                     type=provenance),
        })

    def _episodes_table(self, record: Mapping[str, Any],
                        task_text: Mapping[int, str], from_index: int,
                        statistics: Mapping[str, Any]) -> Any:
        pa, _ = _require_pyarrow()
        episode_index = record['episode_index']
        chunk, ordinal = self._chunk_ordinal(episode_index)
        from_timestamp, to_timestamp = record['timestamp_span']
        columns: dict[str, Any] = {
            'episode_index':
            pa.array([episode_index], type=pa.int64()),
            'tasks':
            pa.array([[task_text[record['task_index']]]],
                     type=pa.list_(pa.string())),
            'length':
            pa.array([record['length']], type=pa.int64()),
            'data/chunk_index':
            pa.array([chunk], type=pa.int64()),
            'data/file_index':
            pa.array([ordinal], type=pa.int64()),
            'dataset_from_index':
            pa.array([from_index], type=pa.int64()),
            'dataset_to_index':
            pa.array([from_index + record['length']], type=pa.int64()),
        }
        for key in sorted(self.video_features):
            columns[f'videos/{key}/chunk_index'] = pa.array([chunk],
                                                            type=pa.int64())
            columns[f'videos/{key}/file_index'] = pa.array([ordinal],
                                                           type=pa.int64())
            columns[f'videos/{key}/from_timestamp'] = pa.array(
                [from_timestamp], type=pa.float32())
            columns[f'videos/{key}/to_timestamp'] = pa.array([to_timestamp],
                                                             type=pa.float32())
        for name, values in statistics.items():
            for statistic in EPISODE_STATISTICS:
                columns[f'stats/{name}/{statistic}'] = _fixed_tensor(
                    np.asarray([values[statistic]], dtype=np.float32),
                    pa.float32())
            columns[f'stats/{name}/count'] = _fixed_tensor(
                np.asarray([values['count']], dtype=np.int64), pa.int64())
        return pa.table(columns)

    def _aggregate_stats(self) -> dict[str, Any]:
        accumulators: dict[str, dict[str, np.ndarray]] = {}
        for record in self.episode_stats:
            for name, values in record['stats'].items():
                count = _floats(values['count'])
                mean = _floats(values['mean'])
                moment = count * (_floats(values['std'])**2 + mean**2)
                entry = accumulators.get(name)
                if entry is None:
                    accumulators[name] = {
                        'count': count,
                        'mean': mean,
                        'moment': moment,
                        'min': _floats(values['min']),
                        'max': _floats(values['max']),
                    }
                    continue
                total = entry['count'] + count
                scale = np.where(total > 0, total, 1.0)
                entry['mean'] = np.where(
                    total > 0,
                    (entry['count'] * entry['mean'] + count * mean) / scale,
                    0.0)
                entry['moment'] = np.where(total > 0,
                                           (entry['moment'] + moment) / scale,
                                           0.0)
                entry['min'] = np.minimum(entry['min'], _floats(values['min']))
                entry['max'] = np.maximum(entry['max'], _floats(values['max']))
                entry['count'] = total
        return {
            name: {
                'min':
                entry['min'].tolist(),
                'max':
                entry['max'].tolist(),
                'mean':
                entry['mean'].tolist(),
                'std':
                np.sqrt(np.maximum(entry['moment'] - entry['mean']**2,
                                   0.0)).tolist(),
                'count': [int(value) for value in entry['count']],
            }
            for name, entry in accumulators.items()
        }
