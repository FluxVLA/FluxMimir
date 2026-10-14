from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from fluxmimir.conversion import EpisodeRecord
from fluxmimir.engines.utils import SOURCES
from fluxmimir.structures import EpisodeRef

from .base import BaseSource


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('LeRobot v2 conversion requires pyarrow') from exc
    return pq


def _flatten_names(names):
    names = tuple(names or ())
    if names and isinstance(names[0], (list, tuple)):
        return tuple(names[0])
    return names


class MP4FrameReader:
    """Lazy RGB frame reader for one LeRobot v2 MP4 file."""

    def __init__(self, path: str | Path, *, expected_length: int) -> None:
        try:
            import av
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('MP4 conversion requires PyAV') from exc
        self.path = Path(path).resolve()
        self.length = int(expected_length)
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            self.actual_frame_count = int(stream.frames or 0)
            self.height = int(stream.height)
            self.width = int(stream.width)
            codec = stream.codec_context.codec
            self.codec = str(getattr(codec, 'canonical_name', codec.name))
        self.shape = (self.length, self.height, self.width, 3)

    def __len__(self) -> int:
        return self.length

    def iter_frames(self) -> Iterator[np.ndarray]:
        import av
        with av.open(str(self.path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format='rgb24')

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frames()


@SOURCES.register_module()
class LeRobotV2Source(BaseSource):
    """Read one-episode-per-file LeRobot v2 Parquet/MP4 datasets."""

    def __init__(
        self,
        root: str | Path,
        camera_mapping: Mapping[str, str],
        state_key: str = 'observation.state',
        action_key: str = 'action',
        eepose_key: str | None = 'observation.eepose',
        max_episodes: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.camera_mapping = dict(camera_mapping)
        self.state_key = state_key
        self.action_key = action_key
        self.eepose_key = eepose_key
        if max_episodes is not None and max_episodes < 1:
            raise ValueError('max_episodes must be positive')
        self.max_episodes = max_episodes
        info_path = self.root / 'meta' / 'info.json'
        if not info_path.is_file():
            raise FileNotFoundError('LeRobot v2 info.json is missing')
        with info_path.open(encoding='utf-8') as stream:
            self.info = json.load(stream)
        if self.info.get('codebase_version') not in ('v2.0', 'v2.1'):
            raise ValueError(
                'LeRobotV2Source requires codebase_version v2.0/v2.1')
        self.tasks = self._load_tasks()
        self.placeholder_task_index = max(self.tasks) + 1
        self.tasks[self.placeholder_task_index] = 'unknown episode-level task'

    def _load_tasks(self) -> dict[int, str]:
        path = self.root / 'meta' / 'tasks.jsonl'
        if not path.is_file():
            raise FileNotFoundError('LeRobot v2 tasks.jsonl is missing')
        tasks: dict[int, str] = {}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    tasks[int(row['task_index'])] = str(row['task'])
        return tasks

    def _paths(self) -> list[Path]:
        paths = sorted(self.root.glob('data/chunk-*/episode_*.parquet'))
        if self.max_episodes is not None:
            paths = paths[:self.max_episodes]
        if not paths:
            raise FileNotFoundError(f'no LeRobot v2 Parquet under {self.root}')
        return paths

    def _video_path(self, parquet_path: Path, video_key: str) -> Path:
        chunk = parquet_path.parent.name
        return self.root / 'videos' / chunk / video_key / (
            parquet_path.stem + '.mp4')

    def discover(self) -> Iterator[EpisodeRef]:
        for path in self._paths():
            relative = path.relative_to(self.root).as_posix()
            identity = f'{self.root.name}:{relative}'
            digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
            yield EpisodeRef(
                episode_uid=f'lerobot-v2-{digest}',
                source_uri=str(path),
                source_episode_index=int(path.stem.split('_')[-1]),
                metadata={
                    'relative_path': relative,
                    'bytes': path.stat().st_size,
                    'data_source': 'lerobot_v2',
                })

    def _episode_task(self, frame_tasks: list[int]) -> int:
        unique = sorted(set(frame_tasks))
        if len(unique) == 1:
            return unique[0]
        return self.placeholder_task_index

    def open_episode(self, episode: EpisodeRef) -> EpisodeRecord:
        pq = _require_pyarrow()
        path = Path(episode.source_uri)
        source_schema = pq.read_schema(path)
        available = set(source_schema.names)
        if self.state_key not in available:
            raise ValueError(f'{path}: missing {self.state_key}')
        columns = ['timestamp', 'task_index']
        for key in ('episode_index', self.state_key, self.action_key,
                    self.eepose_key):
            if key and key in available and key not in columns:
                columns.append(key)
        table = pq.read_table(path, columns=columns)
        length = table.num_rows
        state = np.asarray(table[self.state_key].to_pylist(), dtype=np.float32)

        fields: dict[str, np.ndarray] = {'observations/qpos': state}
        for field, key in (('action', self.action_key), ('observations/eepose',
                                                         self.eepose_key)):
            if key and key in available:
                fields[field] = np.asarray(
                    table[key].to_pylist(), dtype=np.float32)
        fields['timestamp'] = np.asarray(
            table['timestamp'].to_numpy(), dtype=np.float32)

        frame_tasks = [int(value) for value in table['task_index'].to_pylist()]
        unknown = sorted(set(frame_tasks) - set(self.tasks))
        if unknown:
            raise ValueError(f'{path}: unknown task_index {unknown}')
        task_index = self._episode_task(frame_tasks)
        if 'episode_index' in available:
            stored = set(
                int(value) for value in table['episode_index'].to_pylist())
            if stored != {episode.source_episode_index}:
                raise ValueError(f'{path}: episode_index mismatch')

        media: dict[str, MP4FrameReader] = {}
        for canonical_camera, source_key in self.camera_mapping.items():
            video_path = self._video_path(path, source_key)
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            media[canonical_camera] = MP4FrameReader(
                video_path, expected_length=length)

        features = self.info.get('features', {})
        joint_names = _flatten_names(
            features.get(self.state_key, {}).get('names'))
        schema_payload = {
            'schema': str(source_schema.remove_metadata()),
            'joint_names': joint_names,
            'camera_mapping': self.camera_mapping,
            'source_codebase_version': self.info.get('codebase_version'),
        }
        fingerprint = hashlib.sha256(
            json.dumps(schema_payload, sort_keys=True,
                       separators=(',', ':')).encode()).hexdigest()
        metadata = dict(episode.metadata)
        metadata.update({
            'schema_fingerprint': fingerprint,
            'source_schema': schema_payload,
            'joint_names': joint_names,
            'task_index': task_index,
            'task': self.tasks[task_index],
            'available_cameras': list(media),
        })
        return EpisodeRecord(
            ref=episode,
            length=length,
            fields=fields,
            media=media,
            metadata=metadata)
