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
        raise RuntimeError('LeRobot v3 conversion requires pyarrow') from exc
    return pq


class MP4FrameReader:
    """Sequential RGB reader plus immutable source-path metadata."""

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

    def read_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.length:
            raise IndexError(index)
        for frame_index, frame in enumerate(self.iter_frames()):
            if frame_index == index:
                return frame
        raise IndexError(index)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frames()


@SOURCES.register_module()
class LeRobotV3Source(BaseSource):
    """Read one-episode-per-file LeRobot v3 Parquet/MP4 datasets."""

    def __init__(
        self,
        root: str | Path,
        camera_mapping: Mapping[str, str],
        state_key: str = 'observation.state',
        action_key: str = 'action',
        max_episodes: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.camera_mapping = dict(camera_mapping)
        self.state_key = state_key
        self.action_key = action_key
        if max_episodes is not None and max_episodes < 1:
            raise ValueError('max_episodes must be positive')
        self.max_episodes = max_episodes
        info_path = self.root / 'meta/info.json'
        tasks_path = self.root / 'meta/tasks.parquet'
        if not info_path.is_file() or not tasks_path.is_file():
            raise FileNotFoundError(
                'LeRobot v3 info/tasks metadata is missing')
        self.info = json.loads(info_path.read_text(encoding='utf-8'))
        if self.info.get('codebase_version') != 'v3.0':
            raise ValueError('LeRobotV3Source requires codebase_version=v3.0')
        pq = _require_pyarrow()
        task_rows = pq.read_table(tasks_path).to_pylist()
        self.tasks = {
            int(row['task_index']): str(row['task'])
            for row in task_rows
        }

    def _paths(self) -> list[Path]:
        paths = sorted((self.root / 'data').glob('chunk-*/file-*.parquet'))
        if self.max_episodes is not None:
            paths = paths[:self.max_episodes]
        if not paths:
            raise FileNotFoundError(f'no LeRobot v3 Parquet under {self.root}')
        return paths

    def discover(self) -> Iterator[EpisodeRef]:
        pq = _require_pyarrow()
        for path in self._paths():
            episode_values = pq.read_table(
                path, columns=['episode_index'])['episode_index'].to_pylist()
            unique = sorted(set(int(value) for value in episode_values))
            if len(unique) != 1:
                raise ValueError(f'{path}: expected exactly one episode')
            source_index = unique[0]
            relative = path.relative_to(self.root).as_posix()
            identity = f'{self.root.name}:{source_index}:{relative}'
            digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
            yield EpisodeRef(
                episode_uid=f'lerobot-v3-{digest}',
                source_uri=str(path),
                source_episode_index=source_index,
                metadata={
                    'relative_path': relative,
                    'bytes': path.stat().st_size,
                    'data_source': 'lerobot_v3',
                })

    def _video_path(self, parquet_path: Path, video_key: str) -> Path:
        chunk = parquet_path.parent.name
        return self.root / 'videos' / video_key / chunk / (
            parquet_path.stem + '.mp4')

    def open_episode(self, episode: EpisodeRef) -> EpisodeRecord:
        pq = _require_pyarrow()
        path = Path(episode.source_uri)
        source_schema = pq.read_schema(path)
        columns = [self.state_key, self.action_key, 'timestamp', 'task_index']
        table = pq.read_table(path, columns=columns)
        length = table.num_rows
        state = np.asarray(table[self.state_key].to_pylist(), dtype=np.float32)
        action = np.asarray(
            table[self.action_key].to_pylist(), dtype=np.float32)
        timestamp = np.asarray(table['timestamp'].to_numpy(), dtype=np.float32)
        task_values = set(
            int(value) for value in table['task_index'].to_pylist())
        if len(task_values) != 1:
            raise ValueError(f'{path}: mixed task_index values')
        task_index = task_values.pop()
        if task_index not in self.tasks:
            raise ValueError(f'{path}: unknown task_index {task_index}')

        features = self.info.get('features', {})
        state_names = tuple(features.get(self.state_key, {}).get('names', ()))
        action_names = tuple(
            features.get(self.action_key, {}).get('names', ()))
        if state_names != action_names:
            raise ValueError(f'{path}: state/action names differ')

        media: dict[str, MP4FrameReader] = {}
        for canonical_camera, source_key in self.camera_mapping.items():
            video_path = self._video_path(path, source_key)
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            reader = MP4FrameReader(video_path, expected_length=length)
            media[canonical_camera] = reader

        schema_payload = {
            'schema': str(source_schema.remove_metadata()),
            'joint_names': state_names,
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
            'joint_names': state_names,
            'task_index': task_index,
            'task': self.tasks[task_index],
            'available_cameras': list(media),
        })
        return EpisodeRecord(
            ref=episode,
            length=length,
            fields={
                'observations/qpos': state,
                'action': action,
                'timestamp': timestamp,
            },
            media=media,
            metadata=metadata)
