from __future__ import annotations

import hashlib
import json
import re
from abc import ABC
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from fluxmimir.conversion import EpisodeRecord
from fluxmimir.engines.utils import SOURCES
from fluxmimir.structures import EpisodeRef

from .base import BaseSource


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError('HDF conversion requires the `conversion` extra: '
                           '`pip install -e .[conversion]`') from exc
    return h5py


def _natural_key(path: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    for token in re.split(r'(\d+)', path.as_posix().lower()):
        parts.append(int(token) if token.isdigit() else token)
    return tuple(parts)


class HDF5FrameReader:
    """Small lazy reader for one HDF image dataset.

    The HDF file is opened only while a frame is read or iterated, so source
    discovery and robot-state conversion never materialize a complete video.
    """

    def __init__(self, path: str | Path, dataset: str, *, length: int,
                 shape: tuple[int, ...], dtype: str) -> None:
        self.path = Path(path)
        self.dataset = dataset
        self.length = int(length)
        self.shape = tuple(int(item) for item in shape)
        self.dtype = dtype

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _decode(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 3:
            return np.ascontiguousarray(array, dtype=np.uint8)

        # Compressed ALOHA HDF files store one encoded image as a byte vector,
        # sometimes padded with trailing zeros to a fixed width.
        encoded = np.asarray(array, dtype=np.uint8).reshape(-1)
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional data variant
            raise RuntimeError('Decoding compressed HDF images requires '
                               '`opencv-python-headless`') from exc
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            # A padded JPEG normally decodes as-is. Trim only as a fallback.
            nonzero = np.flatnonzero(encoded)
            if len(nonzero):
                frame = cv2.imdecode(encoded[:nonzero[-1] + 1],
                                     cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('failed to decode an HDF image frame')
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def read_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.length:
            raise IndexError(index)
        h5py = _require_h5py()
        with h5py.File(self.path, 'r') as stream:
            return self._decode(stream[self.dataset][index])

    def iter_frames(self) -> Iterator[np.ndarray]:
        h5py = _require_h5py()
        with h5py.File(self.path, 'r') as stream:
            dataset = stream[self.dataset]
            for index in range(self.length):
                yield self._decode(dataset[index])

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frames()


class BaseHDF5Source(BaseSource, ABC):
    """Extension point for HDF5 layouts."""


@SOURCES.register_module()
class AlohaHDF5Source(BaseHDF5Source):
    """Discover and lazily decode AgileX ALOHA episode HDF files."""

    _TIMESTAMP_KEYS = ('timestamp', 'timestamps', 'observations/timestamp',
                       'observations/timestamps')

    def __init__(
        self,
        root: str | Path,
        pattern: str = '**/episode_*.hdf5',
        streaming: bool = True,
        cameras: list[str] | tuple[str, ...] = ('cam_high', 'cam_left_wrist',
                                                'cam_right_wrist'),
        depth_group: str | None = 'observations/images_depth',
        max_episodes: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.pattern = pattern
        self.streaming = bool(streaming)
        self.cameras = tuple(cameras)
        self.depth_group = depth_group
        if max_episodes is not None and max_episodes < 1:
            raise ValueError('max_episodes must be positive')
        self.max_episodes = max_episodes

    def _paths(self) -> list[Path]:
        if not self.root.is_dir():
            raise FileNotFoundError(
                f'HDF source root does not exist: {self.root}')
        paths = sorted(
            (path for path in self.root.glob(self.pattern) if path.is_file()),
            key=lambda path: _natural_key(path.relative_to(self.root)),
        )
        if self.max_episodes is not None:
            paths = paths[:self.max_episodes]
        if not paths:
            raise FileNotFoundError(
                f'no HDF episodes matched {self.pattern!r} under {self.root}')
        return paths

    def discover(self) -> Iterator[EpisodeRef]:
        for path in self._paths():
            relative = path.relative_to(self.root).as_posix()
            digest = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]
            match = re.fullmatch(r'episode_(\d+)\.hdf5', path.name)
            source_index = int(match.group(1)) if match else None
            yield EpisodeRef(
                episode_uid=f'agilex-aloha-{digest}',
                source_uri=str(path),
                source_episode_index=source_index,
                metadata={
                    'relative_path': relative,
                    'collection':
                    path.parent.relative_to(self.root).as_posix(),
                    'bytes': path.stat().st_size,
                },
            )

    @staticmethod
    def _schema_fingerprint(stream: Any) -> tuple[str, list[dict[str, Any]]]:
        h5py = _require_h5py()
        datasets: list[dict[str, Any]] = []

        def visit(name: str, value: Any) -> None:
            if not isinstance(value, h5py.Dataset):
                return
            shape: list[Any] = list(value.shape)
            if shape:
                shape[0] = 'T'
            datasets.append({
                'path': name,
                'shape': shape,
                'dtype': str(value.dtype),
                'chunks': list(value.chunks) if value.chunks else None,
                'compression': value.compression,
            })

        stream.visititems(visit)
        payload = json.dumps(
            datasets, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(payload).hexdigest(), datasets

    def open_episode(self, episode: EpisodeRef) -> EpisodeRecord:
        h5py = _require_h5py()
        path = Path(episode.source_uri)
        with h5py.File(path, 'r') as stream:
            qpos_key = 'observations/qpos'
            if qpos_key not in stream:
                raise KeyError(f'{path}: missing /{qpos_key}')
            qpos = np.asarray(stream[qpos_key][:], dtype=np.float32)
            if qpos.ndim != 2:
                raise ValueError(f'{path}: qpos must be 2D, got {qpos.shape}')
            length = int(qpos.shape[0])

            fields: dict[str, Any] = {qpos_key: qpos}
            for key in ('action', 'base_action', 'observations/eepose'):
                if key in stream:
                    values = np.asarray(stream[key][:], dtype=np.float32)
                    if values.shape[0] != length:
                        raise ValueError(f'{path}: /{key} length '
                                         f'{values.shape[0]} != {length}')
                    fields[key] = values
            timestamp_key = next(
                (key for key in self._TIMESTAMP_KEYS if key in stream), None)
            if timestamp_key is not None:
                timestamps = np.asarray(
                    stream[timestamp_key][:], dtype=np.float32).reshape(-1)
                if len(timestamps) != length:
                    raise ValueError(
                        f'{path}: /{timestamp_key} length mismatch')
                fields['timestamp'] = timestamps

            media: dict[str, HDF5FrameReader] = {}
            image_group = 'observations/images'
            if image_group in stream:
                for camera in self.cameras:
                    key = f'{image_group}/{camera}'
                    if key not in stream:
                        continue
                    dataset = stream[key]
                    if dataset.shape[0] != length:
                        raise ValueError(f'{path}: /{key} length '
                                         f'{dataset.shape[0]} != {length}')
                    media[camera] = HDF5FrameReader(
                        path,
                        key,
                        length=length,
                        shape=tuple(dataset.shape[1:]),
                        dtype=str(dataset.dtype),
                    )

            fingerprint, schema = self._schema_fingerprint(stream)
            attributes = {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in stream.attrs.items()
                if isinstance(value, (str, int, float, bool, np.generic))
            }

        metadata = dict(episode.metadata)
        metadata.update({
            'schema_fingerprint': fingerprint,
            'source_schema': schema,
            'hdf_attributes': attributes,
            'available_cameras': list(media),
        })
        return EpisodeRecord(
            ref=episode,
            length=length,
            fields=fields,
            media=media,
            metadata=metadata,
        )
