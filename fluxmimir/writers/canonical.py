from __future__ import annotations

import hashlib
import json
import shutil
from abc import ABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fluxmimir.conversion import EpisodeRecord, masked_statistics
from fluxmimir.embodiments import SOURCE_CAMERAS, BaseEmbodimentAdapter
from fluxmimir.engines.utils import WRITERS
from fluxmimir.schemas import PARQUET_COLUMNS, TARGET_SOURCE, UNIFIED_DIM
from fluxmimir.structures import ArtifactRecord, PipelineContext, StageResult

from .base import BaseWriter


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError('Parquet conversion requires '
                           '`pip install -e .[conversion]`') from exc
    return pa, pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    payload = ''.join(
        json.dumps(value, ensure_ascii=False) + '\n' for value in values)
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(path)


def _fixed_tensor(values: np.ndarray, value_type: Any) -> Any:
    pa, _ = _require_pyarrow()
    values = np.asarray(values)
    result = pa.array(values.reshape(-1), type=value_type)
    for size in reversed(values.shape[1:]):
        result = pa.FixedSizeListArray.from_arrays(result, int(size))
    return result


def _unified_layout(layout_id: str, gripper_unit: str, profile: str,
                    active_ranges: tuple[tuple[int, int], ...],
                    qpose_reconstruction: str) -> dict[str, Any]:
    return {
        'layout_id':
        layout_id,
        'dimension':
        UNIFIED_DIM,
        'range_convention':
        'half_open_[start,end)',
        'padding_value':
        0.0,
        'slots': [
            {
                'range': [0, 14],
                'semantic': 'dual_arm_joint_position',
                'unit': 'rad',
            },
            {
                'range': [14, 28],
                'semantic': 'dual_end_effector_pose',
                'position_unit': 'm',
                'quaternion_order': 'xyzw',
            },
            {
                'range': [28, 30],
                'semantic': 'dual_gripper_position',
                'unit': gripper_unit,
            },
            {
                'range': [30, 62],
                'semantic': 'left_dexterous_hand'
            },
            {
                'range': [62, 94],
                'semantic': 'right_dexterous_hand'
            },
            {
                'range': [94, 98],
                'semantic': 'waist_position'
            },
            {
                'range': [98, 100],
                'semantic': 'head_position'
            },
            {
                'range': [100, 103],
                'semantic': 'mobile_base_signal'
            },
            {
                'range': [103, 109],
                'semantic': 'left_leg_joint_position',
                'unit': 'rad',
            },
            {
                'range': [109, 115],
                'semantic': 'right_leg_joint_position',
                'unit': 'rad',
            },
            {
                'range': [115, 121],
                'semantic': 'mobile_base_rotation_6d'
            },
            {
                'range': [121, 128],
                'semantic': 'reserved'
            },
        ],
        f'{profile}_observation_active_ranges':
        [list(item) for item in active_ranges],
        f'{profile}_optional_eepose_range': [14, 28],
        f'{profile}_target_active_ranges':
        [list(item) for item in active_ranges],
        f'{profile}_reconstruction': {
            'qpose': qpose_reconstruction,
            'eepose': 'unified[14:28]',
        },
    }


class BaseCanonicalDatasetWriter(BaseWriter, ABC):
    """Extension point for canonical Parquet/video/metadata layouts."""


@WRITERS.register_module()
class CanonicalDatasetWriter(BaseCanonicalDatasetWriter):
    """Write the canonical dataset without depending on LeRobot."""

    def __init__(
        self,
        root: str | Path,
        dataset_id: str,
        dataset_version: str,
        embodiment: BaseEmbodimentAdapter,
        chunks_size: int = 1000,
        publish_manifest: bool = True,
        in_place: bool = False,
    ) -> None:
        if chunks_size < 1:
            raise ValueError('chunks_size must be positive')
        self.root = Path(root).expanduser().resolve()
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.chunks_size = int(chunks_size)
        self.publish_manifest = bool(publish_manifest)
        self.in_place = bool(in_place)
        self.embodiment = embodiment
        self.schema_id = self.embodiment.schema_id
        self.schema_version = self.embodiment.schema_version
        self.layout_id = self.embodiment.layout_id
        self.embodiment_id = self.embodiment.embodiment_id
        self.embodiment_description = self.embodiment.embodiment_description
        self.robot_type = self.embodiment.robot_type
        self.context: PipelineContext | None = None
        self.episodes: list[dict[str, Any]] = []
        self.episode_stats: list[dict[str, Any]] = []
        self.tasks: dict[int, dict[str, Any]] = {}
        self.video_features: dict[str, dict[str, Any]] = {}
        self.video_count = 0
        self.warnings: list[str] = []
        self._artifact_hashes: dict[str, dict[str, Any]] = {}

    def setup(self, context: PipelineContext) -> None:
        self.context = context
        manifest = self.root / 'meta/artifacts.sha256.json'
        if manifest.exists():
            raise FileExistsError('refusing to mutate an already finalized '
                                  f'dataset: {self.root}')
        self.root.mkdir(parents=True, exist_ok=True)

    def _data_path(self, episode_index: int) -> Path:
        chunk = episode_index // self.chunks_size
        return (self.root / f'data/chunk-{chunk:03d}' /
                f'episode_{episode_index:06d}.parquet')

    def _video_path(self, episode_index: int, camera: str) -> Path:
        chunk = episode_index // self.chunks_size
        key = f'observation.images.{camera}'
        return (self.root / f'videos/chunk-{chunk:03d}' / key /
                f'episode_{episode_index:06d}.mp4')

    @staticmethod
    def _validate_canonical(value: Mapping[str, Any], length: int) -> None:
        shapes = {
            'observation.unified_128d': (length, UNIFIED_DIM),
            'observation.unified_128d_mask': (length, UNIFIED_DIM),
            'observation.ego2cam': (length, len(SOURCE_CAMERAS), 4, 4),
            'observation.ego2cam_valid': (length, len(SOURCE_CAMERAS)),
            'target.unified_128d': (length, UNIFIED_DIM),
            'target.unified_128d_mask': (length, UNIFIED_DIM),
            'target.source': (length, ),
            'timestamp': (length, ),
            'frame_index': (length, ),
            'episode_index': (length, ),
            'index': (length, ),
            'subtask_index': (length, ),
        }
        for name, expected in shapes.items():
            if name not in value:
                raise KeyError(f'missing canonical field: {name}')
            actual = np.asarray(value[name]).shape
            if actual != expected:
                raise ValueError(f'{name} shape {actual} != {expected}')
        for name in ('observation.unified_128d', 'observation.ego2cam',
                     'target.unified_128d', 'timestamp'):
            if not np.isfinite(np.asarray(value[name])).all():
                raise ValueError(f'{name} contains NaN/Inf')

    def _write_parquet(self, path: Path, value: Mapping[str, Any]) -> None:
        pa, pq = _require_pyarrow()
        table = pa.table({
            'observation.unified_128d':
            _fixed_tensor(
                np.asarray(
                    value['observation.unified_128d'], dtype=np.float32),
                pa.float32()),
            'observation.unified_128d_mask':
            _fixed_tensor(
                np.asarray(
                    value['observation.unified_128d_mask'], dtype=np.bool_),
                pa.bool_()),
            'observation.ego2cam':
            _fixed_tensor(
                np.asarray(value['observation.ego2cam'], dtype=np.float32),
                pa.float32()),
            'observation.ego2cam_valid':
            _fixed_tensor(
                np.asarray(value['observation.ego2cam_valid'], dtype=np.bool_),
                pa.bool_()),
            'target.unified_128d':
            _fixed_tensor(
                np.asarray(value['target.unified_128d'], dtype=np.float32),
                pa.float32()),
            'target.unified_128d_mask':
            _fixed_tensor(
                np.asarray(value['target.unified_128d_mask'], dtype=np.bool_),
                pa.bool_()),
            'target.source':
            pa.array(value['target.source'], type=pa.int8()),
            'timestamp':
            pa.array(value['timestamp'], type=pa.float32()),
            'frame_index':
            pa.array(value['frame_index'], type=pa.int64()),
            'episode_index':
            pa.array(value['episode_index'], type=pa.int64()),
            'index':
            pa.array(value['index'], type=pa.int64()),
            'subtask_index':
            pa.array(value['subtask_index'], type=pa.int64()),
        })
        if table.column_names != list(PARQUET_COLUMNS):
            raise AssertionError('canonical Parquet column order changed')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            pq.write_table(table, temporary, compression='zstd')
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _video_codec(codec: str, available: set[str]) -> str:
        aliases = {
            'h264': ('libx264', 'h264'),
            'av1': ('libsvtav1', 'av1'),
        }
        for candidate in aliases.get(codec, (codec, )):
            if candidate in available:
                return candidate
        raise ValueError(f'PyAV has no encoder for requested codec {codec!r}')

    def _encode_video(self, path: Path, reader: Any,
                      spec: Mapping[str, Any]) -> tuple[int, int]:
        try:
            import av
        except ImportError as exc:  # pragma: no cover - optional media path
            raise RuntimeError('Video encoding requires PyAV; install the '
                               'conversion extra') from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        container = None
        width = height = 0
        frames = 0
        try:
            container = av.open(str(temporary), mode='w', format='mp4')
            codec = self._video_codec(str(spec['codec']), av.codecs_available)
            stream = container.add_stream(codec, rate=int(spec['fps']))
            stream.pix_fmt = 'yuv420p'
            stream.options = {
                'crf': str(spec['crf']),
                'g': str(spec['gop_size']),
            }
            for image in reader:
                image = np.asarray(image)
                if image.ndim != 3 or image.shape[-1] not in (1, 3):
                    raise ValueError(f'invalid RGB frame shape: {image.shape}')
                if image.shape[-1] == 1:
                    image = np.repeat(image, 3, axis=-1)
                image = np.ascontiguousarray(
                    np.clip(image, 0, 255), dtype=np.uint8)
                if frames == 0:
                    height, width = image.shape[:2]
                    stream.width = width
                    stream.height = height
                elif image.shape[:2] != (height, width):
                    raise ValueError('video frame size changed inside episode')
                frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
                frames += 1
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            if spec['frame_count_must_match'] and frames != len(reader):
                raise ValueError(
                    f'encoded {frames} frames, expected {len(reader)}')
            temporary.replace(path)
        finally:
            if container is not None:
                container.close()
            if temporary.exists():
                temporary.unlink()
        return height, width

    @staticmethod
    def _copy_video(path: Path, reader: Any,
                    spec: Mapping[str, Any]) -> tuple[int, int, str]:
        source = Path(getattr(reader, 'path', ''))
        if not source.is_file():
            raise ValueError('copy video mode requires a source file path')
        actual_frames = int(getattr(reader, 'actual_frame_count', 0))
        if (spec['frame_count_must_match'] and actual_frames
                and actual_frames != len(reader)):
            raise ValueError(
                f'{source}: {actual_frames} frames != {len(reader)} rows')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            shutil.copy2(source, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return (int(reader.height), int(reader.width), str(reader.codec))

    def write_episode(self, episode: EpisodeRecord,
                      result: StageResult) -> tuple[ArtifactRecord, ...]:
        if self.context is None:
            raise RuntimeError('writer.setup() was not called')
        if not isinstance(result.value, Mapping):
            raise TypeError('canonical writer expects a mapped dictionary')
        value = result.value
        self._validate_canonical(value, episode.length)
        episode_indices = np.unique(value['episode_index'])
        if len(episode_indices) != 1:
            raise ValueError(
                'episode_index must be constant within an episode')
        episode_index = int(episode_indices[0])
        resume = bool(self.context.metadata.get('resume', False))

        artifacts: list[ArtifactRecord] = []
        data_path = self._data_path(episode_index)
        if not data_path.exists():
            self._write_parquet(data_path, value)
        elif not resume:
            raise FileExistsError(data_path)
        data_hash = _sha256(data_path)
        artifacts.append(ArtifactRecord('parquet', str(data_path), data_hash))
        self._artifact_hashes[data_path.relative_to(self.root).as_posix()] = {
            'sha256': data_hash,
            'bytes': data_path.stat().st_size,
        }

        video_spec = value.get('_video_encoding', {'enabled': False})
        if video_spec.get('enabled'):
            for camera in SOURCE_CAMERAS:
                if camera not in episode.media:
                    raise ValueError(
                        f'{episode.ref.source_uri}: missing {camera}')
                video_path = self._video_path(episode_index, camera)
                if not video_path.exists():
                    if video_spec.get('mode', 'encode') == 'copy':
                        height, width, codec = self._copy_video(
                            video_path, episode.media[camera], video_spec)
                    else:
                        height, width = self._encode_video(
                            video_path, episode.media[camera], video_spec)
                        codec = str(video_spec['codec'])
                elif resume:
                    reader = episode.media[camera]
                    shape = getattr(reader, 'shape', ())
                    if len(shape) >= 3:
                        height, width = int(shape[-3]), int(shape[-2])
                    else:
                        frame = reader.read_frame(0)
                        height, width = frame.shape[:2]
                    codec = (
                        str(getattr(reader, 'codec', 'source'))
                        if video_spec.get('mode') == 'copy' else str(
                            video_spec['codec']))
                else:
                    raise FileExistsError(video_path)
                key = f'observation.images.{camera}'
                feature = {
                    'dtype': 'video',
                    'shape': [3, height, width],
                    'names': ['channels', 'height', 'width'],
                    'info': {
                        'video.height': height,
                        'video.width': width,
                        'video.codec': codec,
                        'video.pix_fmt': 'yuv420p',
                        'video.is_depth_map': False,
                        'video.fps': int(video_spec['fps']),
                        'video.channels': 3,
                        'has_audio': False,
                    },
                }
                if (key in self.video_features
                        and self.video_features[key] != feature):
                    raise ValueError(
                        f'inconsistent video properties for {key}')
                self.video_features[key] = feature
                self.video_count += 1
                video_hash = _sha256(video_path)
                artifacts.append(
                    ArtifactRecord('video', str(video_path), video_hash,
                                   {'camera': camera}))
                relative = video_path.relative_to(self.root).as_posix()
                self._artifact_hashes[relative] = {
                    'sha256': video_hash,
                    'bytes': video_path.stat().st_size,
                }

        stats = value.get('_statistics')
        if stats is None:
            stats = {
                'observation.unified_128d':
                masked_statistics(value['observation.unified_128d'],
                                  value['observation.unified_128d_mask']),
                'target.unified_128d':
                masked_statistics(value['target.unified_128d'],
                                  value['target.unified_128d_mask']),
            }
        task_index = int(episode.metadata.get('task_index', 0))
        task_text = episode.metadata.get('task')
        if task_text:
            task_record = {
                'task_index': task_index,
                'task_id': f'task_{task_index:02d}_source',
                'task': str(task_text),
                'language': 'en',
                'placeholder_status': 'source_provided',
                'annotation_provenance': {
                    'source': 'source_dataset_task_metadata'
                },
            }
            existing = self.tasks.get(task_index)
            if existing is not None and existing != task_record:
                raise ValueError(f'conflicting task metadata: {task_index}')
            self.tasks[task_index] = task_record
        self.episodes.append({
            'episode_index':
            episode_index,
            'episode_uid':
            episode.ref.episode_uid,
            'source_episode_index':
            episode.ref.source_episode_index,
            'source_episode_identity':
            episode.metadata.get('relative_path'),
            'task_index':
            task_index,
            'length':
            episode.length,
            'data_source':
            episode.metadata.get('data_source', 'hdf5'),
            'outcome':
            None,
            'outcome_status':
            'placeholder_unavailable',
            'schema_fingerprint':
            episode.metadata.get('schema_fingerprint'),
        })
        self.episode_stats.append({
            'episode_index': episode_index,
            'stats': stats,
        })
        self.warnings.extend(result.warnings)
        return tuple(artifacts)

    def _features(self, video_features: Mapping[str, Any]) -> dict[str, Any]:
        features = {
            'observation.unified_128d': {
                'dtype': 'float32',
                'shape': [UNIFIED_DIM],
                'names': None,
                'layout_id': self.layout_id,
            },
            'observation.unified_128d_mask': {
                'dtype': 'bool',
                'shape': [UNIFIED_DIM],
                'names': None,
            },
            'observation.ego2cam': {
                'dtype': 'float32',
                'shape': [3, 4, 4],
                'names': None,
            },
            'observation.ego2cam_valid': {
                'dtype': 'bool',
                'shape': [3],
                'names': None,
            },
            'target.unified_128d': {
                'dtype': 'float32',
                'shape': [UNIFIED_DIM],
                'names': None,
                'layout_id': self.layout_id,
            },
            'target.unified_128d_mask': {
                'dtype': 'bool',
                'shape': [UNIFIED_DIM],
                'names': None,
            },
            'target.source': {
                'dtype': 'int8',
                'shape': [1],
                'names': None
            },
            'timestamp': {
                'dtype': 'float32',
                'shape': [1],
                'names': None
            },
            'frame_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None
            },
            'episode_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None
            },
            'index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None
            },
            'subtask_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None
            },
        }
        features.update(video_features)
        return features

    def _build_info(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        episodes = sorted(
            self.episodes, key=lambda item: item['episode_index'])
        total_frames = sum(item['length'] for item in episodes)
        episode_count = len(episodes)
        chunks = {
            item['episode_index'] // self.chunks_size
            for item in episodes
        }
        fps = int(metadata.get('fps', 30))
        active_ranges = [list(item) for item in self.embodiment.active_ranges]
        info = {
            'codebase_version':
            'v2.1',
            'schema_id':
            self.schema_id,
            'schema_version':
            self.schema_version,
            'dataset_id':
            self.dataset_id,
            'dataset_version':
            self.dataset_version,
            'dataset_version_file':
            'meta/fluxvla_dataset_version.json',
            'content_manifest':
            'meta/artifacts.sha256.json',
            'training_ready':
            False,
            'training_readiness_blockers':
            list(self.embodiment.training_readiness_blockers),
            'warning':
            ('Robot observations are real source artifacts. Missing fields '
             'use masked placeholders; inspect field_validity.'),
            'robot_type':
            self.robot_type,
            'embodiment_id':
            self.embodiment_id,
            'embodiment_description':
            self.embodiment_description,
            'total_episodes':
            episode_count,
            'total_frames':
            total_frames,
            'total_tasks':
            max(1, len(self.tasks)),
            'total_subtasks':
            1,
            'total_videos':
            self.video_count,
            'total_chunks':
            len(chunks),
            'chunks_size':
            self.chunks_size,
            'fps':
            fps,
            'splits': {
                'train': f'0:{episode_count}'
            },
            'data_path': ('data/chunk-{episode_chunk:03d}/'
                          'episode_{episode_index:06d}.parquet'),
            'video_path': ('videos/chunk-{episode_chunk:03d}/{video_key}/'
                           'episode_{episode_index:06d}.mp4'),
            'source_dataset': {
                'root': metadata.get('source_root'),
                'format': self.embodiment.source_format,
                'conversion': self.embodiment.conversion_description,
            },
            'storage_policy': {
                'one_parquet_per_episode':
                True,
                'explicit_qpose_eepose_columns':
                False,
                'target_recipe': ('issued target when explicitly confirmed; '
                                  'otherwise masked placeholder'),
                'video_materialization': (self.embodiment.video_materialization
                                          if self.video_count else 'disabled'),
            },
            'field_validity': {
                'observation.unified_128d': {
                    'available': True,
                    'validity_column': 'observation.unified_128d_mask',
                    'always_active_ranges': active_ranges,
                    'optional_eepose_range': [14, 28],
                },
                'observation.ego2cam': {
                    'available': False,
                    'value': 'all_zero_placeholder',
                    'validity_column': 'observation.ego2cam_valid',
                },
                'target.unified_128d': {
                    'available': metadata.get('action_semantics') ==
                    'issued_joint_position_target',
                    'validity_column': 'target.unified_128d_mask',
                },
            },
            'unified_layout':
            _unified_layout(self.layout_id, self.embodiment.gripper_unit,
                            self.embodiment.layout_profile,
                            self.embodiment.active_ranges,
                            self.embodiment.qpose_reconstruction),
            'ego2cam_contract': {
                'column': 'observation.ego2cam',
                'validity_column': 'observation.ego2cam_valid',
                'camera_order': list(SOURCE_CAMERAS),
                'shape': [3, 4, 4],
                'matrix_convention': 'T_camera_from_ego',
                'equation': 'p_camera = T_camera_from_ego @ p_ego',
                'ego_frame': 'base_link',
            },
            'target_source_enum': {
                str(value): name
                for name, value in TARGET_SOURCE.items()
            },
            'annotation_contract': {
                'task_definitions_location': 'meta/tasks.jsonl',
                'episode_task_assignment': 'meta/episodes.jsonl[*].task_index',
                'subtask_definitions_location': 'meta/subtasks.jsonl',
                'frame_subtask_assignment':
                'data/**/*.parquet[*].subtask_index',
            },
            'placeholder_policy': {
                'numeric_tensor_placeholder': 0.0,
                'rule': 'zero placeholders require an explicit false mask',
            },
            'generation': {
                'converter': 'fluxmimir',
                'converter_version': metadata.get('converter_version'),
                'run_id': metadata.get('run_id'),
            },
            'time_alignment': {
                'timestamp_column': 'timestamp',
                'unit': 's',
                'nominal_period_s': 1.0 / fps,
                'fallback': 'synthetic frame_index/fps',
            },
            'features':
            self._features(self.video_features),
        }
        action_provenance = getattr(self.embodiment, 'target_provenance', None)
        if action_provenance:
            info['target_provenance'] = action_provenance
        return info

    def _write_manifest(self, info: Mapping[str, Any]) -> Path:
        artifacts: dict[str, dict[str, Any]] = {}
        manifest_path = self.root / 'meta/artifacts.sha256.json'
        for directory in ('data', 'videos', 'meta'):
            base = self.root / directory
            if not base.exists():
                continue
            for path in sorted(base.rglob('*')):
                if not path.is_file() or path == manifest_path:
                    continue
                relative = path.relative_to(self.root).as_posix()
                size = path.stat().st_size
                cached = self._artifact_hashes.get(relative)
                if cached is not None and cached['bytes'] == size:
                    artifacts[relative] = cached
                else:
                    artifacts[relative] = {
                        'sha256': _sha256(path),
                        'bytes': size,
                    }
        rules = {
            'schema_id': info['schema_id'],
            'schema_version': info['schema_version'],
            'features': info['features'],
            'unified_layout': info['unified_layout'],
            'field_validity': info['field_validity'],
        }
        fingerprint = {
            'schema': rules,
            'dataset_id': self.dataset_id,
            'dataset_version': self.dataset_version,
            'artifacts': artifacts,
        }
        producer_path = Path(__file__).resolve()
        manifest = {
            'manifest_format': 'fluxvla-dataset-manifest-v1',
            'manifest_version': '1.0.0',
            'synthetic': False,
            'schema': {
                'id': self.schema_id,
                'version': self.schema_version,
                'rules_sha256': _canonical_sha256(rules),
            },
            'dataset': {
                'id': self.dataset_id,
                'version': self.dataset_version,
                'content_sha256': _canonical_sha256(fingerprint),
            },
            'producer': {
                'name':
                'fluxmimir',
                'version': (self.context.metadata.get('converter_version')
                            if self.context else None),
                'sha256':
                _sha256(producer_path),
            },
            'artifact_count': len(artifacts),
            'artifacts': artifacts,
        }
        _write_json(manifest_path, manifest)
        return manifest_path

    def finalize(self, metadata: Mapping[str, Any]) -> ArtifactRecord:
        if not self.episodes:
            raise RuntimeError('cannot finalize a dataset with no episodes')
        indices = sorted(item['episode_index'] for item in self.episodes)
        if indices != list(range(len(indices))):
            raise ValueError(f'episode indices are not contiguous: {indices}')
        self.episodes.sort(key=lambda item: item['episode_index'])
        self.episode_stats.sort(key=lambda item: item['episode_index'])

        tasks = sorted(
            self.tasks.values(), key=lambda item: item['task_index'])
        if not tasks:
            tasks = [{
                'task_index': 0,
                'task_id': 'task_00_unknown_episode_level_task',
                'task': 'unknown episode-level task',
                'language': 'en',
                'placeholder_status': 'placeholder_unavailable',
                'annotation_provenance': {
                    'source': self.embodiment.unavailable_annotation_source
                },
            }]
        subtasks = [{
            'subtask_index': 0,
            'subtask_id': 'subtask_00_unknown',
            'subtask': 'unknown',
            'language': 'en',
            'placeholder_status': 'placeholder_unavailable',
            'annotation_provenance': {
                'source': self.embodiment.unavailable_annotation_source
            },
        }]
        info = self._build_info(metadata)
        released_at = datetime.now(timezone.utc).replace(
            microsecond=0).isoformat().replace('+00:00', 'Z')
        version = {
            'fluxvla_dataset_version': self.dataset_version,
            'dataset_id': self.dataset_id,
            'qualified_dataset_id':
            f'{self.dataset_id}@{self.dataset_version}',
            'schema_id': self.schema_id,
            'schema_version': self.schema_version,
            'versioning_scheme': 'Semantic Versioning 2.0.0',
            'release_status': 'converted_with_explicit_placeholders',
            'released_at_utc': released_at,
            'immutable_release': True,
            'content_manifest': 'meta/artifacts.sha256.json',
        }
        _write_json(self.root / 'meta/info.json', info)
        _write_json(self.root / 'meta/fluxvla_dataset_version.json', version)
        _write_jsonl(self.root / 'meta/tasks.jsonl', tasks)
        _write_jsonl(self.root / 'meta/subtasks.jsonl', subtasks)
        _write_jsonl(self.root / 'meta/episodes.jsonl', self.episodes)
        _write_jsonl(self.root / 'meta/episodes_stats.jsonl',
                     self.episode_stats)
        if self.publish_manifest:
            manifest = self._write_manifest(info)
            return ArtifactRecord('manifest', str(manifest), _sha256(manifest))
        info_path = self.root / 'meta/info.json'
        return ArtifactRecord('metadata', str(info_path), _sha256(info_path))
