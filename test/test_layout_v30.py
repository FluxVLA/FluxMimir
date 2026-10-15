import json
from pathlib import Path

import numpy as np
import pytest
from mmengine import Config

from fluxmimir.conversion.stages import EncodeEpisodeVideos
from fluxmimir.embodiments import AgileXAlohaAdapter
from fluxmimir.engines import WRITERS, build_runner_from_cfg
from fluxmimir.schemas import PARQUET_COLUMNS, UNIFIED_DIM
from fluxmimir.writers import (BaseCanonicalDatasetWriter,
                               CanonicalDatasetV30Writer,
                               CanonicalDatasetWriter)

REPO_ROOT = Path(__file__).resolve().parents[1]
V30_CONFIG = (
    REPO_ROOT / 'configs' / 'conversion' /
    'agilex_aloha_hdf_to_canonical_v30.py')
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def _write_source(root: Path, length: int = 3) -> np.ndarray:
    h5py = pytest.importorskip('h5py')
    root.mkdir(parents=True, exist_ok=True)
    qpos = np.arange(length * 14, dtype=np.float32).reshape(length, 14) / 10
    with h5py.File(root / 'episode_0.hdf5', 'w') as stream:
        stream.create_dataset('observations/qpos', data=qpos)
        stream.create_dataset('action', data=qpos + 1)
        for camera in CAMERAS:
            stream.create_dataset(
                f'observations/images/{camera}',
                data=np.zeros((length, 4, 6, 3), dtype=np.uint8))
    return qpos


def _convert(tmp_path: Path, writer_type: str, validators: list):
    pytest.importorskip('pyarrow.parquet', reason='pyarrow is optional')
    pytest.importorskip('av', reason='PyAV is part of the conversion extra')
    source = tmp_path / 'raw' / 'collection'
    qpos = _write_source(source)
    output = tmp_path / writer_type
    runner = build_runner_from_cfg(
        dict(
            type='PipelineRunner',
            work_dir=str(tmp_path / f'work_{writer_type}'),
            source=dict(type='AlohaHDF5Source', root=str(tmp_path / 'raw')),
            embodiment=dict(
                type='AgileXAlohaAdapter', action_semantics='unknown'),
            stages=[
                dict(type='ProbeSourceSchema'),
                dict(type='MapAlohaToCanonical128D'),
                dict(type='EncodeEpisodeVideos', enabled=True, codec='h264'),
                dict(type='ComputeMaskedEpisodeStatistics'),
                dict(type='BuildCanonicalMetadata'),
            ],
            writer=dict(
                type=writer_type,
                root=str(output),
                dataset_id='unit_test',
                dataset_version='0.0.1',
            ),
            validators=validators,
        ))
    try:
        summary = runner.run()
    finally:
        runner.close()
    return output, qpos, summary


def test_v30_writer_is_registered_as_a_layout():
    assert WRITERS.get(
        'CanonicalDatasetV30Writer') is CanonicalDatasetV30Writer
    assert issubclass(CanonicalDatasetV30Writer, BaseCanonicalDatasetWriter)


def test_v30_paths_use_the_per_chunk_file_ordinal(tmp_path):
    writer = CanonicalDatasetV30Writer(
        root=tmp_path / 'out',
        dataset_id='unit_test',
        dataset_version='0.0.1',
        embodiment=AgileXAlohaAdapter(),
        chunks_size=1,
    )
    root = writer.root
    assert writer._data_path(0).relative_to(root).as_posix() == (
        'data/chunk-000/file-000.parquet')
    assert writer._data_path(1).relative_to(root).as_posix() == (
        'data/chunk-001/file-000.parquet')
    assert writer._video_path(1, 'cam_high').relative_to(root).as_posix() == (
        'videos/observation.images.cam_high/chunk-001/file-000.mp4')


def test_v30_layout_end_to_end(tmp_path):
    pq = pytest.importorskip('pyarrow.parquet', reason='pyarrow is optional')
    pytest.importorskip('av', reason='PyAV is part of the conversion extra')
    output, qpos, summary = _convert(tmp_path, 'CanonicalDatasetV30Writer', [
        dict(type='CanonicalV30LayoutValidator'),
        dict(type='ArtifactManifestValidator')
    ])

    assert summary['episodes'] == 1
    assert summary['frames'] == 3

    data_path = output / 'data/chunk-000/file-000.parquet'
    assert data_path.is_file()
    assert not (output / 'data/chunk-000/episode_000000.parquet').exists()
    table = pq.read_table(data_path)
    assert tuple(table.column_names) == PARQUET_COLUMNS
    assert table.num_rows == 3
    observation = np.asarray(
        table['observation.unified_128d'].to_pylist(), dtype=np.float32)
    mask = np.asarray(table['observation.unified_128d_mask'].to_pylist())
    target_mask = np.asarray(table['target.unified_128d_mask'].to_pylist())
    assert observation.shape == (3, UNIFIED_DIM)
    assert np.allclose(observation[:, :6], qpos[:, :6])
    assert np.allclose(observation[:, 7:13], qpos[:, 7:13])
    assert np.allclose(observation[:, 28], qpos[:, 6])
    assert np.allclose(observation[:, 29], qpos[:, 13])
    assert mask[:, :6].all() and mask[:, 7:13].all()
    assert mask[:, 28:30].all() and not mask[:, 14:28].any()
    assert not observation[:, 103:].any() and not mask[:, 103:].any()
    assert not target_mask.any()
    assert table['target.source'].to_pylist() == [5, 5, 5]

    for camera in CAMERAS:
        video_path = (
            output / f'videos/observation.images.{camera}' /
            'chunk-000/file-000.mp4')
        assert video_path.is_file()
        assert video_path.stem == data_path.stem
    assert not list(
        (output / 'videos').glob('chunk-*/observation.images.*/*.mp4'))

    info = json.loads((output / 'meta/info.json').read_text(encoding='utf-8'))
    assert info['codebase_version'] == 'v3.0'
    assert info['chunks_size'] == 1000
    assert info['data_files_size_in_mb'] == 100
    assert info['video_files_size_in_mb'] == 200
    assert 'total_chunks' not in info and 'total_videos' not in info
    assert info['unified_layout']['dimension'] == UNIFIED_DIM
    assert info['features']['observation.unified_128d']['shape'] == [
        UNIFIED_DIM
    ]
    assert info['data_path'] == (
        'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet')

    tasks = pq.read_table(output / 'meta/tasks.parquet').to_pylist()
    assert [row['task_index'] for row in tasks] == [0]

    episodes = pq.read_table(output /
                             'meta/episodes/chunk-000/file-000.parquet')
    assert episodes.num_rows == 1
    row = episodes.to_pylist()[0]
    assert row['episode_index'] == 0
    assert row['length'] == 3
    assert row['data/chunk_index'] == 0 and row['data/file_index'] == 0
    assert row['dataset_from_index'] == 0
    assert row['dataset_to_index'] == 3
    assert row['tasks'] == [tasks[0]['task']]
    for camera in CAMERAS:
        key = f'videos/observation.images.{camera}'
        assert row[f'{key}/chunk_index'] == 0
        assert row[f'{key}/file_index'] == 0
        assert row[f'{key}/from_timestamp'] <= row[f'{key}/to_timestamp']

    stats = json.loads(
        (output / 'meta/stats.json').read_text(encoding='utf-8'))
    assert stats['observation.unified_128d']['count'] == (
        mask.sum(axis=0).tolist())
    assert stats['target.unified_128d']['count'] == (
        target_mask.sum(axis=0).tolist())
    assert row['stats/observation.unified_128d/count'] == (
        mask.sum(axis=0).tolist())

    assert (output / 'meta/fluxvla_dataset_version.json').is_file()
    assert (output / 'meta/artifacts.sha256.json').is_file()
    assert not (output / 'meta/tasks.jsonl').exists()
    assert not (output / 'meta/episodes.jsonl').exists()
    assert not (output / 'meta/episodes_stats.jsonl').exists()


def test_v30_content_matches_the_v21_layout(tmp_path):
    pq = pytest.importorskip('pyarrow.parquet', reason='pyarrow is optional')
    pytest.importorskip('av', reason='PyAV is part of the conversion extra')
    legacy, _, _ = _convert(tmp_path, 'CanonicalDatasetWriter', [])
    v30, _, _ = _convert(tmp_path, 'CanonicalDatasetV30Writer', [])

    left = pq.read_table(legacy / 'data/chunk-000/episode_000000.parquet')
    right = pq.read_table(v30 / 'data/chunk-000/file-000.parquet')
    assert left.column_names == right.column_names
    for name in PARQUET_COLUMNS:
        assert left[name].to_pylist() == right[name].to_pylist()

    import av
    for camera in CAMERAS:
        video_path = (
            v30 / f'videos/observation.images.{camera}' /
            'chunk-000/file-000.mp4')
        with av.open(str(video_path)) as container:
            assert len(list(container.decode(video=0))) == 3


def test_video_options_keep_the_defaults_without_preset():
    assert CanonicalDatasetWriter._video_options({
        'crf': 23,
        'gop_size': 30,
    }) == {
        'crf': '23',
        'g': '30',
    }
    assert CanonicalDatasetWriter._video_options({
        'crf': 18,
        'gop_size': 30,
        'preset': 'veryfast',
    }) == {
        'crf': '18',
        'g': '30',
        'preset': 'veryfast',
    }


def test_encode_episode_videos_resolves_the_preset():
    assert EncodeEpisodeVideos().spec['preset'] is None
    spec = EncodeEpisodeVideos(preset='veryfast', crf=18).spec
    assert spec['preset'] == 'veryfast'
    assert spec['crf'] == 18


def test_v30_config_uses_the_production_encoder():
    cfg = Config.fromfile(V30_CONFIG)

    assert cfg.pipeline.source.type == 'AlohaHDF5Source'
    assert cfg.pipeline.embodiment.schema_version == '2.0.0'
    assert cfg.pipeline.embodiment.layout_id == (
        'lingbot_vla2_dexhand32x2_128_v1')
    assert cfg.pipeline.writer.type == 'CanonicalDatasetV30Writer'
    assert cfg.pipeline.writer.chunks_size == 1000
    stages = {stage['type']: stage for stage in cfg.pipeline.stages}
    assert stages['EncodeEpisodeVideos']['preset'] == 'veryfast'
    assert stages['EncodeEpisodeVideos']['crf'] == 18
    validator_types = [item['type'] for item in cfg.pipeline.validators]
    assert 'CanonicalV30LayoutValidator' in validator_types
