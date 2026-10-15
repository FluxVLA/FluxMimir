import hashlib
import json

import numpy as np
import pytest

from fluxmimir.embodiments import REBOT_DUAL_JOINT_NAMES
from fluxmimir.engines import build_runner_from_cfg


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_video(path, frames):
    av = pytest.importorskip('av')
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('mpeg4', rate=30)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = 'yuv420p'
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_rebot_lerobot_v3_to_canonical_end_to_end(tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    pytest.importorskip('av')

    raw = tmp_path / 'raw'
    (raw / 'meta').mkdir(parents=True)
    info = {
        'codebase_version': 'v3.0',
        'robot_type': 'bi_rebot_b601_follower_ros',
        'features': {
            'observation.state': {
                'names': list(REBOT_DUAL_JOINT_NAMES)
            },
            'action': {
                'names': list(REBOT_DUAL_JOINT_NAMES)
            },
        },
    }
    (raw / 'meta/info.json').write_text(json.dumps(info), encoding='utf-8')
    pq.write_table(
        pa.table({
            'task_index': pa.array([0], type=pa.int64()),
            'task': pa.array(['place both blocks on the plate']),
        }), raw / 'meta/tasks.parquet')

    length = 3
    state = np.arange(length * 14, dtype=np.float32).reshape(length, 14)
    action = state + np.float32(0.25)
    action[:, 6] = np.array([-4.0, -2.0, 0.0], dtype=np.float32)
    action[:, 13] = np.array([-4.7, -1.0, 0.0], dtype=np.float32)
    data_path = raw / 'data/chunk-000/file-000.parquet'
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({
            'observation.state':
            pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
            'action':
            pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
            'timestamp':
            pa.array(np.arange(length, dtype=np.float32) / 30),
            'frame_index':
            pa.array(np.arange(length, dtype=np.int64)),
            'episode_index':
            pa.array(np.zeros(length, dtype=np.int64)),
            'index':
            pa.array(np.arange(length, dtype=np.int64)),
            'task_index':
            pa.array(np.zeros(length, dtype=np.int64)),
        }), data_path)

    frames = np.zeros((length, 4, 6, 3), dtype=np.uint8)
    frames[:, :, :, 1] = 127
    mapping = {
        'cam_high': 'observation.images.cam_front',
        'cam_left_wrist': 'observation.images.cam_wrist_left',
        'cam_right_wrist': 'observation.images.cam_wrist_right',
    }
    source_videos = []
    for source_key in mapping.values():
        path = raw / 'videos' / source_key / 'chunk-000/file-000.mp4'
        _write_video(path, frames)
        source_videos.append(path)

    output = tmp_path / 'canonical'
    runner = build_runner_from_cfg(
        dict(
            type='PipelineRunner',
            work_dir=str(tmp_path / 'work'),
            source=dict(
                type='LeRobotV3Source', root=str(raw), camera_mapping=mapping),
            embodiment=dict(type='ReBotDualAdapter'),
            stages=[
                dict(type='ProbeSourceSchema'),
                dict(type='MapReBotToCanonical128D'),
                dict(type='EncodeEpisodeVideos', codec='source', mode='copy'),
                dict(type='ComputeMaskedEpisodeStatistics'),
                dict(type='BuildCanonicalMetadata'),
            ],
            writer=dict(
                type='CanonicalDatasetWriter',
                root=str(output),
                dataset_id='rebot_test',
                dataset_version='0.0.1'),
            validators=[
                dict(type='CanonicalParquetSchemaValidator'),
                dict(type='UnifiedMaskValidator'),
                dict(type='EpisodeIndexValidator'),
                dict(type='ForeignKeyClosureValidator'),
                dict(type='VideoAlignmentValidator'),
                dict(type='MaskedStatisticsValidator'),
                dict(type='ArtifactManifestValidator'),
            ]))
    try:
        summary = runner.run()
    finally:
        runner.close()

    assert summary['episodes'] == 1
    table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
    observation = np.asarray(
        table['observation.unified_128d'].to_pylist(), dtype=np.float32)
    target = np.asarray(
        table['target.unified_128d'].to_pylist(), dtype=np.float32)
    assert np.allclose(observation[:, :6], state[:, :6])
    assert np.allclose(observation[:, 28], state[:, 6])
    assert np.allclose(target[:, :6], action[:, :6])
    assert np.allclose(target[:, 29], action[:, 13])
    assert table['target.source'].to_pylist() == [1, 1, 1]

    output_videos = sorted((output / 'videos').rglob('*.mp4'))
    assert len(output_videos) == 3
    assert sorted(_sha256(path) for path in output_videos) == sorted(
        _sha256(path) for path in source_videos)
    output_info = json.loads((output / 'meta/info.json').read_text())
    assert output_info['schema_id'] == 'rebot_dual_unified'
    assert output_info['embodiment_id'] == 'rebot_b601_dual'
    assert output_info['unified_layout']['slots'][2]['unit'] == 'rad'
    assert output_info['target_provenance']['semantic'] == (
        'pre-driver-soft-limit joint-position target')
    task = json.loads((output / 'meta/tasks.jsonl').read_text().strip())
    assert task['task'] == 'place both blocks on the plate'
