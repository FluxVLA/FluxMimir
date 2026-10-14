import hashlib
import json

import numpy as np
import pytest

from fluxmimir.engines import build_runner_from_cfg

MAPPING = {
    'cam_high': 'observation.images.cam_high',
    'cam_left_wrist': 'observation.images.cam_left_wrist',
    'cam_right_wrist': 'observation.images.cam_right_wrist',
}

STATE_NAMES = [
    'left_waist', 'left_shoulder', 'left_elbow', 'left_forearm_roll',
    'left_wrist_angle', 'left_wrist_rotate', 'left_gripper', 'right_waist',
    'right_shoulder', 'right_elbow', 'right_forearm_roll', 'right_wrist_angle',
    'right_wrist_rotate', 'right_gripper'
]


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


def _write_v2_dataset(raw,
                      length=4,
                      with_action=True,
                      with_eepose=True,
                      frame_tasks=None):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    pytest.importorskip('av')

    features = {
        'observation.state': {
            'dtype': 'float32',
            'shape': [14],
            'names': [STATE_NAMES],
        },
        'timestamp': {
            'dtype': 'float32',
            'shape': [1]
        },
    }
    if with_action:
        features['action'] = {'dtype': 'float32', 'shape': [14]}
    if with_eepose:
        features['observation.eepose'] = {'dtype': 'float32', 'shape': [14]}
    (raw / 'meta').mkdir(parents=True)
    (raw / 'meta/info.json').write_text(
        json.dumps({
            'codebase_version': 'v2.1',
            'robot_type': 'aloha',
            'fps': 30,
            'total_episodes': 1,
            'features': features,
        }),
        encoding='utf-8')
    task_rows = [{'task_index': 0, 'task': 'fold the cloth'}]
    if frame_tasks is not None:
        task_rows.append({'task_index': 1, 'task': 'place the cloth'})
    (raw / 'meta/tasks.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in task_rows), encoding='utf-8')

    state = (np.arange(length * 14, dtype=np.float32).reshape(length, 14) / 10)
    action = state + np.float32(0.25)
    eepose = state + np.float32(0.5)
    columns = {
        'observation.state':
        pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
        'timestamp':
        pa.array(np.arange(length, dtype=np.float32) / 30),
        'frame_index':
        pa.array(np.arange(length, dtype=np.int64)),
        'episode_index':
        pa.array(np.zeros(length, dtype=np.int64)),
        'index':
        pa.array(np.arange(length, dtype=np.int64)),
        'task_index':
        pa.array(
            np.zeros(length, dtype=np.int64) if frame_tasks is
            None else np.asarray(frame_tasks, dtype=np.int64)),
    }
    if with_action:
        columns['action'] = pa.array(
            action.tolist(), type=pa.list_(pa.float32(), 14))
    if with_eepose:
        columns['observation.eepose'] = pa.array(
            eepose.tolist(), type=pa.list_(pa.float32(), 14))
    data_path = raw / 'data/chunk-000/episode_000000.parquet'
    data_path.parent.mkdir(parents=True)
    pq.write_table(pa.table(columns), data_path)

    frames = np.zeros((length, 4, 6, 3), dtype=np.uint8)
    frames[:, :, :, 1] = 127
    source_videos = []
    for source_key in MAPPING.values():
        path = (raw / 'videos/chunk-000' / source_key / 'episode_000000.mp4')
        _write_video(path, frames)
        source_videos.append(path)
    return state, eepose, source_videos


def _run_conversion(tmp_path, raw, output):
    runner = build_runner_from_cfg(
        dict(
            type='PipelineRunner',
            work_dir=str(tmp_path / 'work'),
            source=dict(
                type='LeRobotV2Source', root=str(raw), camera_mapping=MAPPING),
            embodiment=dict(type='AgileXAlohaAdapter'),
            stages=[
                dict(type='ProbeSourceSchema'),
                dict(type='MapAlohaToCanonical107D'),
                dict(type='EncodeEpisodeVideos', codec='source', mode='copy'),
                dict(type='ComputeMaskedEpisodeStatistics'),
                dict(type='BuildCanonicalMetadata'),
            ],
            writer=dict(
                type='CanonicalDatasetWriter',
                root=str(output),
                dataset_id='aloha_v2_test',
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
        return runner.run()
    finally:
        runner.close()


def test_lerobot_v2_to_canonical_end_to_end(tmp_path):
    pq = pytest.importorskip('pyarrow.parquet')

    raw = tmp_path / 'raw'
    output = tmp_path / 'canonical'
    state, eepose, source_videos = _write_v2_dataset(raw)

    summary = _run_conversion(tmp_path, raw, output)

    assert summary['episodes'] == 1
    table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
    observation = np.asarray(
        table['observation.unified_107d'].to_pylist(), dtype=np.float32)
    mask = np.asarray(
        table['observation.unified_107d_mask'].to_pylist(), dtype=np.bool_)
    assert np.allclose(observation[:, :6], state[:, :6])
    assert np.allclose(observation[:, 7:13], state[:, 7:13])
    assert np.allclose(observation[:, 28], state[:, 6])
    assert np.allclose(observation[:, 29], state[:, 13])
    assert np.allclose(observation[:, 14:28], eepose)
    assert mask[:, 14:28].all()
    assert not np.asarray(table['target.unified_107d_mask'].to_pylist()).any()
    assert table['target.source'].to_pylist() == [5] * 4
    output_videos = sorted((output / 'videos').rglob('*.mp4'))
    assert len(output_videos) == 3
    assert sorted(_sha256(path) for path in output_videos) == sorted(
        _sha256(path) for path in source_videos)
    info = json.loads((output / 'meta/info.json').read_text())
    assert info['schema_id'] == 'agilex_aloha_unified'
    assert info['embodiment_id'] == 'agilex_aloha'
    task = json.loads((output / 'meta/tasks.jsonl').read_text().strip())
    assert task['task'] == 'fold the cloth'


def test_lerobot_v2_mixed_frame_labels_use_an_unavailable_task(tmp_path):
    pq = pytest.importorskip('pyarrow.parquet')

    raw = tmp_path / 'raw'
    output = tmp_path / 'canonical'
    _write_v2_dataset(
        raw, with_action=False, with_eepose=False, frame_tasks=[0, 0, 1, 1])

    assert _run_conversion(tmp_path, raw, output)['episodes'] == 1

    episode = json.loads((output / 'meta/episodes.jsonl').read_text().strip())
    tasks = {
        row['task_index']: row['task']
        for row in (json.loads(line)
                    for line in (output /
                                 'meta/tasks.jsonl').read_text().splitlines())
    }
    assert tasks[episode['task_index']] == 'unknown episode-level task'
    table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
    mask = np.asarray(
        table['observation.unified_107d_mask'].to_pylist(), dtype=np.bool_)
    assert table.num_rows == 4
    assert not mask[:, 14:28].any()
