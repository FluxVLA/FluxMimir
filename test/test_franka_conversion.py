import json

import numpy as np
import pytest

from fluxmimir.conversion.structures import EpisodeRecord
from fluxmimir.embodiments import FRANKA_DUAL_JOINT_NAMES, FrankaDualAdapter
from fluxmimir.embodiments.franka_dual import (_map_pose, _map_state,
                                               _state_mask)
from fluxmimir.engines import build_runner_from_cfg
from fluxmimir.schemas import CAMERA_ORDER, TARGET_SOURCE, UNIFIED_DIM
from fluxmimir.structures import EpisodeRef

MAPPING = {
    'cam_high': 'observation.images.cam_front',
    'cam_left_wrist': 'observation.images.cam_wrist_left',
    'cam_right_wrist': 'observation.images.cam_wrist_right',
}


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


def _source_state(length):
    values = (
        np.arange(length * 16, dtype=np.float32).reshape(length, 16) / 100)
    values[:, 7] = np.linspace(0.01, 0.08, length, dtype=np.float32)
    values[:, 15] = np.linspace(0.02, 0.07, length, dtype=np.float32)
    return values


def _episode(state, eepose=None, action=None, joint_names=None, cameras=None):
    fields = {
        'observations/qpos': state,
        'timestamp': np.arange(len(state), dtype=np.float32) / 30,
    }
    if eepose is not None:
        fields['observations/eepose'] = eepose
    if action is not None:
        fields['action'] = action
    return EpisodeRecord(
        ref=EpisodeRef(
            episode_uid='franka-test', source_uri='memory://franka'),
        length=len(state),
        fields=fields,
        media=dict.fromkeys(cameras or CAMERA_ORDER),
        metadata={'joint_names': joint_names or FRANKA_DUAL_JOINT_NAMES})


def test_franka_maps_state_and_pose_slots():
    state = _source_state(3)
    eepose = state + np.float32(1.0)
    expected = np.zeros((3, UNIFIED_DIM), dtype=np.float32)
    expected[:, 0:7] = state[:, 0:7]
    expected[:, 7:14] = state[:, 8:15]
    expected[:, 28] = state[:, 7]
    expected[:, 29] = state[:, 15]
    expected[:, 14:21] = eepose[:, 0:7]
    expected[:, 21:28] = eepose[:, 8:15]

    assert np.array_equal(_map_state(state) + _map_pose(eepose), expected)


def test_franka_state_mask_covers_arm_and_gripper():
    mask = _state_mask(4)

    assert mask.shape == (4, UNIFIED_DIM)
    assert mask.dtype == np.bool_
    assert np.flatnonzero(mask[0]).tolist() == [*range(0, 14), 28, 29]
    assert np.array_equal(mask, np.tile(mask[0], (4, 1)))


@pytest.mark.parametrize('width,renamed,mutate,code', [
    (15, False, None, 'invalid_state_shape'),
    (16, True, None, 'joint_order'),
    (16, False, 'gripper', 'gripper_unit_range'),
    (16, False, 'negative_gripper', 'gripper_unit_range'),
    (16, False, 'nan', 'nonfinite_state'),
])
def test_franka_preflight_rejects_bad_source(width, renamed, mutate, code):
    state = _source_state(3)[:, :width]
    if mutate == 'gripper':
        state[:, (7, 15)] = 0.5
    elif mutate == 'negative_gripper':
        state[:, (7, 15)] = -0.01
    elif mutate == 'nan':
        state[0, 0] = np.nan
    names = FRANKA_DUAL_JOINT_NAMES
    if renamed:
        names = ('left_panda_joint8', *FRANKA_DUAL_JOINT_NAMES[1:])

    report = FrankaDualAdapter().preflight(_episode(state, joint_names=names))

    assert [issue.code for issue in report.issues] == [code]


def test_franka_preflight_rejects_missing_camera():
    episode = _episode(_source_state(3), cameras=CAMERA_ORDER[:2])

    report = FrankaDualAdapter().preflight(episode)

    assert [issue.code for issue in report.issues] == ['missing_cameras']


def test_franka_target_stays_masked_without_action():
    target = FrankaDualAdapter().map_target(_episode(_source_state(4)))

    assert target['target.unified_107d'].shape == (4, UNIFIED_DIM)
    assert not target['target.unified_107d'].any()
    assert not target['target.unified_107d_mask'].any()
    assert target['target.source'].tolist() == [5, 5, 5, 5]


def test_franka_observation_without_eepose_keeps_pose_masked():
    mapped = FrankaDualAdapter().map_observation(_episode(_source_state(3)))

    assert not mapped['observation.unified_107d'][:, 14:28].any()
    assert not mapped['observation.unified_107d_mask'][:, 14:28].any()


def test_franka_confirmed_action_maps_to_target():
    action = _source_state(4) + np.float32(1.0)
    adapter = FrankaDualAdapter(
        action_semantics='issued_joint_position_target')
    target = adapter.map_target(_episode(_source_state(4), action=action))
    mask = target['target.unified_107d_mask']

    assert np.array_equal(target['target.unified_107d'], _map_state(action))
    assert np.flatnonzero(mask[0]).tolist() == [*range(0, 14), 28, 29]
    assert np.array_equal(mask, np.tile(mask[0], (4, 1)))
    assert (target['target.source'] == TARGET_SOURCE['operator']).all()


def _write_v2_dataset(raw, length=3):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    pytest.importorskip('av')

    (raw / 'meta').mkdir(parents=True)
    (raw / 'meta/info.json').write_text(
        json.dumps({
            'codebase_version': 'v2.1',
            'robot_type': 'franka_dual',
            'fps': 30,
            'total_episodes': 1,
            'features': {
                'observation.state': {
                    'dtype': 'float32',
                    'shape': [16],
                    'names': [list(FRANKA_DUAL_JOINT_NAMES)],
                },
                'observation.eepose': {
                    'dtype': 'float32',
                    'shape': [16],
                },
            },
        }),
        encoding='utf-8')
    (raw / 'meta/tasks.jsonl').write_text(
        json.dumps({
            'task_index': 0,
            'task': 'hand the bucket to the left arm'
        }) + '\n',
        encoding='utf-8')

    state = _source_state(length)
    eepose = state + np.float32(1.0)
    data_path = raw / 'data/chunk-000/episode_000000.parquet'
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({
            'observation.state':
            pa.array(state.tolist(), type=pa.list_(pa.float32(), 16)),
            'observation.eepose':
            pa.array(eepose.tolist(), type=pa.list_(pa.float32(), 16)),
            'timestamp':
            pa.array(np.arange(length, dtype=np.float32) / 30),
            'task_index':
            pa.array(np.zeros(length, dtype=np.int64)),
        }), data_path)

    frames = np.zeros((length, 4, 6, 3), dtype=np.uint8)
    frames[:, :, :, 1] = 127
    for source_key in MAPPING.values():
        path = raw / 'videos/chunk-000' / source_key / 'episode_000000.mp4'
        _write_video(path, frames)
    return state, eepose


def test_franka_lerobot_v2_end_to_end(tmp_path):
    pq = pytest.importorskip('pyarrow.parquet')

    raw = tmp_path / 'raw'
    output = tmp_path / 'canonical'
    state, eepose = _write_v2_dataset(raw)

    runner = build_runner_from_cfg(
        dict(
            type='PipelineRunner',
            work_dir=str(tmp_path / 'work'),
            source=dict(
                type='LeRobotV2Source', root=str(raw), camera_mapping=MAPPING),
            embodiment=dict(type='FrankaDualAdapter'),
            stages=[
                dict(type='ProbeSourceSchema'),
                dict(type='MapFrankaToCanonical107D'),
                dict(type='EncodeEpisodeVideos', codec='source', mode='copy'),
                dict(type='ComputeMaskedEpisodeStatistics'),
                dict(type='BuildCanonicalMetadata'),
            ],
            writer=dict(
                type='CanonicalDatasetWriter',
                root=str(output),
                dataset_id='franka_test',
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
        table['observation.unified_107d'].to_pylist(), dtype=np.float32)
    mask = np.asarray(
        table['observation.unified_107d_mask'].to_pylist(), dtype=np.bool_)
    assert np.allclose(observation[:, 0:7], state[:, 0:7])
    assert np.allclose(observation[:, 7:14], state[:, 8:15])
    assert np.allclose(observation[:, 28], state[:, 7])
    assert np.allclose(observation[:, 29], state[:, 15])
    assert np.allclose(observation[:, 14:21], eepose[:, 0:7])
    assert np.allclose(observation[:, 21:28], eepose[:, 8:15])
    assert mask[:, 0:30].all()
    assert not mask[:, 30:].any()
    assert not np.asarray(table['target.unified_107d_mask'].to_pylist()).any()
    assert table['target.source'].to_pylist() == [5] * 3
    assert len(sorted((output / 'videos').rglob('*.mp4'))) == 3

    info = json.loads((output / 'meta/info.json').read_text())
    assert info['schema_id'] == 'franka_dual_unified'
    assert info['embodiment_id'] == 'franka_dual'
