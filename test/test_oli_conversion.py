import json
from pathlib import Path

import numpy as np
import pytest
from mmengine import Config

from fluxmimir.conversion.structures import EpisodeRecord
from fluxmimir.embodiments import OliAdapter
from fluxmimir.embodiments.oli import OLI_BLOCKS
from fluxmimir.engines import build_runner_from_cfg
from fluxmimir.schemas import (OLI_128D_SCHEMA, OLI_CAMERA_ORDER,
                               OLI_STATE_JOINT_NAMES, TARGET_SOURCE,
                               UNIFIED_DIM, oli_qpose_indices)
from fluxmimir.structures import EpisodeRef

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT / 'configs' / 'conversion' /
    'oli_lerobot_v2_to_canonical_128d.py')

MAPPING = {
    'head': 'observation.images.head',
    'left_wrist': 'observation.images.left_wrist',
}

OBSERVATION_ACTIVE = [
    *range(0, 14), 28, 29, *range(94, 97), 98, 99, *range(103, 115)
]
TARGET_ACTIVE = sorted([*OBSERVATION_ACTIVE, 100, 101, 102, *range(115, 121)])
OBSERVATION_INERT = ((14, 28), (30, 94), (97, 98), (100, 103), (115, 128))


def _state(length=4):
    return (np.arange(length * 33, dtype=np.float32).reshape(length, 33) / 100)


def _action(length=4):
    values = np.arange(length * 42, dtype=np.float32).reshape(length, 42)
    return values / 50 + np.float32(1.0)


def _episode(state, action=None, cameras=None, joint_names=None, length=None):
    fields = {
        'observations/qpos': state,
        'timestamp': np.arange(len(state), dtype=np.float32) / 30,
    }
    if action is not None:
        fields['action'] = action
    return EpisodeRecord(
        ref=EpisodeRef(episode_uid='oli-test', source_uri='memory://oli'),
        length=length if length is not None else len(state),
        fields=fields,
        media=dict.fromkeys(cameras or OLI_CAMERA_ORDER),
        metadata={'joint_names': joint_names or OLI_STATE_JOINT_NAMES})


def test_oli_schema_declares_the_two_camera_profile():
    shapes = {
        feature.name: feature.shape
        for feature in OLI_128D_SCHEMA.features
    }

    assert OLI_128D_SCHEMA.schema_id == 'oli_unified'
    assert OLI_128D_SCHEMA.schema_version == '1.0.0'
    assert OLI_128D_SCHEMA.layout_id == 'oli_128_v1'
    assert shapes['observation.unified_128d'] == (UNIFIED_DIM, )
    assert shapes['observation.ego2cam'] == (2, 4, 4)
    assert shapes['observation.ego2cam_valid'] == (2, )
    assert oli_qpose_indices() == (0, 1, 2, 3, 4, 5, 6, 28, 7, 8, 9, 10, 11,
                                   12, 13, 29)


def test_oli_observation_follows_the_frozen_mapping_table():
    state = _state()
    mapped = OliAdapter().map_observation(_episode(state))
    unified = mapped['observation.unified_128d']
    mask = mapped['observation.unified_128d_mask']

    for block in OLI_BLOCKS:
        if block.state is None:
            continue
        start, end = block.unified
        low, high = block.state
        assert np.array_equal(unified[:, start:end], state[:, low:high]), (
            block.name)
        assert mask[:, start:end].all(), block.name
    assert np.array_equal(unified[:, 0:14], state[:, 17:31])
    assert np.array_equal(unified[:, 28:30], state[:, 31:33])
    assert np.array_equal(unified[:, 94:97], state[:, 12:15])
    assert np.array_equal(unified[:, 98:100], state[:, 15:17])
    assert np.array_equal(unified[:, 103:109], state[:, 0:6])
    assert np.array_equal(unified[:, 109:115], state[:, 6:12])
    assert np.flatnonzero(mask[0]).tolist() == OBSERVATION_ACTIVE


def test_oli_observation_leaves_absent_blocks_masked():
    mapped = OliAdapter().map_observation(_episode(_state()))
    unified = mapped['observation.unified_128d']
    mask = mapped['observation.unified_128d_mask']

    for start, end in OBSERVATION_INERT:
        assert not unified[:, start:end].any(), (start, end)
        assert not mask[:, start:end].any(), (start, end)
    assert not unified[:, 30:94].any()


def test_oli_target_follows_the_frozen_mapping_table():
    action = _action()
    adapter = OliAdapter(
        action_semantics='issued_joint_position_target',
        action_source='operator')
    target = adapter.map_target(_episode(_state(), action=action))
    values = target['target.unified_128d']
    mask = target['target.unified_128d_mask']

    for block in OLI_BLOCKS:
        if block.action is None:
            continue
        start, end = block.unified
        low, high = block.action
        assert np.array_equal(values[:, start:end], action[:, low:high]), (
            block.name)
        assert mask[:, start:end].all(), block.name
    assert np.array_equal(values[:, 0:14], action[:, 17:31])
    assert np.array_equal(values[:, 28:30], action[:, 40:42])
    assert np.array_equal(values[:, 100:103], action[:, 31:34])
    assert np.array_equal(values[:, 103:109], action[:, 0:6])
    assert np.array_equal(values[:, 109:115], action[:, 6:12])
    assert np.array_equal(values[:, 115:121], action[:, 34:40])
    assert np.flatnonzero(mask[0]).tolist() == TARGET_ACTIVE
    assert (target['target.source'] == TARGET_SOURCE['operator']).all()


def test_oli_target_stays_masked_until_semantics_are_confirmed():
    target = OliAdapter().map_target(_episode(_state(), action=_action()))

    assert not target['target.unified_128d'].any()
    assert not target['target.unified_128d_mask'].any()
    assert set(target['target.source'].tolist()) == {TARGET_SOURCE['unknown']}


def test_oli_observation_declares_two_camera_ego2cam():
    mapped = OliAdapter().map_observation(_episode(_state(length=5)))

    assert mapped['observation.ego2cam'].shape == (5, 2, 4, 4)
    assert mapped['observation.ego2cam_valid'].shape == (5, 2)
    assert not mapped['observation.ego2cam'].any()
    assert not mapped['observation.ego2cam_valid'].any()


@pytest.mark.parametrize('mutation,code', [
    ('state_width', 'invalid_state_shape'),
    ('state_length', 'invalid_state_shape'),
    ('nonfinite_state', 'nonfinite_state'),
    ('action_width', 'invalid_action_shape'),
    ('nonfinite_action', 'nonfinite_action'),
    ('joint_order', 'joint_order'),
    ('missing_camera', 'missing_cameras'),
])
def test_oli_preflight_rejects_mismatched_sources(mutation, code):
    state = _state()
    action = _action()
    episode = _episode(state, action=action)
    if mutation == 'state_width':
        state = state[:, :32]
        episode = _episode(state, action=action)
    elif mutation == 'state_length':
        episode = _episode(state, action=_action(5), length=len(state) + 1)
    elif mutation == 'nonfinite_state':
        state[0, 0] = np.nan
        episode = _episode(state, action=action)
    elif mutation == 'action_width':
        action = action[:, :41]
        episode = _episode(state, action=action)
    elif mutation == 'nonfinite_action':
        action[0, 0] = np.inf
        episode = _episode(state, action=action)
    elif mutation == 'joint_order':
        episode = _episode(
            state,
            action=action,
            joint_names=('right_hip_pitch_joint', *OLI_STATE_JOINT_NAMES[1:]))
    elif mutation == 'missing_camera':
        episode = _episode(state, action=action, cameras=OLI_CAMERA_ORDER[:1])

    report = OliAdapter(
        action_semantics='issued_joint_position_target').preflight(episode)

    assert [issue.code for issue in report.issues] == [code]


def test_oli_preflight_accepts_the_production_shape():
    report = OliAdapter(
        action_semantics='issued_joint_position_target').preflight(
            _episode(_state(), action=_action()))

    assert report.ok
    assert report.issues == ()


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


def _write_oli_v2_dataset(raw, length=4):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    pytest.importorskip('av')

    (raw / 'meta').mkdir(parents=True)
    (raw / 'meta/info.json').write_text(
        json.dumps({
            'codebase_version': 'v2.1',
            'robot_type': 'oli',
            'fps': 30,
            'total_episodes': 1,
            'total_frames': length,
            'features': {
                'observation.images.head': {
                    'dtype': 'video',
                    'shape': [480, 640, 3],
                },
                'observation.images.left_wrist': {
                    'dtype': 'video',
                    'shape': [480, 640, 3],
                },
                'observation.state': {
                    'dtype': 'float32',
                    'shape': [33],
                    'names': list(OLI_STATE_JOINT_NAMES),
                },
                'action': {
                    'dtype': 'float32',
                    'shape': [42],
                    'names': None,
                },
            },
        }),
        encoding='utf-8')
    (raw / 'meta/tasks.jsonl').write_text(
        json.dumps({
            'task_index': 0,
            'task': 'lift the red basket'
        }) + '\n',
        encoding='utf-8')

    state = _state(length)
    action = _action(length)
    data_path = raw / 'data/chunk-000/episode_000000.parquet'
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({
            'observation.state':
            pa.array(state.tolist(), type=pa.list_(pa.float32(), 33)),
            'action':
            pa.array(action.tolist(), type=pa.list_(pa.float32(), 42)),
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

    frames = np.zeros((length, 8, 8, 3), dtype=np.uint8)
    frames[:, :, :, 1] = 127
    for source_key in MAPPING.values():
        _write_video(
            raw / 'videos/chunk-000' / source_key / 'episode_000000.mp4',
            frames)
    return state, action


def test_oli_lerobot_v2_end_to_end(tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')

    raw = tmp_path / 'raw'
    output = tmp_path / 'canonical'
    state, action = _write_oli_v2_dataset(raw)
    pipeline = Config.fromfile(CONFIG).pipeline
    runner = build_runner_from_cfg(
        dict(
            pipeline.runner,
            work_dir=str(tmp_path / 'work'),
        ),
        default_args={
            'source': dict(pipeline.source, root=str(raw), max_episodes=1),
            'embodiment': pipeline.embodiment,
            'stages': pipeline.stages,
            'writer': dict(pipeline.writer, root=str(output)),
            'validators': pipeline.get('validators', []),
        })
    try:
        summary = runner.run()
    finally:
        runner.close()

    assert summary['episodes'] == 1
    table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
    unified_type = table.schema.field('observation.unified_128d').type
    assert unified_type == pa.list_(pa.float32(), UNIFIED_DIM)
    observation = np.asarray(
        table['observation.unified_128d'].to_pylist(), dtype=np.float32)
    mask = np.asarray(
        table['observation.unified_128d_mask'].to_pylist(), dtype=np.bool_)
    target = np.asarray(
        table['target.unified_128d'].to_pylist(), dtype=np.float32)

    assert np.array_equal(observation[:, 0:14], state[:, 17:31])
    assert np.array_equal(observation[:, 28:30], state[:, 31:33])
    assert np.array_equal(observation[:, 94:97], state[:, 12:15])
    assert np.array_equal(observation[:, 98:100], state[:, 15:17])
    assert np.array_equal(observation[:, 103:109], state[:, 0:6])
    assert np.array_equal(observation[:, 109:115], state[:, 6:12])
    assert np.array_equal(target[:, 100:103], action[:, 31:34])
    assert np.array_equal(target[:, 115:121], action[:, 34:40])
    assert not observation[:, 121:128].any()
    assert not mask[:, 121:128].any()
    assert not mask[:, 100:103].any()
    assert not observation[:, 115:121].any()
    assert len(sorted((output / 'videos').rglob('*.mp4'))) == 2

    info = json.loads((output / 'meta/info.json').read_text())
    assert info['schema_id'] == 'oli_unified'
    assert info['embodiment_id'] == 'oli'
    assert info['robot_type'] == 'oli'
    assert info['ego2cam_contract']['camera_order'] == ['head', 'left_wrist']
    assert info['ego2cam_contract']['shape'] == [2, 4, 4]
    assert info['unified_layout']['dimension'] == UNIFIED_DIM
    assert info['features']['observation.ego2cam']['shape'] == [2, 4, 4]
    assert info['features']['observation.ego2cam_valid']['shape'] == [2]
