import json

import numpy as np
import pytest

from fluxmimir.embodiments import AgileXAlohaAdapter
from fluxmimir.engines import build_runner_from_cfg
from fluxmimir.quality import (StaticFrameCriteria, static_frame_mask,
                               static_segments)
from fluxmimir.quality.stages import FilterStaticFrames
from fluxmimir.schemas import UNIFIED_DIM
from fluxmimir.structures import PipelineContext

STEP_THRESHOLD = {'arm_joint_position': 1.0e-4, 'gripper_position': 1.5e-4}
GROUPS = ('arm_joint_position', 'gripper_position')
LENGTH = 180
HOLD_START = 60


def _episode(hold_frames=60, length=LENGTH, hold_start=HOLD_START):
    values = np.zeros((length, UNIFIED_DIM), dtype=np.float32)
    mask = np.zeros((length, UNIFIED_DIM), dtype=np.bool_)
    mask[:, 0:6] = True
    mask[:, 28:30] = True
    values[:, 0] = np.arange(length, dtype=np.float32) * 0.05
    hold_value = values[hold_start - 1, 0] + 0.3
    values[hold_start:hold_start + hold_frames, 0] = hold_value
    after = hold_start + hold_frames
    values[after:, 0] = (
        hold_value + 0.05 * np.arange(1, length - after + 1, dtype=np.float32))
    return {
        'observation.unified_128d': values,
        'observation.unified_128d_mask': mask,
        'target.unified_128d': np.zeros_like(values),
        'target.unified_128d_mask': np.zeros_like(mask),
        'timestamp': np.arange(length, dtype=np.float32) / 30.0,
        '_provenance': {
            'source_relative_path': 'collection/episode_0.hdf5'
        },
    }


def _criteria():
    return StaticFrameCriteria(
        step_threshold=STEP_THRESHOLD,
        groups=GROUPS,
        static_frame_threshold=50)


def _stage(tmp_path, policy):
    stage = FilterStaticFrames(
        embodiment=AgileXAlohaAdapter(),
        enabled=True,
        policy=policy,
        static_frame_threshold=50,
        signals=('observation', 'target'),
        groups=GROUPS,
        step_threshold=STEP_THRESHOLD,
        audit_name='static_frames.jsonl')
    context = PipelineContext(
        run_id='run-1', work_dir=str(tmp_path), metadata={'episode_index': 0})
    stage.setup(context)
    return stage, context


def _rows(tmp_path):
    path = tmp_path / 'static_frames.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_static_frame_mask_marks_held_frames():
    episode = _episode()

    static = static_frame_mask(episode['observation.unified_128d'],
                               episode['observation.unified_128d_mask'],
                               _criteria().thresholds())

    assert int(static.sum()) == 59
    assert static_segments(static) == ((61, 119), )


@pytest.mark.parametrize('hold_frames,frames', [(51, 50), (50, 49)])
def test_static_frame_mask_boundary(hold_frames, frames):
    episode = _episode(hold_frames=hold_frames)

    static = static_frame_mask(episode['observation.unified_128d'],
                               episode['observation.unified_128d_mask'],
                               _criteria().thresholds())

    assert static_segments(static) == ((61, 60 + frames), )


def test_static_frame_mask_ignores_invalid_dimension():
    episode = _episode()
    values = episode['observation.unified_128d']
    mask = episode['observation.unified_128d_mask'].copy()
    values[:, 1] = np.where(np.arange(len(values)) % 2 == 0, 1.0, -1.0)
    mask[:, 1] = False

    static = static_frame_mask(values, mask, _criteria().thresholds())

    assert int(static.sum()) == 59


def test_static_frame_mask_step_threshold_edge():
    threshold = 2.0**-10
    criteria = StaticFrameCriteria(
        step_threshold={'arm_joint_position': threshold},
        groups=('arm_joint_position', ),
        static_frame_threshold=50)
    values = np.zeros((10, UNIFIED_DIM), dtype=np.float32)
    mask = np.zeros((10, UNIFIED_DIM), dtype=np.bool_)
    mask[:, 0] = True
    values[:, 0] = np.arange(10, dtype=np.float32) * threshold

    static = static_frame_mask(values, mask, criteria.thresholds())
    assert int(static.sum()) == 9

    values[:, 0] = np.arange(10, dtype=np.float32) * threshold * 2
    static = static_frame_mask(values, mask, criteria.thresholds())
    assert not static.any()


def test_static_frame_mask_requires_valid_dimension():
    episode = _episode()

    static = static_frame_mask(
        episode['observation.unified_128d'],
        np.zeros_like(episode['observation.unified_128d_mask']),
        _criteria().thresholds())

    assert not static.any()


def test_filter_reports_episode_without_dropping(tmp_path):
    stage, context = _stage(tmp_path, policy='report')
    try:
        result = stage.process(_episode(), context)
    finally:
        stage.close()

    assert result.skipped is False
    assert result.metrics['max_static_window'] == 59.0
    rows = _rows(tmp_path)
    assert rows[1]['decision'] == 'kept'
    assert rows[1]['static_hit'] is True


def test_filter_drops_episode_with_long_static_segment(tmp_path):
    stage, context = _stage(tmp_path, policy='drop_episode')
    try:
        result = stage.process(_episode(), context)
    finally:
        stage.close()

    assert result.skipped is True
    assert '59' in result.warnings[0]
    assert '50' in result.warnings[0]
    assert _rows(tmp_path)[1]['decision'] == 'dropped'


def test_filter_requires_both_signals_static(tmp_path):
    episode = _episode()
    episode['target.unified_128d_mask'][:, 0:6] = True
    episode['target.unified_128d'][:, 0] = (
        np.arange(LENGTH, dtype=np.float32) * 0.05)
    stage, context = _stage(tmp_path, policy='drop_episode')
    try:
        result = stage.process(episode, context)
    finally:
        stage.close()

    assert result.skipped is False
    assert result.metrics['max_static_window'] == 0.0


def test_criteria_rejects_unknown_group():
    with pytest.raises(ValueError, match='unknown layout groups'):
        StaticFrameCriteria(
            step_threshold={'am_joint_position': 1.0e-4},
            groups=('am_joint_position', ),
            static_frame_threshold=50)


def test_filter_rejects_episode_without_valid_dimensions(tmp_path):
    episode = _episode()
    episode['observation.unified_128d_mask'] = np.zeros_like(
        episode['observation.unified_128d_mask'])
    stage, context = _stage(tmp_path, policy='report')
    try:
        with pytest.raises(ValueError, match='valid dimension'):
            stage.process(episode, context)
    finally:
        stage.close()


def _write_episode(h5py, path, frozen_frames, length=80):
    qpos = np.zeros((length, 14), dtype=np.float32)
    if frozen_frames:
        moving = np.arange(length - frozen_frames, dtype=np.float32)
        qpos[frozen_frames:] = ((frozen_frames + 1 + moving)[:, None] * 0.05)
    else:
        qpos[:] = np.arange(length, dtype=np.float32)[:, None] * 0.05
    with h5py.File(path, 'w') as stream:
        stream.create_dataset('observations/qpos', data=qpos)
        stream.create_dataset('action', data=qpos)
        for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            stream.create_dataset(
                f'observations/images/{camera}',
                data=np.zeros((length, 4, 6, 3), dtype=np.uint8))


def test_static_frame_filter_end_to_end(tmp_path):
    h5py = pytest.importorskip('h5py')
    pq = pytest.importorskip('pyarrow.parquet', reason='pyarrow is optional')
    pytest.importorskip('av', reason='PyAV is part of the conversion extra')

    raw = tmp_path / 'raw' / 'collection'
    raw.mkdir(parents=True)
    _write_episode(h5py, raw / 'episode_0.hdf5', frozen_frames=60)
    _write_episode(h5py, raw / 'episode_1.hdf5', frozen_frames=0)

    output = tmp_path / 'canonical'
    work_dir = tmp_path / 'work'
    runner = build_runner_from_cfg(
        dict(
            type='PipelineRunner',
            work_dir=str(work_dir),
            source=dict(type='AlohaHDF5Source', root=str(tmp_path / 'raw')),
            embodiment=dict(
                type='AgileXAlohaAdapter', action_semantics='unknown'),
            stages=[
                dict(type='ProbeSourceSchema'),
                dict(type='MapAlohaToCanonical128D'),
                dict(
                    type='FilterStaticFrames',
                    enabled=True,
                    policy='drop_episode',
                    static_frame_threshold=50,
                    signals=('observation', 'target'),
                    groups=GROUPS,
                    step_threshold=STEP_THRESHOLD,
                    audit_name='static_frames.jsonl'),
                dict(type='EncodeEpisodeVideos', enabled=True, codec='h264'),
                dict(type='ComputeMaskedEpisodeStatistics'),
                dict(type='BuildCanonicalMetadata'),
            ],
            writer=dict(
                type='CanonicalDatasetWriter',
                root=str(output),
                dataset_id='unit_test',
                dataset_version='0.0.1'),
            validators=[
                dict(type='CanonicalParquetSchemaValidator'),
                dict(type='EpisodeIndexValidator'),
                dict(type='MaskedStatisticsValidator'),
                dict(type='ArtifactManifestValidator'),
            ],
        ))
    try:
        summary = runner.run()
    finally:
        runner.close()

    assert summary['discovered_episodes'] == 2
    assert summary['episodes'] == 1
    assert summary['skipped_episodes'] == 1
    parquet_files = sorted((output / 'data').rglob('episode_*.parquet'))
    assert len(parquet_files) == 1
    table = pq.read_table(parquet_files[0])
    assert table.num_rows == 80
    assert np.array_equal(table['frame_index'].to_numpy(),
                          np.arange(80, dtype=np.int64))
    assert table['episode_index'].to_pylist() == [0] * 80
    assert len(list((output / 'videos').rglob('*.mp4'))) == 3
    episodes = [
        json.loads(line)
        for line in (output / 'meta/episodes.jsonl').read_text().splitlines()
    ]
    assert len(episodes) == 1
    assert episodes[0]['episode_index'] == 0
    assert episodes[0]['length'] == 80
    assert episodes[0]['source_episode_index'] == 1
    info = json.loads((output / 'meta/info.json').read_text())
    assert info['total_episodes'] == 1
    rows = _rows(work_dir)
    assert len(rows) == 3
    assert rows[0]['record'] == 'header'
    assert rows[1]['episode_index'] is None
    assert [row['decision'] for row in rows[1:]] == ['dropped', 'kept']
