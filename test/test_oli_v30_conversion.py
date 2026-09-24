"""End-to-end coverage for the Oli LeRobot v2.1 to v3.0 layout config."""

import json
from pathlib import Path

import numpy as np
import pytest
from mmengine.config import Config
from test_oli_conversion import _write_oli_v2_dataset

from fluxmimir.engines import build_runner_from_cfg
from fluxmimir.schemas import UNIFIED_DIM

CONFIG = str(
    Path(__file__).resolve().parents[1] /
    'configs/conversion/oli_lerobot_v2_to_canonical_v30.py')


def test_oli_v30_end_to_end(tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')

    raw = tmp_path / 'raw'
    output = tmp_path / 'canonical'
    state, _ = _write_oli_v2_dataset(raw)
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
    table = pq.read_table(output / 'data/chunk-000/file-000.parquet')
    unified_type = table.schema.field('observation.unified_128d').type
    assert unified_type == pa.list_(pa.float32(), UNIFIED_DIM)
    observation = np.asarray(
        table['observation.unified_128d'].to_pylist(), dtype=np.float32)
    assert np.array_equal(observation[:, 103:109], state[:, 0:6])
    assert np.array_equal(observation[:, 109:115], state[:, 6:12])
    assert not observation[:, 121:128].any()

    episodes = pq.read_table(
        output / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert [int(row['episode_index']) for row in episodes] == [0]
    assert (output / 'meta/tasks.parquet').is_file()
    assert (output / 'meta/stats.json').is_file()
    assert not (output / 'meta/episodes.jsonl').exists()
    assert not (output / 'meta/tasks.jsonl').exists()
    videos = sorted((output / 'videos').rglob('*.mp4'))
    assert len(videos) == 2
    assert {path.name for path in videos} == {'file-000.mp4'}

    info = json.loads((output / 'meta/info.json').read_text(encoding='utf-8'))
    assert info['codebase_version'] == 'v3.0'
    assert info['ego2cam_contract']['camera_order'] == ['head', 'left_wrist']
    assert info['unified_layout']['dimension'] == UNIFIED_DIM
