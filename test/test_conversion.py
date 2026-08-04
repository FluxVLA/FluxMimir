import numpy as np
import pytest

from fluxmimir.conversion import BaseEpisodeMapper, EpisodeRecord
from fluxmimir.engines import build_runner_from_cfg
from fluxmimir.structures import EpisodeRef, PipelineContext


class UnitTestMapper(BaseEpisodeMapper):

    def map_episode(self, episode, context):
        return {
            'episode_uid': episode.ref.episode_uid,
            'run_id': context.run_id,
        }


def test_episode_mapper_bridges_ir_into_stage_result():
    episode = EpisodeRecord(
        ref=EpisodeRef('episode-1', '/raw/episode_1.hdf5'),
        length=2,
        fields={'observations/qpos': object()},
    )
    context = PipelineContext(run_id='run-1', work_dir='/tmp/run-1')

    result = UnitTestMapper().process(episode, context)

    assert result.value == {
        'episode_uid': 'episode-1',
        'run_id': 'run-1',
    }


def test_hdf_to_canonical_parquet_end_to_end(tmp_path):
    h5py = pytest.importorskip('h5py')
    pq = pytest.importorskip('pyarrow.parquet', reason='pyarrow is optional')
    pytest.importorskip('av', reason='PyAV is part of the conversion extra')

    raw = tmp_path / 'raw' / 'collection'
    raw.mkdir(parents=True)
    source_path = raw / 'episode_0.hdf5'
    qpos = np.arange(42, dtype=np.float32).reshape(3, 14) / 10
    with h5py.File(source_path, 'w') as stream:
        stream.create_dataset('observations/qpos', data=qpos)
        stream.create_dataset('action', data=qpos + 1)
        for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            stream.create_dataset(
                f'observations/images/{camera}',
                data=np.zeros((3, 4, 6, 3), dtype=np.uint8),
            )

    output = tmp_path / 'canonical'
    runner = build_runner_from_cfg(
        dict(
            type='PipelineRunner',
            work_dir=str(tmp_path / 'work'),
            source=dict(type='AlohaHDF5Source', root=str(tmp_path / 'raw')),
            embodiment=dict(
                type='AgileXAlohaAdapter', action_semantics='unknown'),
            stages=[
                dict(type='ProbeHDF5Schema'),
                dict(type='MapAlohaToCanonical107D'),
                dict(type='EncodeEpisodeVideos', enabled=True, codec='h264'),
                dict(type='ComputeMaskedEpisodeStatistics'),
                dict(type='BuildCanonicalMetadata'),
            ],
            writer=dict(
                type='CanonicalDatasetWriter',
                root=str(output),
                dataset_id='unit_test',
                dataset_version='0.0.1',
            ),
            validators=[
                dict(type='CanonicalParquetSchemaValidator'),
                dict(type='ForeignKeyClosureValidator'),
                dict(type='MaskedStatisticsValidator'),
                dict(type='ArtifactManifestValidator'),
            ],
        ))
    try:
        summary = runner.run()
    finally:
        runner.close()

    assert summary['episodes'] == 1
    assert summary['frames'] == 3
    table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
    assert table.num_rows == 3
    observation = np.asarray(
        table['observation.unified_107d'].to_pylist(), dtype=np.float32)
    mask = np.asarray(table['observation.unified_107d_mask'].to_pylist())
    target_mask = np.asarray(table['target.unified_107d_mask'].to_pylist())
    assert np.allclose(observation[:, :6], qpos[:, :6])
    assert np.allclose(observation[:, 7:13], qpos[:, 7:13])
    assert np.allclose(observation[:, 28], qpos[:, 6])
    assert np.allclose(observation[:, 29], qpos[:, 13])
    assert mask[:, :6].all() and mask[:, 7:13].all()
    assert mask[:, 28:30].all() and not mask[:, 14:28].any()
    assert not target_mask.any()
    assert table['target.source'].to_pylist() == [5, 5, 5]
    assert (output / 'meta/info.json').is_file()
    assert (output / 'meta/artifacts.sha256.json').is_file()
    assert len(list((output / 'videos').rglob('*.mp4'))) == 3
