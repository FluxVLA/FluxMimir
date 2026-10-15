import numpy as np

from fluxmimir.conversion import EpisodeRecord
from fluxmimir.embodiments import (AgileXAlohaAdapter, FrankaDualAdapter,
                                   ReBotDualAdapter)
from fluxmimir.schemas import CANONICAL_128D_SCHEMA, LAYOUT_RANGES, UNIFIED_DIM
from fluxmimir.structures import EpisodeRef


def test_layout_ranges_extend_the_legacy_prefix():
    assert UNIFIED_DIM == 128
    assert [(item.name, item.start, item.end)
            for item in LAYOUT_RANGES[:8]] == [
                ('arm_joint_position', 0, 14),
                ('end_effector_pose', 14, 28),
                ('gripper_position', 28, 30),
                ('left_dexterous_hand', 30, 62),
                ('right_dexterous_hand', 62, 94),
                ('waist_position', 94, 98),
                ('head_position', 98, 100),
                ('mobile_base_signal', 100, 103),
            ]
    assert [(item.name, item.start, item.end, item.unit)
            for item in LAYOUT_RANGES[8:]] == [
                ('left_leg_joint_position', 103, 109, 'rad'),
                ('right_leg_joint_position', 109, 115, 'rad'),
                ('mobile_base_rotation_6d', 115, 121, None),
                ('reserved', 121, 128, None),
            ]


def test_schema_features_use_the_unified_dimension():
    shapes = {
        feature.name: feature.shape
        for feature in CANONICAL_128D_SCHEMA.features
    }
    assert shapes['observation.unified_128d'] == (UNIFIED_DIM, )
    assert shapes['observation.unified_128d_mask'] == (UNIFIED_DIM, )
    assert shapes['target.unified_128d'] == (UNIFIED_DIM, )
    assert shapes['target.unified_128d_mask'] == (UNIFIED_DIM, )


def _aloha_episode(length: int = 2) -> EpisodeRecord:
    qpos = np.arange(length * 14, dtype=np.float32).reshape(length, 14) / 10
    return EpisodeRecord(
        ref=EpisodeRef('episode-1', '/raw/episode_1.hdf5'),
        length=length,
        fields={'observations/qpos': qpos},
    )


def _paired_episode(length: int = 2, width: int = 16) -> EpisodeRecord:
    values = np.arange(length * width, dtype=np.float32).reshape(length, width)
    return EpisodeRecord(
        ref=EpisodeRef('episode-1', '/raw/episode_1.hdf5'),
        length=length,
        fields={
            'observations/qpos': values,
            'action': values
        },
    )


def _assert_extended_range_inert(mapped) -> None:
    assert not np.asarray(mapped['observation.unified_128d'])[:, 103:].any()
    assert not np.asarray(mapped['observation.unified_128d_mask'])[:,
                                                                   103:].any()


def test_existing_profiles_leave_the_extended_range_inert():
    aloha = _aloha_episode()
    mapped = AgileXAlohaAdapter().map_observation(aloha)
    assert mapped['observation.unified_128d'].shape == (2, UNIFIED_DIM)
    assert mapped['observation.unified_128d_mask'][:, :6].all()
    assert mapped['observation.unified_128d_mask'][:, 7:13].all()
    _assert_extended_range_inert(mapped)

    for adapter, width in ((FrankaDualAdapter(), 16), (ReBotDualAdapter(),
                                                       14)):
        episode = _paired_episode(width=width)
        _assert_extended_range_inert(adapter.map_observation(episode))
        target = adapter.map_target(episode)
        values = np.asarray(target['target.unified_128d'])
        masks = np.asarray(target['target.unified_128d_mask'])
        assert not values[:, 103:].any() and not masks[:, 103:].any()
