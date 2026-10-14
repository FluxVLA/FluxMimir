from fluxmimir.schemas import (CANONICAL_107D_SCHEMA, LAYOUT_RANGES,
                               PARQUET_COLUMNS, REBOT_DUAL_107D_SCHEMA,
                               REBOT_DUAL_LAYOUT_ID, UNIFIED_DIM,
                               aloha_qpose_indices, franka_qpose_indices)


def test_canonical_schema_has_exact_physical_columns():
    assert len(PARQUET_COLUMNS) == 12
    assert tuple(feature.name
                 for feature in CANONICAL_107D_SCHEMA.features) == (
                     PARQUET_COLUMNS)


def test_unified_layout_is_contiguous_and_107d():
    assert LAYOUT_RANGES[0].start == 0
    assert all(left.end == right.start
               for left, right in zip(LAYOUT_RANGES, LAYOUT_RANGES[1:]))
    assert LAYOUT_RANGES[-1].end == UNIFIED_DIM
    assert sum(field.size for field in LAYOUT_RANGES) == UNIFIED_DIM


def test_aloha_qpose_view_uses_gripper_slots_not_reserved_joints():
    indices = aloha_qpose_indices()
    assert len(indices) == 14
    assert indices == (0, 1, 2, 3, 4, 5, 28, 7, 8, 9, 10, 11, 12, 29)
    assert 6 not in indices
    assert 13 not in indices


def test_rebot_schema_has_distinct_versioned_layout():
    assert REBOT_DUAL_107D_SCHEMA.schema_id == 'rebot_dual_unified'
    assert REBOT_DUAL_107D_SCHEMA.layout_id == REBOT_DUAL_LAYOUT_ID
    assert tuple(feature.name
                 for feature in REBOT_DUAL_107D_SCHEMA.features) == (
                     PARQUET_COLUMNS)


def test_franka_qpose_view_uses_seventh_joint_slots():
    indices = franka_qpose_indices()
    assert len(indices) == 16
    assert indices == (0, 1, 2, 3, 4, 5, 6, 28, 7, 8, 9, 10, 11, 12, 13, 29)
    assert 6 in indices
    assert 13 in indices
