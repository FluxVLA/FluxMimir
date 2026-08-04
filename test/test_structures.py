import pytest

from fluxmimir.structures import (EpisodeRef, IssueSeverity, ValidationIssue,
                                  ValidationReport)


def test_episode_ref_requires_stable_identity():
    with pytest.raises(ValueError, match='episode_uid'):
        EpisodeRef(episode_uid='', source_uri='/raw/episode_0.hdf5')

    with pytest.raises(ValueError, match='source_uri'):
        EpisodeRef(episode_uid='episode-0', source_uri='')


def test_validation_report_only_errors_block_release():
    warning = ValidationIssue(
        code='missing_optional_depth',
        message='depth is unavailable',
        severity=IssueSeverity.WARNING,
    )
    assert ValidationReport(issues=(warning, )).ok is True

    error = ValidationIssue(
        code='target_semantics_unknown',
        message='issued target provenance is not confirmed',
    )
    assert ValidationReport(issues=(warning, error)).ok is False
