from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from fluxmimir.engines.utils import VALIDATORS
from fluxmimir.structures import ValidationIssue, ValidationReport

from .base import BaseValidator
from .canonical import (CanonicalParquetSchemaValidator, EpisodeIndexValidator,
                        UnifiedMaskValidator)

DATA_PATTERN = 'chunk-*/file-*.parquet'
EPISODES_PATTERN = 'meta/episodes/chunk-*/file-*.parquet'
VIDEO_PATTERN = 'observation.images.*/chunk-*/file-*.mp4'


def _issue(code: str,
           message: str,
           path: Path | None = None) -> ValidationIssue:
    return ValidationIssue(
        code, message, location=None if path is None else str(path))


def _shared_issues(root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for validator in (CanonicalParquetSchemaValidator(DATA_PATTERN),
                      EpisodeIndexValidator(DATA_PATTERN),
                      UnifiedMaskValidator(DATA_PATTERN)):
        issues.extend(validator.validate(root).issues)
    return issues


def _metadata_issues(root: Path, files: list[Path]) -> list[ValidationIssue]:
    import pyarrow.parquet as pq
    issues: list[ValidationIssue] = []
    rows = [row for path in files for row in pq.read_table(path).to_pylist()]
    if len(rows) != len(files):
        issues.append(_issue('episodes_rows', 'expected one row per file'))
    starts = list(itertools.accumulate([0] + [row['length'] for row in rows]))
    expected = {
        'episode_index': list(range(len(rows))),
        'data/chunk_index':
        [int(path.parent.name.split('-')[1]) for path in files],
        'data/file_index': [int(path.stem.split('-')[1]) for path in files],
        'dataset_from_index': starts[:-1],
        'dataset_to_index': starts[1:],
    }
    issues.extend(
        _issue('episode_metadata', f'{name} differs from the file layout')
        for name, wanted in expected.items()
        if [row[name] for row in rows] != wanted)
    tasks_path = root / 'meta/tasks.parquet'
    if not tasks_path.is_file():
        return issues + [_issue('missing_tasks', 'no meta/tasks.parquet')]
    tasks = set(pq.read_table(tasks_path)['task'].to_pylist())
    issues.extend(
        _issue('task_fk', f'unknown tasks {sorted(set(row["tasks"]) - tasks)}')
        for row in rows if set(row['tasks']) - tasks)
    return issues


def _statistics_issues(root: Path, files: list[Path]) -> list[ValidationIssue]:
    import pyarrow.parquet as pq
    path = root / 'meta/stats.json'
    if not path.is_file():
        return [_issue('missing_stats', 'no meta/stats.json')]
    stored = json.loads(path.read_text(encoding='utf-8'))
    totals: dict[str, np.ndarray] = {}
    for data_path in files:
        table = pq.read_table(data_path)
        for name in ('observation.unified_128d', 'target.unified_128d'):
            mask = np.asarray(
                table[f'{name}_mask'].to_pylist(), dtype=np.bool_)
            totals[name] = totals.get(name, 0) + mask.sum(axis=0)
    return [
        _issue('statistics_count', f'{name} count differs from the masks')
        for name, counts in totals.items()
        if stored.get(name, {}).get('count') != counts.tolist()
    ]


@VALIDATORS.register_module()
class CanonicalV30LayoutValidator(BaseValidator):
    """Validate the canonical content stored in the LeRobot v3.0 layout."""

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        info = json.loads(
            (root / 'meta/info.json').read_text(encoding='utf-8'))
        data_files = sorted((root / 'data').glob(DATA_PATTERN))
        episodes = sorted(root.glob(EPISODES_PATTERN))
        issues = _shared_issues(root)
        if info.get('codebase_version') != 'v3.0':
            issues.append(_issue('codebase_version', 'expected v3.0'))
        if len(episodes) != len(data_files) or len(episodes) != info.get(
                'total_episodes'):
            issues.append(_issue('episodes_count', 'episode file count'))
        issues.extend(_metadata_issues(root, episodes))
        issues.extend(_statistics_issues(root, data_files))
        features = info['features']
        videos = [k for k in features if features[k]['dtype'] == 'video']
        found = len(list((root / 'videos').glob(VIDEO_PATTERN)))
        expected = len(data_files) * len(videos)
        if found != expected:
            issues.append(_issue('video_alignment', f'{found} != {expected}'))
        return ValidationReport(issues=tuple(issues))
