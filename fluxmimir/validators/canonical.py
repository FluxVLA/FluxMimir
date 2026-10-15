from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from fluxmimir.engines.utils import VALIDATORS
from fluxmimir.schemas import PARQUET_COLUMNS, UNIFIED_DIM
from fluxmimir.structures import (IssueSeverity, ValidationIssue,
                                  ValidationReport)

from .base import BaseValidator


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _parquet_files(root: Path) -> list[Path]:
    return sorted((root / 'data').glob('chunk-*/episode_*.parquet'))


@VALIDATORS.register_module()
class SourceSemanticsValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        info = _json(root / 'meta/info.json')
        if info.get('training_ready'):
            return ValidationReport(
                issues=(ValidationIssue(
                    'unsafe_training_ready',
                    'unresolved source semantics cannot be training-ready',
                ), ))
        return ValidationReport(
            issues=(ValidationIssue(
                'semantic_placeholders_retained',
                'target/calibration/annotation blockers remain explicit',
                severity=IssueSeverity.WARNING,
            ), ))


@VALIDATORS.register_module()
class CanonicalParquetSchemaValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues: list[ValidationIssue] = []
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('validation requires pyarrow') from exc
        expected_types = (
            pa.list_(pa.float32(), UNIFIED_DIM),
            pa.list_(pa.bool_(), UNIFIED_DIM),
            pa.list_(pa.list_(pa.list_(pa.float32(), 4), 4), 3),
            pa.list_(pa.bool_(), 3),
            pa.list_(pa.float32(), UNIFIED_DIM),
            pa.list_(pa.bool_(), UNIFIED_DIM),
            pa.int8(),
            pa.float32(),
            pa.int64(),
            pa.int64(),
            pa.int64(),
            pa.int64(),
        )
        files = _parquet_files(root)
        if not files:
            issues.append(
                ValidationIssue('missing_parquet',
                                'dataset contains no Parquet files'))
        for path in files:
            table = pq.read_table(path)
            if tuple(table.column_names) != PARQUET_COLUMNS:
                issues.append(
                    ValidationIssue(
                        'parquet_columns',
                        'canonical column order mismatch',
                        location=str(path)))
                continue
            for field, expected in zip(table.schema, expected_types):
                if field.type != expected:
                    issues.append(
                        ValidationIssue(
                            'parquet_type',
                            f'{field.name}: {field.type} != {expected}',
                            location=str(path)))
                if table[field.name].null_count:
                    issues.append(
                        ValidationIssue(
                            'parquet_null',
                            f'{field.name} contains nulls',
                            location=str(path)))
            frame_index = table['frame_index'].to_numpy()
            if not np.array_equal(frame_index,
                                  np.arange(table.num_rows, dtype=np.int64)):
                issues.append(
                    ValidationIssue(
                        'frame_index',
                        'frame_index is not contiguous',
                        location=str(path)))
            timestamp = table['timestamp'].to_numpy()
            if len(timestamp) > 1 and np.any(np.diff(timestamp) <= 0):
                issues.append(
                    ValidationIssue(
                        'timestamp_order',
                        'timestamp is not strictly increasing',
                        location=str(path)))
        return ValidationReport(
            issues=tuple(issues), metrics={'parquet_files': float(len(files))})


@VALIDATORS.register_module()
class UnifiedMaskValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues: list[ValidationIssue] = []
        import pyarrow.parquet as pq
        for path in _parquet_files(root):
            table = pq.read_table(path)
            for name in ('observation.unified_128d', 'target.unified_128d'):
                values = np.asarray(table[name].to_pylist(), dtype=np.float32)
                mask = np.asarray(
                    table[f'{name}_mask'].to_pylist(), dtype=np.bool_)
                if not np.isfinite(values[mask]).all():
                    issues.append(
                        ValidationIssue(
                            'nonfinite_valid_value',
                            f'{name} has valid NaN/Inf',
                            location=str(path)))
                if np.any(values[~mask] != 0):
                    issues.append(
                        ValidationIssue(
                            'nonzero_placeholder',
                            f'{name} has nonzero masked data',
                            location=str(path)))
        return ValidationReport(issues=tuple(issues))


@VALIDATORS.register_module()
class EpisodeIndexValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues: list[ValidationIssue] = []
        import pyarrow.parquet as pq
        expected_global = 0
        for expected_episode, path in enumerate(_parquet_files(root)):
            table = pq.read_table(path)
            episodes = np.unique(table['episode_index'].to_numpy())
            if not np.array_equal(episodes, [expected_episode]):
                issues.append(
                    ValidationIssue(
                        'episode_index',
                        f'expected episode {expected_episode}',
                        location=str(path)))
            indices = table['index'].to_numpy()
            expected = np.arange(
                expected_global,
                expected_global + table.num_rows,
                dtype=np.int64)
            if not np.array_equal(indices, expected):
                issues.append(
                    ValidationIssue(
                        'global_index',
                        'global index is not contiguous',
                        location=str(path)))
            expected_global += table.num_rows
        return ValidationReport(issues=tuple(issues))


@VALIDATORS.register_module()
class ForeignKeyClosureValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues: list[ValidationIssue] = []
        tasks = {
            row['task_index']
            for row in _jsonl(root / 'meta/tasks.jsonl')
        }
        subtasks = {
            row['subtask_index']
            for row in _jsonl(root / 'meta/subtasks.jsonl')
        }
        episodes = _jsonl(root / 'meta/episodes.jsonl')
        for row in episodes:
            if row['task_index'] not in tasks:
                issues.append(
                    ValidationIssue('task_fk',
                                    f"unknown task {row['task_index']}"))
        import pyarrow.parquet as pq
        for path in _parquet_files(root):
            used = set(
                pq.read_table(path, columns=['subtask_index'
                                             ])['subtask_index'].to_pylist())
            if not used <= subtasks:
                issues.append(
                    ValidationIssue(
                        'subtask_fk',
                        f'unknown subtasks {sorted(used - subtasks)}',
                        location=str(path)))
        return ValidationReport(issues=tuple(issues))


@VALIDATORS.register_module()
class VideoAlignmentValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        info = _json(root / 'meta/info.json')
        video_keys = [
            key for key, spec in info['features'].items()
            if spec['dtype'] == 'video'
        ]
        files = list(
            (root /
             'videos').glob('chunk-*/observation.images.*/episode_*.mp4'))
        expected = info['total_episodes'] * len(video_keys)
        issues = ()
        if len(files) != expected or info['total_videos'] != expected:
            issues = (ValidationIssue(
                'video_count',
                f'expected {expected} videos, found {len(files)}',
            ), )
        return ValidationReport(
            issues=issues, metrics={'video_files': float(len(files))})


@VALIDATORS.register_module()
class MaskedStatisticsValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        issues: list[ValidationIssue] = []
        records = _jsonl(root / 'meta/episodes_stats.jsonl')
        import pyarrow.parquet as pq
        files = _parquet_files(root)
        if len(records) != len(files):
            return ValidationReport(
                issues=(ValidationIssue(
                    'stats_count', 'episode statistics count mismatch'), ))
        for record, path in zip(records, files):
            table = pq.read_table(path)
            for name in ('observation.unified_128d', 'target.unified_128d'):
                mask = np.asarray(
                    table[f'{name}_mask'].to_pylist(), dtype=np.bool_)
                actual = mask.sum(axis=0).tolist()
                stored = record['stats'][name]['count']
                if actual != stored:
                    issues.append(
                        ValidationIssue(
                            'masked_stats_count',
                            f'{name} counts differ',
                            location=str(path)))
        return ValidationReport(issues=tuple(issues))


@VALIDATORS.register_module()
class ArtifactManifestValidator(BaseValidator):

    def validate(self, value: Any) -> ValidationReport:
        root = Path(value)
        path = root / 'meta/artifacts.sha256.json'
        if not path.is_file():
            return ValidationReport(
                issues=(ValidationIssue('missing_manifest',
                                        'artifact manifest is missing'), ))
        manifest = _json(path)
        issues: list[ValidationIssue] = []
        actual_paths = set()
        for directory in ('data', 'videos', 'meta'):
            base = root / directory
            if base.exists():
                actual_paths.update(
                    item.relative_to(root).as_posix()
                    for item in base.rglob('*')
                    if item.is_file() and item != path)
        declared = set(manifest['artifacts'])
        if actual_paths != declared:
            issues.append(
                ValidationIssue(
                    'manifest_closure',
                    f'missing={sorted(actual_paths - declared)}, '
                    f'extra={sorted(declared - actual_paths)}'))
        for relative, record in manifest['artifacts'].items():
            artifact = root / relative
            if artifact.is_file() and _sha256(artifact) != record['sha256']:
                issues.append(
                    ValidationIssue('manifest_hash',
                                    f'hash mismatch for {relative}'))
        if manifest.get('artifact_count') != len(declared):
            issues.append(
                ValidationIssue('manifest_count',
                                'artifact_count does not match records'))
        return ValidationReport(issues=tuple(issues))
