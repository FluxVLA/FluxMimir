from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fluxmimir.structures import PipelineContext, StageResult

from ..utils import (RUNNERS, build_embodiment_from_cfg, build_source_from_cfg,
                     build_stage_from_cfg, build_validator_from_cfg,
                     build_writer_from_cfg)
from .base import BaseRunner


@RUNNERS.register_module()
class PipelineRunner(BaseRunner):
    """Execute a source -> stages -> writer canonical conversion."""

    def __init__(
        self,
        source: Mapping[str, Any] | Any,
        embodiment: Mapping[str, Any] | Any,
        stages: Sequence[Mapping[str, Any] | Any],
        writer: Mapping[str, Any] | Any,
        validators: Sequence[Mapping[str, Any] | Any] = (),
        work_dir: str | Path = 'work_dirs/conversion',
        resume: bool = True,
        overwrite: bool = False,
        atomic_episode_commit: bool = True,
    ) -> None:
        if overwrite:
            raise ValueError('destructive overwrite is unsupported; choose a '
                             'new output root')
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.resume = bool(resume)
        self.overwrite = bool(overwrite)
        self.atomic_episode_commit = bool(atomic_episode_commit)
        self.source = (
            build_source_from_cfg(source)
            if isinstance(source, Mapping) else source)
        self.embodiment = (
            build_embodiment_from_cfg(embodiment) if isinstance(
                embodiment, Mapping) else embodiment)
        self.stages = []
        for stage in stages:
            if not isinstance(stage, Mapping):
                self.stages.append(stage)
            elif stage.get('type') == 'MapAlohaToCanonical107D':
                self.stages.append(
                    build_stage_from_cfg(
                        stage, default_args={'embodiment': self.embodiment}))
            else:
                self.stages.append(build_stage_from_cfg(stage))
        self.writer = (
            build_writer_from_cfg(writer)
            if isinstance(writer, Mapping) else writer)
        self.validators = [(build_validator_from_cfg(item) if isinstance(
            item, Mapping) else item) for item in validators]
        self._context: PipelineContext | None = None

    def run_setup(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        self._context = PipelineContext(
            run_id=run_id,
            work_dir=str(self.work_dir),
            metadata={
                'resume': self.resume,
                'overwrite': self.overwrite,
                'atomic_episode_commit': self.atomic_episode_commit,
                'converter_version': '0.2.0',
            },
        )
        for stage in self.stages:
            stage.setup(self._context)
        self.writer.setup(self._context)

    def _episode_context(self, episode_index: int,
                         global_index_start: int) -> PipelineContext:
        assert self._context is not None
        metadata = dict(self._context.metadata)
        metadata.update({
            'episode_index': episode_index,
            'global_index_start': global_index_start,
        })
        return PipelineContext(
            run_id=self._context.run_id,
            work_dir=self._context.work_dir,
            metadata=metadata,
        )

    def run(self) -> Mapping[str, Any]:
        if self._context is None:
            self.run_setup()
        assert self._context is not None
        references = list(self.source.discover())
        global_index = 0
        artifact_count = 0
        warning_count = 0
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(references, desc='Converting episodes', unit='ep')
        except ImportError:  # pragma: no cover
            iterator = references
        for episode_index, reference in enumerate(iterator):
            episode = self.source.open_episode(reference)
            context = self._episode_context(episode_index, global_index)
            value: Any = episode
            warnings: list[str] = []
            provenance: dict[str, Any] = {}
            for stage in self.stages:
                stage_result = stage.process(value, context)
                if not isinstance(stage_result, StageResult):
                    raise TypeError(f'{type(stage).__name__}.process must '
                                    'return StageResult')
                value = stage_result.value
                warnings.extend(stage_result.warnings)
                provenance.update(stage_result.provenance)
            final_result = StageResult(
                value=value, warnings=tuple(warnings), provenance=provenance)
            artifacts = self.writer.write_episode(episode, final_result)
            artifact_count += len(artifacts)
            warning_count += len(warnings)
            global_index += episode.length

        final_metadata = {
            **self._context.metadata,
            'run_id':
            self._context.run_id,
            'source_root':
            str(self.source.root),
            'fps':
            getattr(self.embodiment, 'nominal_fps', 30),
            'action_semantics':
            getattr(self.embodiment, 'action_semantics', 'unknown'),
        }
        manifest = self.writer.finalize(final_metadata)
        reports = [
            validator.validate(self.writer.root)
            for validator in self.validators
        ]
        errors = [
            issue for report in reports for issue in report.issues
            if issue.severity.value == 'error'
        ]
        if errors:
            summary = '; '.join(f'{issue.code}: {issue.message}'
                                for issue in errors[:10])
            raise RuntimeError(f'output validation failed: {summary}')
        return {
            'run_id': self._context.run_id,
            'output_root': str(self.writer.root),
            'episodes': len(references),
            'frames': global_index,
            'artifacts': artifact_count + 1,
            'warnings': warning_count,
            'manifest': manifest.uri,
        }

    def close(self) -> None:
        for stage in reversed(self.stages):
            stage.close()
        self.writer.close()
