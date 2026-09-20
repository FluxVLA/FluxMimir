from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True)
class EpisodeRef:
    """Stable reference to one source episode.

    ``source_uri`` may be a local path or an object-store URI. Payload arrays
    and images deliberately stay outside this structure so discovery remains
    cheap.
    """

    episode_uid: str
    source_uri: str
    source_episode_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_uid:
            raise ValueError('episode_uid must not be empty')
        if not self.source_uri:
            raise ValueError('source_uri must not be empty')


@dataclass(frozen=True)
class ArtifactRecord:
    """Artifact emitted by a pipeline stage or writer."""

    artifact_type: str
    uri: str
    sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    """Common result envelope shared by conversion and synthesis stages."""

    value: Any = None
    artifacts: tuple[ArtifactRecord, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    skipped: bool = False


class IssueSeverity(str, Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.ERROR
    location: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.severity is IssueSeverity.ERROR
                       for issue in self.issues)


@dataclass(frozen=True)
class PipelineContext:
    """Runtime context passed to stages without owning data payloads."""

    run_id: str
    work_dir: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
