from abc import ABC, abstractmethod
from typing import Any, Mapping

from fluxmimir.conversion import EpisodeRecord
from fluxmimir.structures import ArtifactRecord, PipelineContext, StageResult


class BaseWriter(ABC):
    """Transactional dataset writer contract."""

    def setup(self, context: PipelineContext) -> None:
        del context

    @abstractmethod
    def write_episode(self, episode: EpisodeRecord,
                      result: StageResult) -> tuple[ArtifactRecord, ...]:
        """Write one episode into a staging release."""

    @abstractmethod
    def finalize(self, metadata: Mapping[str, Any]) -> ArtifactRecord:
        """Validate and atomically publish the release manifest."""

    def close(self) -> None:
        """Release writer resources without publishing partial output."""
