from abc import abstractmethod
from typing import Any, Mapping

from fluxmimir.engines.stages import BaseStage
from fluxmimir.structures import PipelineContext, StageResult

from .structures import EpisodeRecord


class BaseEpisodeMapper(BaseStage):
    """Map a source-neutral episode into a versioned canonical contract."""

    @abstractmethod
    def map_episode(self, episode: EpisodeRecord,
                    context: PipelineContext) -> Mapping[str, Any]:
        """Return canonical fields without assigning storage paths."""

    def process(self, episode: Any, context: PipelineContext) -> StageResult:
        if not isinstance(episode, EpisodeRecord):
            raise TypeError(
                f'expected EpisodeRecord, got {type(episode).__name__}')
        return StageResult(value=self.map_episode(episode, context))
