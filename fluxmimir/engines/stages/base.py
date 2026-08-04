from abc import ABC, abstractmethod
from typing import Any

from fluxmimir.structures import PipelineContext, StageResult


class BaseStage(ABC):
    """Lifecycle contract for a single offline processing stage."""

    def setup(self, context: PipelineContext) -> None:
        del context

    @abstractmethod
    def process(self, episode: Any, context: PipelineContext) -> StageResult:
        """Process one episode without assuming a storage implementation."""

    def close(self) -> None:
        """Release backend resources."""
