from abc import ABC, abstractmethod
from typing import Any, Mapping

from fluxmimir.conversion.structures import EpisodeRecord
from fluxmimir.structures import ValidationReport


class BaseEmbodimentAdapter(ABC):
    """Own robot-specific ordering, units, and validity semantics."""

    @abstractmethod
    def preflight(self, episode: EpisodeRecord) -> ValidationReport:
        """Reject unknown order, units, timing, or target provenance."""

    @abstractmethod
    def map_observation(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        """Map observed values and explicit per-dimension validity."""

    @abstractmethod
    def map_target(self, episode: EpisodeRecord) -> Mapping[str, Any]:
        """Map issued targets only when source semantics are verified."""
