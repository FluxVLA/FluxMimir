from abc import ABC, abstractmethod
from typing import Any, Mapping

from fluxmimir.structures import ArtifactRecord, EpisodeRef


class BaseVideoEncoder(ABC):
    """Stream frames into a versioned video artifact."""

    @abstractmethod
    def encode(self, episode: EpisodeRef, frames: Any,
               metadata: Mapping[str, Any]) -> ArtifactRecord:
        """Encode frames without changing their semantic camera identity."""
