from abc import ABC, abstractmethod
from collections.abc import Iterable

from fluxmimir.conversion import EpisodeRecord
from fluxmimir.structures import EpisodeRef


class BaseSource(ABC):
    """Physical data source that does not assign canonical semantics."""

    @abstractmethod
    def discover(self) -> Iterable[EpisodeRef]:
        """Yield lightweight episode references."""

    @abstractmethod
    def open_episode(self, episode: EpisodeRef) -> EpisodeRecord:
        """Open one episode, preferably using lazy or streaming payloads."""
