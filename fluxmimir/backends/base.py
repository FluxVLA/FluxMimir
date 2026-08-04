from abc import ABC

from fluxmimir.structures import PipelineContext


class BaseBackend(ABC):
    """Lifecycle shared by replaceable external model/tool adapters."""

    def setup(self, context: PipelineContext) -> None:
        del context

    def close(self) -> None:
        """Release model, process, or remote-client resources."""
