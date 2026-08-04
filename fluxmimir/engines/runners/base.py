from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseRunner(ABC):
    """Top-level pipeline lifecycle contract."""

    def run_setup(self) -> None:
        """Allocate resources and run preflight checks."""

    @abstractmethod
    def run(self) -> Mapping[str, Any]:
        """Execute the configured pipeline."""

    def close(self) -> None:
        """Release resources after success or failure."""
