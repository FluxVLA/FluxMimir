from abc import ABC, abstractmethod
from typing import Any

from fluxmimir.structures import ValidationReport


class BaseValidator(ABC):
    """Return structured issues; never silently repair source semantics."""

    @abstractmethod
    def validate(self, value: Any) -> ValidationReport:
        """Validate one episode, artifact, or complete dataset release."""
