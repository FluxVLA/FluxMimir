from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from fluxmimir.backends import BaseBackend


class BaseAnnotationProvider(BaseBackend, ABC):
    """Provider-neutral JSON response contract.

    Implementations resolve credentials from an environment-variable name or
    secret manager. Resolved secret values must never enter config snapshots,
    logs, caches, or annotation records.
    """

    @abstractmethod
    def complete_json(self, *, request_id: str,
                      messages: Sequence[Mapping[str, Any]],
                      response_schema: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return schema-validated JSON plus non-secret usage metadata."""
