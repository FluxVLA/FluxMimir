from dataclasses import dataclass, field
from typing import Any, Mapping

from fluxmimir.structures import EpisodeRef


@dataclass
class EpisodeRecord:
    """Source-neutral episode IR.

    Field and media values may be lazy readers. Sources should not eagerly
    materialize complete videos unless an implementation explicitly requests
    it.
    """

    ref: EpisodeRef
    length: int
    fields: Mapping[str, Any]
    media: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError('episode length must be non-negative')
