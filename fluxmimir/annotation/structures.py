from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AnnotationRequest:
    request_id: str
    episode_uid: str
    schema_id: str
    media_uris: tuple[str, ...]
    context: Mapping[str, Any] = field(default_factory=dict)
    privileged_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError('request_id must not be empty')
        if not self.schema_id:
            raise ValueError('schema_id must not be empty')


@dataclass(frozen=True)
class AnnotationRecord:
    request_id: str
    schema_id: str
    schema_version: str
    payload: Mapping[str, Any]
    provider: str
    model: str
    prompt_version: str
    confidence: float | None = None
    raw_response_uri: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
