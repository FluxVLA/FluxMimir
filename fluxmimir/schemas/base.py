from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    required: bool = True


@dataclass(frozen=True)
class DatasetSchema:
    schema_id: str
    schema_version: str
    layout_id: str
    features: tuple[FeatureSpec, ...]
