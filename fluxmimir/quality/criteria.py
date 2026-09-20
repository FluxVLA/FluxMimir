from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fluxmimir.schemas import LAYOUT_RANGES, UNIFIED_DIM


@dataclass(frozen=True)
class StaticFrameCriteria:
    """Static frame thresholds for one canonical layout selection."""

    step_threshold: Mapping[str, float]
    groups: tuple[str, ...]
    static_frame_threshold: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'step_threshold',
                           MappingProxyType(dict(self.step_threshold)))
        object.__setattr__(self, 'groups', tuple(self.groups))
        if not self.groups:
            raise ValueError('groups must not be empty')
        if len(set(self.groups)) != len(self.groups):
            raise ValueError(f'duplicate layout groups: {self.groups}')
        known = {item.name for item in LAYOUT_RANGES}
        unknown = sorted(set(self.groups) - known)
        if unknown:
            raise ValueError(f'unknown layout groups: {unknown}')
        if set(self.step_threshold) != set(self.groups):
            raise ValueError('step_threshold keys must match the groups')
        for name in self.groups:
            value = self.step_threshold[name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f'step threshold must be finite and positive: {name}')
        if (not isinstance(self.static_frame_threshold, int)
                or isinstance(self.static_frame_threshold, bool)):
            raise ValueError('static_frame_threshold must be an int')
        if self.static_frame_threshold < 1:
            raise ValueError('static_frame_threshold must be positive')

    def thresholds(self) -> np.ndarray:
        """Return the per-dimension noise floor over the unified layout."""
        values = np.full(UNIFIED_DIM, np.inf, dtype=np.float64)
        for item in LAYOUT_RANGES:
            if item.name in self.step_threshold:
                values[item.start:item.end] = self.step_threshold[item.name]
        return values

    def record(self) -> dict[str, Any]:
        """Return the JSON-serializable threshold record for the audit."""
        groups = [{
            'name': item.name,
            'range': [item.start, item.end],
            'unit': item.unit,
            'step_threshold': self.step_threshold[item.name],
        } for item in LAYOUT_RANGES if item.name in self.step_threshold]
        return {
            'groups': groups,
            'static_frame_threshold': self.static_frame_threshold,
        }

    def fingerprint(self) -> str:
        """Return a stable hash of the resolved thresholds."""
        payload = json.dumps(
            self.record(), sort_keys=True,
            separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()
