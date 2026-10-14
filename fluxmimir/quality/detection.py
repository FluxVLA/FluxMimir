from __future__ import annotations

import numpy as np

DETECTOR_VERSION = '1.0.0'


def static_frame_mask(values: np.ndarray, valid: np.ndarray,
                      step_threshold: np.ndarray) -> np.ndarray:
    """Return the frames whose incoming step stays inside the noise floor."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError(f'values/valid must be matching 2-D arrays, '
                         f'got {values.shape} and {valid.shape}')
    static = np.zeros(values.shape[0], dtype=np.bool_)
    if values.shape[0] < 2:
        return static
    active = valid[1:] & valid[:-1]
    step = np.abs(np.diff(values, axis=0)) / step_threshold
    ratio = np.where(active, step, -np.inf).max(axis=1)
    static[1:] = active.any(axis=1) & (ratio <= 1.0)
    return static


def static_segments(static: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return the maximal inclusive ``(start, end)`` runs of static frames."""
    static = np.asarray(static, dtype=np.bool_)
    if static.ndim != 1:
        raise ValueError(f'static flags must be 1-D, got {static.shape}')
    padded = np.concatenate(([False], static, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))
