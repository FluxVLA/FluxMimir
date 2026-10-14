from .criteria import StaticFrameCriteria
from .detection import DETECTOR_VERSION, static_frame_mask, static_segments
from .stages import FilterStaticFrames

__all__ = [
    'DETECTOR_VERSION', 'FilterStaticFrames', 'StaticFrameCriteria',
    'static_frame_mask', 'static_segments'
]
