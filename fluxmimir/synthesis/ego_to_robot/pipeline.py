from enum import Enum


class EgoToRobotStage(str, Enum):
    RETARGET_TRAJECTORY = 'retarget_trajectory'
    SEGMENT_HUMAN_REGION = 'segment_human_region'
    INPAINT_BACKGROUND = 'inpaint_background'
    SEARCH_ROBOT_BASE = 'search_robot_base'
    RENDER_ROBOT = 'render_robot'
    ESTIMATE_SCENE_DEPTH = 'estimate_scene_depth'
    COMPOSITE_WITH_OCCLUSION = 'composite_with_occlusion'


DEFAULT_STAGE_ORDER = tuple(stage.value for stage in EgoToRobotStage)

HUMAN_REGION_MASK_KEY = 'human_region_mask'
ACTION_VALIDITY_MASK_KEY = 'action_validity_mask'
