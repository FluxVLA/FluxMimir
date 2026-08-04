from abc import abstractmethod
from typing import Any, Mapping, Sequence

from .base import BaseBackend


class HandTrajectoryRetargeter(BaseBackend):

    @abstractmethod
    def retarget(self, hand_keypoints: Any,
                 context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a smoothed robot end-effector/gripper trajectory."""


class HumanRegionSegmenter(BaseBackend):

    @abstractmethod
    def segment(self, frames: Any,
                prompts: Sequence[str]) -> Mapping[str, Any]:
        """Return ``human_region_mask[T,H,W]`` and provenance."""


class VideoInpainter(BaseBackend):

    @abstractmethod
    def inpaint(self, frames: Any,
                human_region_mask: Any) -> Mapping[str, Any]:
        """Return temporally consistent clean-background frames."""


class SceneDepthEstimator(BaseBackend):

    @abstractmethod
    def estimate(self, frames: Any, camera: Mapping[str,
                                                    Any]) -> Mapping[str, Any]:
        """Return metric scene depth with explicit units and frame."""


class RobotKinematicsRenderer(BaseBackend):

    @abstractmethod
    def search_base(self, trajectory: Any,
                    robot: Mapping[str, Any]) -> Mapping[str, Any]:
        """Find a base pose using representative-frame IK feasibility."""

    @abstractmethod
    def solve(self, trajectory: Any, base_pose: Any,
              robot: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return joint trajectories and per-frame feasibility."""

    @abstractmethod
    def render(self, joints: Any, camera: Mapping[str, Any],
               robot: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return robot RGB, metric depth, and render provenance."""
