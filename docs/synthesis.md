# Ego-to-Robot Synthesis

> **Planning status:** this document defines an intended contract. FluxMimir
> does not currently implement, bundle, download, or host SAM3, ProPainter,
> Depth Anything v3, MuJoCo-based IK/rendering, or any equivalent backend.
> Naming a backend below is a reference to the Qwen-RobotManip recipe, not a
> claim that the backend is available.

## Scope and ownership

FluxMimir is intended to own offline data synthesis and the artifacts needed to
audit it. It does not own policy training, inference, checkpoints, or online
robot control. The source recording remains immutable. A synthesis run writes
new, versioned artifacts and a manifest; it must not edit the source episode in
place.

The target output is an ego demonstration in which the visible human hand and
arm have been removed and a kinematically valid robot has been rendered in the
same scene. This is not a direct “turn a mask into a robot” operation. The mask
is used to remove the human pixels, while robot motion, geometry, camera
projection, depth, and occlusion are produced by separate stages.

## Reference pipeline

Figure 1 of the Qwen-RobotManip technical report presents six numbered blocks:
input, retarget and smooth, arm segmentation, hand removal, base search and IK,
and depth composite. FluxMimir plans the same method as seven executable stages
so that rendering and depth estimation can be cached and validated separately:

1. **Validate and normalize input.** Read synchronized RGB frames, timestamps
   or FPS, hand poses, handedness, and camera calibration or trajectory. Hand
   poses should be expressible as MANO parameters and/or 21-keypoint tracks.
   Validate the target robot description, including URDF or MJCF, joint limits,
   base and TCP/EEF conventions, gripper limits, reach, and render assets.
2. **Retarget and smooth motion.** Map human keypoints to a parallel-jaw EEF
   pose and gripper width. The report constructs a virtual finger as
   `0.7 * index_tip + 0.3 * middle_tip`, places the EEF at the midpoint between
   it and the thumb tip, and uses their distance as gripper width. A
   handedness-aware orthonormal frame supplies orientation. Position and width
   are smoothed with Savitzky-Golay filtering; orientation is smoothed with
   Gaussian-weighted SLERP. Degenerate frames and missing hands must be marked
   invalid rather than silently filled with plausible-looking values.
3. **Segment the human region.** A text-prompted segmenter (SAM3 in the report)
   produces a temporal binary mask for the human hand and arm. Prompts, model
   identity, checkpoint identity, thresholds, and any manual corrections are
   provenance, not hidden implementation details.
4. **Remove the human region.** A temporal video inpainter (ProPainter in the
   report) uses the RGB sequence and human-region masks to produce a clean
   background sequence. Processing frames independently is not equivalent: the
   backend must expose or preserve temporal consistency.
5. **Search the robot base and solve IK.** Select representative trajectory
   keyframes, including spatial extremes. Search candidate base placements
   around the trajectory and score each placement by the fraction of keyframes
   with feasible IK. Then solve the full smoothed trajectory in the target
   robot model. The search is embodiment-specific and must record joint-limit,
   collision, residual, and infeasibility diagnostics.
6. **Render the robot and estimate scene depth.** Render robot color and robot
   depth from the input camera using the solved joint trajectory (MuJoCo in the
   report). Independently estimate scene depth for the original/clean scene
   (Depth Anything v3 in the report). Intrinsics, extrinsics, image dimensions,
   depth convention, and metric scale must agree before compositing.
7. **Depth-aware composite.** Construct a robot visibility/occlusion mask using
   the report's rule `M_occ = 1[D_robot <= D_scene]`, then composite
   `I_syn = M_occ * I_robot + (1 - M_occ) * I_clean`. Undefined depth, render
   background pixels, near/far clipping, and tolerance at contact boundaries
   need explicit policies. The synthetic RGB, robot state/action trajectory,
   validity signals, and audit artifacts are written together.

Action-speed alignment is a separate, configurable post-stage. The report uses
source-specific frame subsampling because human motion is faster than robot
teleoperation data. FluxMimir must record the original and output timestamps and
must never imply that one hard-coded subsampling factor is valid for every
source.

## Mask vocabulary

The report uses “mask” for unrelated tensors. FluxMimir contracts must use
specific names and shapes:

- `human_region_mask: bool[T, H, W]` is an image-space segmentation artifact.
  It selects human hand/arm pixels for removal. The depth compositor's
  `robot_visibility_mask: bool[T, H, W]` is also image-space, but is a distinct
  derived artifact and must not overwrite the human mask.
- `action_validity_mask: bool[T, D]` is a supervision signal, not an image. In
  the report it is the logical AND of the embodiment's populated dimension-slot
  mask, the per-step validity mask, and per-hand visibility. Once a step is
  invalidated, subsequent steps may also be invalidated to preserve causal
  consistency.

Code, schema fields, artifact names, and metrics must not expose an unqualified
field named simply `mask`.

## Planned backend boundary

Backends are replaceable adapters. A runner should depend on typed inputs and
outputs rather than importing a research repository inside pipeline logic. The
following interfaces are proposed; they are not implemented yet:

```python
class BaseBackend:
    def setup(self, context): ...
    def capabilities(self) -> dict: ...
    def close(self): ...

class HandRetargeter(BaseBackend):
    def retarget(self, hand_tracks, robot_spec, camera) -> StageResult: ...

class ArmSegmenter(BaseBackend):
    def segment(self, frames, prompts) -> StageResult: ...

class VideoInpainter(BaseBackend):
    def inpaint(self, frames, human_region_masks) -> StageResult: ...

class KinematicsBackend(BaseBackend):
    def search_base(self, eef_trajectory, robot_spec) -> StageResult: ...
    def solve(self, eef_trajectory, base_pose, robot_spec) -> StageResult: ...

class RobotRenderer(BaseBackend):
    def render(self, joint_trajectory, robot_spec, camera) -> StageResult: ...

class DepthEstimator(BaseBackend):
    def estimate(self, frames, camera) -> StageResult: ...
```

`StageResult` should carry artifact references, typed metrics, warnings, and a
machine-readable status/reason code. A backend may eventually be implemented by
a local Python library, isolated worker, command-line tool, or approved remote
service; those execution choices must not change the data contract. Setup and
close must be idempotent. Each stage should be resumable from verified artifacts
and safe to run in a spawned worker.

Configuration should select a backend by registry name and non-secret
parameters. Model weights, robot assets, prompts, and thresholds are explicit
references. Credentials are never backend parameters; see
[security.md](security.md).

## Quality gates

Thresholds are dataset- and embodiment-specific configuration, not universal
constants. Every rejection must preserve a reason code and metrics in the run
manifest. At minimum, a future implementation should gate on:

- frame/timestamp synchronization, camera calibration completeness, hand-track
  coverage, and finite input values;
- retarget continuity, valid rotations, bounded gripper width, trajectory speed,
  and the fraction of valid hand frames;
- segmentation coverage, boundary quality, temporal flicker, and accidental
  removal of manipulated objects;
- temporal inpainting stability, residual hand pixels, duplicated/warped
  objects, and boundary artifacts;
- IK feasibility ratio, pose residuals, joint limits, self/environment
  collision, and discontinuous joint solutions;
- matching camera/depth coordinate systems, matching depth units and scale,
  valid render coverage, and stable occlusion boundaries;
- final-video checks for black, corrupted, blurred, or prolonged static frames,
  while explicitly preserving subtle task-critical events such as gripper
  closure.

The report's video-state consistency gate should be represented explicitly:
project the robot from its URDF and recorded joint state, segment the visible
robot in the image with an independently configured segmenter, and measure mask
IoU. Low overlap first routes to camera-parameter diagnosis/optimization; if it
cannot be explained and corrected, the episode is rejected. The projected mask
and observed robot mask are separate artifacts, and the threshold and correction
history must be recorded.

## Required provenance

Every output should be traceable without retaining credentials. A synthesis
manifest should include:

- source dataset, episode/frame identifiers, immutable URI/path references, and
  content fingerprints;
- canonical schema version and input/output time bases;
- ordered stage graph and a sanitized resolved configuration;
- backend/provider name, code version or commit, model/checkpoint identifier and
  digest, robot asset digest, prompt-template identifier/hash, parameters, random
  seed, device/runtime versions, and start/end timestamps;
- input and output artifact fingerprints for each stage, cache/resume decisions,
  warnings, quality metrics, status, and rejection reason;
- parent/child relationships among human masks, clean frames, robot RGB/depth,
  scene depth, visibility masks, synthesized frames, states, and actions.

Authentication headers, API keys, access tokens, cookies, signed URLs, secret
environment values, and unredacted request/response bodies are never provenance.

