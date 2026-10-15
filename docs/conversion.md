# Canonical Dataset Conversion Plan

> Status: the HDF-to-canonical 128D path is implemented. This document keeps
> the design, audit findings, and unresolved training-readiness blockers
> together. Implementing conversion does not confirm raw-data semantics.

## 1. Goals and boundaries

The converter transforms acquisition-side HDF episodes into the
[`agilex_aloha_unified@2.0.0`](data_contract.md) dataset while preserving
traceability and representing missing information explicitly.

Core boundaries:

- A source adapter interprets physical storage only; it does not decide robot
  or action semantics.
- An embodiment adapter owns joint, camera, unit, and frame mappings.
- A canonical transform accepts only semantically verified input and produces
  values plus masks.
- Annotation providers remain separate from numeric conversion.
- The writer does not depend on HDF, and validators do not trust writer output.
- Raw data is read-only; all output first enters a new staging directory.

The initial scope leaves visual segmentation, robot-appearance replacement,
and online API annotation to their dedicated pipelines and exposes clear
interfaces for them.

## 2. Raw HDF sample findings

The following findings come from read-only inspection of three episodes. They
guide scanner and adapter design but do not imply that all 339 top-level
collections have the same schema:

| HDF path | Sample shape/type |
| --- | --- |
| `/action` | `float32[T,14]` |
| `/base_action` | `float32[T,2]` |
| `/observations/qpos` | `float32[T,14]` |
| `/observations/qvel` | `float32[T,14]` |
| `/observations/effort` | `float32[T,14]` |
| `/observations/images/cam_high` | `uint8[T,480,640,3]` |
| `/observations/images/cam_left_wrist` | `uint8[T,480,640,3]` |
| `/observations/images/cam_right_wrist` | `uint8[T,480,640,3]` |
| `/observations/images_depth/cam_high` | `uint8[T,480,640]` |
| `/observations/images_depth/cam_left_wrist` | `uint8[T,480,640]` |
| `/observations/images_depth/cam_right_wrist` | `uint8[T,480,640]` |

The sampled files also have root attributes named `compress` and `sim`, but
this audit confirms only their names and assigns no semantics to their values.
Sampled lengths are 247, 258, and 324 frames. Files are approximately
0.9-1.2 GB; RGB and depth datasets do not use HDF compression and store images
in single-frame chunks.

Important negative findings:

- The samples do not contain `/observations/eepose`.
- Depth keys use the original camera names without a `_depth` suffix.
- Depth uses `uint8` and cannot be interpreted directly as `uint16`
  millimetres.
- The sampled HDF files contain no independent timestamp dataset.
- Episodes are nested below 339 observed top-level raw directories; discovery
  cannot assume a single task directory.

The scanner must generate a schema fingerprint for every file, group the
fingerprints, and report anomalies. The first file must not stand in for the
entire raw dataset.

## 3. Audit of FluxVLA reference scripts

Reference files:

- `FluxVLA/tools/convert_hdf_to_lerobot.py`
- `FluxVLA/tools/hdf_to_lerobot_pipeline.py`
- `FluxVLA/tools/hdf_to_lerobot_direct_video.py`

### 3.1 Reusable ideas

- Recursively discover `episode_*.hdf5` files.
- Decode both raw RGB arrays and encoded image-byte arrays from HDF.
- Handle cameras through a camera-name loop.
- Keep 16DoF-to-14DoF compatibility as a separate, explicitly enabled
  migration transform.
- Use the LeRobot v2.1 feature, per-episode Parquet, and statistics interfaces
  as references.
- Encode frames incrementally with PyAV instead of materializing large
  temporary JPEG/PNG collections.
- Release episode buffers promptly and track memory and synchronization
  validation costs.

Reusing an idea means reimplementing it behind tested interfaces, not copying
private monkey patches or inheriting their data semantics.

### 3.2 Behaviors that cannot be reused directly

| Problem | Impact |
| --- | --- |
| Emits `observation.state/action/eepose` | Does not satisfy the 12-column canonical 128D schema |
| Infers action/eepose/depth from the first HDF | Misinterprets mixed-schema datasets |
| Loads all RGB/depth for a dataset into memory | Risks OOM on large episodes and prevents stable concurrency |
| Looks for `{camera}_depth` | Does not match sampled raw keys |
| Declares depth as fixed `uint16` | Does not match sampled `uint8` data |
| Hard-codes cameras, 480x640, and 30 fps | Hides acquisition differences and may create invalid timestamps |
| Uses right-first ALOHA motor metadata | Conflicts with the canonical left-first assumption |
| Leaves task input disconnected and inserts a fixed sentence | Produces incorrect or untraceable annotations |
| Recursively deletes an existing output directory | Violates raw/output safety and recoverability requirements |
| Monkey-patches private LeRobot methods | Couples conversion to a fragile LeRobot implementation version |
| Rebuilds the complete HF dataset after each episode | Does not scale with dataset size |
| Omits schema/dataset versions and a manifest | Cannot prove release contents or converter version |

FluxMimir therefore needs a version-constrained writer adapter. Any retained
direct-video path must use public APIs or a separately version-locked and
tested compatibility layer.

## 4. Intermediate representation

Source readers and canonical writers communicate through a neutral streaming
`EpisodeRecord` concept:

```text
EpisodeRecord
├── source_identity
├── source_schema_fingerprint
├── frame_count
├── clocks / timestamps
├── observations
│   ├── joint_position
│   ├── optional eef_pose
│   ├── optional base/effort/velocity
│   └── camera frame streams
├── issued_target_candidate
├── calibration references
├── task/subtask candidates
└── provenance + warnings
```

Every IR field should carry units, order, frame, origin, and validity. A NumPy
array alone is insufficient for cross-module semantics. Large images travel
through iterators or chunk readers rather than loading three complete camera
streams into memory.

## 5. Planned stages

### Stage 0: Run specification

Parse declarative configuration and fix the source root, output root,
schema/version, embodiment adapter, camera map, fps policy, annotation
provider, and resource limits. Generate a run id and a resolved, redacted
configuration.

Output: `run_spec`. Credentials are injected only through environment
variables or a secret provider and never enter configuration, logs, or
metadata.

### Stage 1: Discovery

Discover candidate HDF files read-only, derive stable source identities,
detect duplicate episodes, and sort deterministically. Directory names are
provenance candidates only and are not automatically treated as task text.

Output: an inventory with relative source id, size, and mtime/etag where
available, without copying data contents.

### Stage 2: Schema probe and preflight

Read HDF metadata without reading full frames. Record dataset paths, shapes,
dtypes, chunks, compression, and attribute names for every file. Group schema
fingerprints and verify that modality lengths agree.

Output: preflight report, schema groups, blockers, and warnings. Unknown
schemas fail closed rather than being skipped silently.

### Stage 3: Streaming decode

Read qpos/action and camera data in small frame batches. Support raw arrays and
compressed byte arrays. Decoders perform format conversion, color-space
normalization, and corrupt-frame reporting only; they do not reorder joints.

Output: an `EpisodeRecord` stream with explicit physical semantics.

### Stage 4: Embodiment normalization

The `agilex_aloha` adapter verifies left/right order, joint and gripper units,
camera order, and action alignment before mapping into 128D:

- Write confirmed qpos dimensions to observations and set their masks true.
- Write eepose only when supplied by raw data or correctly computed by a
  versioned URDF/FK implementation.
- Map `/action` to a target only after issued-target semantics, timing, and
  units are confirmed.
- Do not map `/base_action` into `[100,103)` merely because its shape appears
  compatible.
- Do not add qvel, effort, or depth columns undefined by v2.0.0. Record them as
  ignored with a reason, or introduce them through a future versioned schema.

Output: canonical values, masks, and mapping provenance.

### Stage 5: Geometry and time

Apply explicitly selected calibration and kinematics providers:

- FK records the URDF URI and hash, joint convention, base frame, and
  implementation version.
- ego2cam records the calibration artifact hash, matrix convention, and camera
  order.
- Missing trusted calibration produces all-zero values with validity false.
- Timestamps prefer acquisition clocks. If only `frame_index/fps` is
  available, fps must first be confirmed and synthetic provenance recorded.

Output: geometry/time fields and blockers.

### Stage 6: Annotation

Import deterministic human/source annotations before calling an automated
annotation provider under explicit configuration. Validate provider output
against JSON schemas, vocabularies, foreign keys, and time ranges before it
enters a review queue. It must not overwrite human annotations directly.

Output: task/subtask definitions, frame assignments, provenance, confidence,
and review status. API failure does not corrupt numeric conversion; an episode
may instead become annotation-pending.

### Stage 7: Write staging dataset

Create one Parquet per episode, three video streams, and draft metadata under
`<output>.staging/<run_id>`. Use temporary files and atomic rename for each
episode, with safe resume based on source identity and checksum.

Prohibited operations:

- modifying raw HDF
- deleting an existing output root
- overwriting artifacts from another run or version
- marking a staging directory as a release before validation

### Stage 8: Statistics, metadata, and manifest

Compute masked per-episode statistics, generate info and task/subtask/episode
metadata, and record versions. After all artifacts complete, compute SHA-256
and the schema-rules hash, then write the manifest.

Output: a complete but not yet promoted release candidate.

### Stage 9: Independent validation

Validators reopen files from disk and verify:

- exactly 12 Parquet columns with fixed shapes/dtypes and no null, NaN, or Inf
- 128D padding, masks, and target-source consistency
- contiguous frame/episode/global indices and monotonic timestamps
- agreement among Parquet rows, metadata lengths, and video frame counts
- task/subtask/episode foreign-key closure
- recomputable mask-aware statistics
- schema/dataset versions consistent with the manifest
- no missing, extra, or checksum-mismatched artifacts

Validators derive `training_ready` and its blockers for the intended use case;
readability alone does not imply training readiness.

### Stage 10: Promote

Promote a staging directory atomically into an immutable dataset version only
after validation passes. Promotion emits an audit summary and does not upload
data or call external services without separate explicit authorization.

## 6. Required confirmations

At minimum, confirm the following before implementation or full conversion:

1. Whether raw `qpos/action` uses the canonical left-first arm and gripper
   order.
2. Whether `action` is an observation, operator command, policy command,
   controller setpoint, or a delayed/shifted label.
3. Action-observation time alignment, boundary padding, and episode termination
   rules.
4. Actual acquisition fps, dropped-frame behavior, and any hidden hardware
   timestamps.
5. The coordinate frame, unit, and control semantics of `base_action[T,2]`.
6. The encoding, unit, and invalid values of `uint8` depth, and whether it is
   physical depth.
7. Applicable URDF, joint limits, base frame, and FK convention.
8. Intrinsics, extrinsics, hand-eye calibration, and calibration version for
   all three cameras.
9. Exact meanings of root attributes `compress` and `sim`, including whether
   multiple image encodings exist.
10. Authoritative task, subtask, and success/outcome sources, and whether
    directory names are only organizational metadata.
11. Additional schema variants, corrupt files, or duplicate episodes within
    the 339 top-level collections.
12. Annotation API provider, data-egress policy, rate/cost budget, and human
    review process.

These blockers belong in the machine-readable preflight report. The
implementation must not remove them by guessing values.

## 7. Errors, recovery, and observability

- Fail closed by default: unknown schema or semantics stop an episode or the
  run according to policy.
- Every error includes source identity, stage, error code, and redacted
  context, but no frame contents.
- Run reports distinguish converted, skipped, blocked, failed, and
  annotation-pending episodes.
- Resume reuses an artifact only when its checksum, run specification, and
  converter version all match.
- Concurrency is bounded by memory and video-encoding resources rather than
  scaling unboundedly with CPU count while reading GB-scale HDF files.
- Tests begin with minimal synthetic HDF fixtures and never copy real data
  into the repository.

## 8. Security and privacy

- API keys, app secrets, and access tokens enter only through environment
  variables or a secret manager.
- Credentials must not appear in documentation, example configuration,
  command history, logs, metadata, or exceptions.
- Automated annotation does not send raw images to an external API by default.
  Data egress requires explicit configuration and policy compliance.
- Logs retain only required shapes, dtypes, hashes, and anonymized source
  identities; they do not print frames, annotation content, or complete HDF
  metadata.
- This repository stores code, schemas, synthetic fixtures, and redacted
  reports, not raw or converted real datasets.

## 9. Current implementation scope

The current implementation provides deterministic discovery, per-file schema
fingerprints, qpos/optional-eepose mapping into 128D, explicit masks,
synthetic-timestamp provenance, strict 12-column Parquet, optional streaming
video encoding, per-episode statistics, metadata, manifest, resume, and
on-disk validation. A real 258-frame episode verified lossless qpos
reconstruction.

It does not enable raw action targets before confirmation, infer eepose from
qpos, generate ego2cam, interpret `uint8` depth, perform automated annotation,
claim `training_ready=true`, or replace/delete an immutable release.

### 9.1 Path overrides

`--raw-root PATH` and `--output-root PATH` override `pipeline.source.root` and
`pipeline.writer.root` at run time, on top of the config defaults, the
`FLUXMIMIR_*` environment variables, and `--cfg-options`, in that order.
`--raw-root` matches whichever variable the config declares (`raw_root`,
`source_root`, `dataset_root`); `--output-root` is rejected for pipelines whose
writer owns no root, such as `fluxmimir-validate`.

## 10. ReBot dual LeRobot v3 conversion

`rebot_dual_unified@2.0.0` (layout `rebot_dual_128_v1`) converts ReBot B601
dual-arm LeRobot v3 data into canonical 128D. It has a distinct schema/layout
because gripper values are motor angles in radians rather than ALOHA opening
positions in metres.

Confirmed source semantics:

- Joint order is left six arm joints, left gripper, right six arm joints,
  right gripper.
- `observation.state` is follower joint-position feedback.
- `action` comes from the left and right `target_joint_state` ROS topics and
  represents operator-teleoperation targets before driver soft-limit clipping.
  The recorder reads and concatenates them at `rebot_ros_record.py:471` and
  `rebot_ros_record.py:528`, respectively.
- All 14 values use radians. Approximately `0` is closed and `-4.712389` is the
  maximum gripper opening.

Mapping:

```text
unified[0:6]  <- values[0:6]
unified[28]   <- values[6]
unified[7:13] <- values[7:13]
unified[29]   <- values[13]
```

Camera names map as `cam_front -> cam_high`, `cam_wrist_left ->
cam_left_wrist`, and `cam_wrist_right -> cam_right_wrist`. Existing MP4 files
are copied without transcoding, and video frame counts are checked against
Parquet row counts.

Convert one episode first:

```bash
fluxmimir-convert \
  configs/conversion/rebot_dual_lerobot_v3_to_canonical_128d.py \
  --execute \
  --cfg-options \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=/tmp/rebot_dual_smoke
```

Remove `pipeline.source.max_episodes` for full conversion. Override input and
output paths through `FLUXMIMIR_RAW_ROOT` and `FLUXMIMIR_OUTPUT_ROOT`. The
current release remains `training_ready=false` because camera extrinsics,
hardware-clock provenance, and frame-level subtasks are unavailable. Confirmed
action semantics are not a blocker.

## 11. LeRobot v2.1 conversion

`LeRobotV2Source` converts LeRobot v2.0/v2.1 ALOHA datasets
(`data/chunk-*/episode_*.parquet` plus
`videos/chunk-*/<video_key>/episode_*.mp4`) into `agilex_aloha_unified@2.0.0`.

Confirmed source semantics:

- `observation.state` is the 14D follower joint position in the canonical
  left-arm/gripper/right-arm/gripper order, so it maps like raw ALOHA qpos.
- `meta/tasks.jsonl` supplies the episode-level task text.
- `observation.eepose`, when present, fills the optional `[14,28)` slot and
  marks it valid.
- `action` is optional. Datasets exported without it, or without confirmed
  issue provenance, keep a fully masked target placeholder with
  `target.source=unknown` instead of a fabricated label.
- Source camera keys map through `camera_mapping`, for example
  `cam_high -> observation.images.cam_high`.
- Frame `task_index` may vary inside one episode, which marks frame-level
  subtask labels. The contract forbids inventing an episode goal from an
  unordered set of subtasks, so such an episode records an explicit
  unavailable episode-level task instead of one frame label.

Existing MP4 files are copied without transcoding and frame counts are checked
against Parquet rows. Convert one episode first:

```bash
fluxmimir-convert \
  configs/conversion/agilex_aloha_lerobot_v2_to_canonical_128d.py \
  --execute \
  --cfg-options \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=/tmp/aloha_v2_smoke
```

Override input and output paths through `FLUXMIMIR_RAW_ROOT` and
`FLUXMIMIR_OUTPUT_ROOT`. The release stays `training_ready=false` while stored
action provenance, camera extrinsics, and frame-level subtasks remain
unconfirmed.

## 12. Static frame filtering

`FilterStaticFrames` is an optional conversion stage that drops recordings
which hold a single pose for too long. Frame `t` counts as static when every
dimension that is valid on both `t - 1` and `t` moved by at most its configured
step threshold; frame 0 has no incoming step and never counts. An episode is
dropped once its longest run of static frames reaches `static_frame_threshold`
(for example 50 frames, about 1.7 s at 30 fps).

The stage ships disabled in both ALOHA conversion configs. Thresholds are
grouped by canonical layout range and come from the configuration; the operator
carries no threshold defaults. The ALOHA profile
uses `arm_joint_position` `1.0e-4` and `gripper_position` `1.5e-4`. Two
policies are available: `report` only audits, `drop_episode` asks the runner to
leave the episode out of the release.

Dropped episodes produce no Parquet or video, and the runner keeps
`episode_index`, `index`, and `frame_index` contiguous. Every source episode is
recorded in `<runner.work_dir>/<audit_name>` as JSON lines: one header per run
with the resolved criteria and their fingerprint, then one row per episode with
its qualifying segments, static frame count, and decision. The stage fails
closed when an episode has no valid dimension, when a configured group is
absent from the canonical layout, or when it runs after
`ComputeMaskedEpisodeStatistics`.

## 13. franka_dual conversion

`franka_dual_unified@2.0.0` (layout `franka_dual_128_v1`) converts dual Franka
Panda LeRobot v2.1 data into canonical 128D. It has a distinct schema/layout
because each arm exposes seven joints plus one gripper opening.

Confirmed source semantics:

- `observation.state` is `float32[16]`: left `panda_joint1..7`, `left_gripper`,
  right `panda_joint1..7`, `right_gripper`. Joints are radians, grippers are
  opening positions in metres (`0.0` closed to `0.081` open); near the closed
  position the recorded opening can be slightly negative, so the config sets
  `gripper_min_m=-0.001` while the adapter default stays `0.0`.
- `observation.eepose` is `float32[16]`: `x, y, z, qx, qy, qz, qw` plus a
  redundant `gripper` per arm. Both arms share one base frame, so values are
  copied without any frame or unit transform.
- `action` is optional: while `action_semantics='unknown'` it is provenance
  only and the target stays fully masked.

Mapping:

```text
unified[0:7]   <- state[0:7]
unified[7:14]  <- state[8:15]
unified[28]    <- state[7]
unified[29]    <- state[15]
unified[14:21] <- eepose[0:7]
unified[21:28] <- eepose[8:15]
```

Slots 6 and 13 hold real seventh joints here, so downstreams select
`franka_qpose_indices()` through `schema_id` instead of assuming the ALOHA
view. `preflight` fails closed when the state/eepose width, the joint names,
the gripper range, or the camera set does not match this contract.

Camera names map as `cam_front -> cam_high`, `cam_wrist_left ->
cam_left_wrist`, and `cam_wrist_right -> cam_right_wrist`. Existing MP4 files
are copied without transcoding. Convert one episode first:

```bash
fluxmimir-convert \
  configs/conversion/franka_dual_lerobot_v2_to_canonical_128d.py \
  --execute \
  --cfg-options \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=/tmp/franka_dual_smoke
```

Override input and output paths through `FLUXMIMIR_RAW_ROOT` and
`FLUXMIMIR_OUTPUT_ROOT`; the default release root is
`work_dirs/datasets/franka_dual_128d`. The release stays `training_ready=false`
because the export has no action provenance, no calibrated ego2cam, no hardware
timestamps, and no frame-level subtasks. `FilterStaticFrames` ships disabled.

## 14. LeRobot v3.0 layout

`CanonicalDatasetV30Writer` writes the same canonical 12-column 128D content as
`CanonicalDatasetWriter`, but under the LeRobot v3.0 file layout. One episode
per file is kept, so each chunk holds up to `chunks_size` episodes and the
`file-XXX` ordinal is per chunk.

| Item | v2.1 layout | v3.0 layout |
|---|---|---|
| Data | `data/chunk-{c:03d}/episode_{i:06d}.parquet` | `data/chunk-{c:03d}/file-{f:03d}.parquet` |
| Video | `videos/chunk-{c:03d}/observation.images.{cam}/episode_{i:06d}.mp4` | `videos/observation.images.{cam}/chunk-{c:03d}/file-{f:03d}.mp4` |
| Episodes | `meta/episodes.jsonl` | `meta/episodes/chunk-{c:03d}/file-{f:03d}.parquet`, one row per episode |
| Tasks | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| Statistics | `meta/episodes_stats.jsonl` | `meta/stats.json` pooled over episodes, plus `stats/*` columns in the episode metadata |
| Info | `codebase_version=v2.1` | `codebase_version=v3.0`, `data_files_size_in_mb=100`, `video_files_size_in_mb=200`, no `total_chunks`/`total_videos` |

An episode metadata row carries `episode_index`, `tasks`, `length`,
`data/chunk_index`, `data/file_index`, `dataset_from_index`,
`dataset_to_index`, `videos/{key}/{chunk_index,file_index,from_timestamp,to_timestamp}`
and `stats/{feature}/{min,max,mean,std,q01,q99,count}`. `meta/stats.json` holds
the pooled `min`/`max`/`mean`/`std` and the per-dimension `count` over all
episodes; quantiles stay per episode.

Mimir publish conventions are unchanged: `meta/fluxvla_dataset_version.json`,
`meta/artifacts.sha256.json` and the `meta/subtasks.jsonl` annotation file are
written by both layouts. As with v2.1, the product holds canonical content, so
consumers read it through the canonical contract and not through the native
LeRobot reader.

```bash
fluxmimir-convert \
  configs/conversion/agilex_aloha_hdf_to_canonical_v30.py \
  --execute \
  --cfg-options \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=/tmp/aloha_v30_smoke
```

The config passes `preset='veryfast'` and `crf=18` to `EncodeEpisodeVideos`,
matching the production Oli v3.0 encoding; `preset` defaults to `None`, which
keeps the previous ffmpeg options byte-identical for the v2.1 configs.
`CanonicalV30LayoutValidator` validates the layout: canonical columns and
types, contiguous frame and dataset indices, one episode row per data file,
`meta/tasks.parquet` coverage of the episode tasks, `meta/stats.json` counts
against the masks, and video file alignment.
