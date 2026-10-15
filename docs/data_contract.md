# FluxMimir Data Contract

> Status: the contract and HDF-to-Parquet writer are implemented. Semantics
> that remain unverified in the raw data are represented through masks,
> placeholders, and `training_ready=false`.

The specification is based on the Feishu document
[Data Structure v1](https://cwjgfm21di.feishu.cn/wiki/ORYMwWwYqiJswJkeVhqcqndWnth)
and cross-checked against the local read-only example at
`/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_AgileX_aloha_lerobot/example_canonical_107d_v3_1`.
Access credentials are not part of the data contract and must not enter the
repository or a release.

## 1. Contract identity and compatibility

The initial FluxMimir target contract is:

- schema id: `agilex_aloha_unified`
- schema version: `2.0.0`
- canonical layout: `lingbot_vla2_dexhand32x2_128_v1`
- LeRobot codebase version: `v2.1`
- embodiment id: `agilex_aloha`
- embodiment description: `AgileX ALOHA dual-arm robot`

The schema version and dataset version must be recorded independently.
Consumers must validate `schema_id`, `schema_version`, and the dataset version
instead of inferring the format from directory names.

Versions follow Semantic Versioning:

- major: breaks compatibility of columns, layout, episode identity, splits,
  tasks, or content semantics.
- minor: adds episodes, annotations, or modalities without changing existing
  sample semantics.
- patch: corrects metadata or samples without changing the schema contract.

Published dataset versions must be immutable. Any content change creates a
new version rather than modifying an existing release in place.

## 2. Directory layout

```text
<dataset_root>/
├── data/
│   └── chunk-{episode_chunk:03d}/
│       └── episode_{episode_index:06d}.parquet
├── videos/
│   └── chunk-{episode_chunk:03d}/
│       └── observation.images.<camera>/
│           └── episode_{episode_index:06d}.mp4
└── meta/
    ├── info.json
    ├── fluxvla_dataset_version.json
    ├── artifacts.sha256.json
    ├── tasks.jsonl
    ├── subtasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

Each episode has one Parquet file. RGB videos are stored separately; video
bytes and per-frame paths are not duplicated in Parquet. Video keys use this
fixed order:

1. `observation.images.cam_high`
2. `observation.images.cam_left_wrist`
3. `observation.images.cam_right_wrist`

## 3. Physical Parquet schema

Each episode Parquet must contain exactly the following 12 non-null columns:

| Column | Arrow/logical type | Semantics |
| --- | --- | --- |
| `observation.unified_128d` | `float32[128]` | canonical robot observation |
| `observation.unified_128d_mask` | `bool[128]` | per-dimension observation validity |
| `observation.ego2cam` | `float32[3,4,4]` | extrinsics for three cameras |
| `observation.ego2cam_valid` | `bool[3]` | per-camera extrinsic validity |
| `target.unified_128d` | `float32[128]` | canonical issued target/action |
| `target.unified_128d_mask` | `bool[128]` | per-dimension target validity |
| `target.source` | `int8` | target source enum |
| `timestamp` | `float32` | monotonic time within the episode, in seconds |
| `frame_index` | `int64` | contiguous frame number starting at zero |
| `episode_index` | `int64` | episode foreign key within the dataset |
| `index` | `int64` | global contiguous frame number within the dataset |
| `subtask_index` | `int64` | foreign key into `meta/subtasks.jsonl` |

Square brackets denote Arrow nested fixed-size lists, not variable-length
lists. Physical column order is also part of the contract and must follow the
table above in both the writer and validator.

Video features are recorded in `meta/info.json`. Their shapes follow the
LeRobot `[channels, height, width]` convention, for example `[3,480,640]`.
They are not part of the 12 Parquet columns above.

## 4. Unified 128D layout

All ranges use half-open intervals `[start,end)`. The prefix `[0,103)` is
frozen across releases; the appended leg and base-rotation slots come from
the Oli source profile:

| Range | Dimensions | Semantics | Convention |
| --- | ---: | --- | --- |
| `[0,14)` | 14 | dual-arm joint positions | rad; six active joints and one reserved joint slot per arm |
| `[14,28)` | 14 | dual end-effector poses | per arm: `x,y,z,qx,qy,qz,qw`; position in m; quaternion in `xyzw` order |
| `[28,30)` | 2 | left and right gripper positions | m |
| `[30,62)` | 32 | left dexterous-hand joint positions | `left_hand_joint_00..31` |
| `[62,94)` | 32 | right dexterous-hand joint positions | `right_hand_joint_00..31` |
| `[94,98)` | 4 | waist positions | names and units are defined by the embodiment adapter |
| `[98,100)` | 2 | head positions | names and units are defined by the embodiment adapter |
| `[100,103)` | 3 | mobile-base signals | must remain empty until semantics are confirmed |
| `[103,109)` | 6 | left leg joint positions | rad |
| `[109,115)` | 6 | right leg joint positions | rad |
| `[115,121)` | 6 | mobile-base rotation | 6D rotation representation |
| `[121,128)` | 7 | reserved | must be zero with `mask=false` |

For ALOHA, a 14D `qpose` ordered as left six arm joints, left gripper,
right six arm joints, and right gripper maps as follows:

```text
unified[0:6]  <- qpose[0:6]
unified[28]   <- qpose[6]
unified[7:13] <- qpose[7:13]
unified[29]   <- qpose[13]
```

The inverse view is:

```text
qpose = unified[0:6] + unified[28] + unified[7:13] + unified[29]
eepose = unified[14:28]
```

This mapping may be applied only after the raw joint order is confirmed.
`unified[6]` and `unified[13]` are reserved joint slots and must not contain
gripper values.

When both qpose and eepose are genuinely available, valid ALOHA observation
ranges are `[0,6)`, `[7,13)`, and `[14,30)`. Typical valid target ranges are
`[0,6)`, `[7,13)`, and `[28,30)`, but they may be marked valid only after the
raw action is confirmed to be the issued target with matching order and units.

## 5. Masks, placeholders, and validity

Missing dense tensors use numeric zero placeholders and explicit masks:

- unavailable 128D dimensions use `0.0` with `*_mask=false`.
- unavailable camera extrinsics use an all-zero 4x4 matrix with
  `ego2cam_valid=false` for that camera.
- an all-zero extrinsic is not an identity transform; consumers must check
  validity first.
- a numeric zero does not itself indicate missing data; only a mask or
  validity field does.
- all floating-point values marked valid must be finite.

Statistics must be computed per dimension using masks. Every 128D feature in
`episodes_stats.jsonl` must include at least `min`, `max`, `mean`, `std`,
`q01`, `q99`, and `count`. Invalid zero placeholders do not contribute, and a
completely invalid dimension has `count=0`.

### 5.1 `ego2cam`

- camera order: declared by the embodiment (`camera_order`); the ALOHA,
  dual-Franka, and dual-ReBot profiles keep `cam_high`, `cam_left_wrist`,
  `cam_right_wrist`
- shape: `[camera_count,4,4]`, with `camera_count = len(camera_order)`
- convention: `T_camera_from_ego`
- equation: `p_camera = T_camera_from_ego @ p_ego`
- ego frame: `base_link`

Measured hand-eye calibration must not be inferred from eepose. When trusted
calibration is unavailable, retain the all-zero placeholder and set validity
to false.

### 5.2 `target.source`

| Value | Meaning |
| ---: | --- |
| 0 | `policy` |
| 1 | `operator` |
| 2 | `blended_policy_operator` |
| 3 | `safety_override` |
| 4 | `scripted` |
| 5 | `unknown` |

A field named `action` does not prove its source or that it is the command
actually issued. Use `5`, keep the target mask false, and set the dataset to
`training_ready=false` when the source is unconfirmed.

## 6. Metadata contract

### 6.1 `meta/info.json`

It must describe at least:

- dataset and schema ids and versions
- `embodiment_id`, `embodiment_description`, and integer `fps`
- episode, frame, task, subtask, video, and chunk counts
- `data_path` and `video_path` templates
- all 12 Parquet features and three video features
- the complete `unified_layout`
- geometry, kinematics, time-alignment, and `ego2cam` conventions
- field validity, placeholder policy, and storage policy
- the target-source enum and annotation contract
- data origin, generator version, warnings, and `training_ready`

`qpose` and `eepose` are not duplicated as parallel ground-truth columns;
they are decoded from the versioned layout. Do not store a scalar
`target.valid` that can be derived from the mask, duplicate policy targets, or
repeat task/subtask text on every frame.

### 6.2 Tasks and subtasks

- `tasks.jsonl`: episode-level task vocabulary containing at least
  `task_index`, `task_id`, `task`, and `language`.
- `episodes.jsonl[*].task_index`: foreign key from an episode to a task.
- `subtasks.jsonl`: frame-level atomic-task vocabulary containing at least
  `subtask_index`, `subtask_id`, `subtask`, and `language`.
- Parquet `subtask_index`: per-frame foreign key to a subtask.

Recommended definition records also include category, required effectors,
object metadata, goal conditions, success evaluator, placeholder status, and
annotation provenance. Automated annotations must preserve provider/model,
prompt/schema version, time, confidence, and human-review status, but never an
API secret.

If an episode-level task is unknown, record an explicit unavailable
placeholder rather than fabricating an overall goal from an unordered set of
subtasks. Segments derivable from per-frame `subtask_index` are not duplicated.

### 6.3 Episodes and statistics

Each `episodes.jsonl` row must contain at least `episode_index`, a stable
`episode_uid`, source episode identity, `task_index`, `length`, `data_source`,
and outcome with its status. `episodes_stats.jsonl` has exactly one row per
episode linked by the same `episode_index`.

## 7. Version and content manifest

`meta/fluxvla_dataset_version.json` fixes the dataset and schema versions,
qualified dataset id, publication time, release status, immutable marker,
SemVer policy, and the path to `meta/artifacts.sha256.json`.

The content manifest must provide at least:

- manifest format and version
- schema id/version and canonical schema-rules hash
- dataset id/version and content hash
- converter name, version, and verifiable hash
- relative path, size, and SHA-256 for every released artifact

The manifest covers Parquet, videos, and all metadata except the manifest
itself. Release validation must reject missing or extra artifacts, hash
mismatches, and path traversal.

## 8. Training readiness

`training_ready` is a validation result, not a manually assigned label. It
must be false when any of the following remain unverified:

- issued target/action semantics, order, units, or alignment
- required observation state
- required cameras, videos, or time alignment
- task/subtask foreign-key closure
- schema, masks, statistics, or manifest validation
- calibration, kinematics, or episode-level tasks required by the target
  training workload

Validators must emit structured blocker lists so that "browsable data" and
"ready for supervised training" remain distinct states.

## 9. Current implementation status

The repository implements HDF-to-canonical Parquet conversion, optional MP4
materialization, base metadata, masked statistics, an artifact manifest, and
schema/foreign-key/video-count/statistics/hash validators. It does not
currently generate FK/eepose or camera extrinsics, interpret depth, call an
automated annotation API, or certify training readiness while action, timing,
and calibration semantics remain unverified.

## 10. ReBot dual-arm profile

ReBot B601 dual-arm data uses the same 12 Parquet columns and 128D slot ranges,
but its gripper slots represent motor angles rather than metric opening
positions. It therefore uses a distinct data-contract identity:

- schema id: `rebot_dual_unified`
- schema version: `2.0.0`
- layout id: `rebot_dual_128_v1`
- embodiment id: `rebot_b601_dual`
- gripper slots `[28,30)` unit: `rad`
- target source: `operator`

See [`conversion.md`](conversion.md) for the mapping and confirmed target
provenance. Consumers that support only `agilex_aloha_unified` must reject the
ReBot profile; sharing 128 dimensions does not imply unit compatibility.

## 11. Oli profile

LimX Oli humanoid data is recorded as a LeRobot v2.1 dataset with
`observation.state` `float32[33]`, `action` `float32[42]`, 30 fps, and two
cameras: `head` and `left_wrist`. It uses a distinct data-contract identity:

- schema id: `oli_unified`
- schema version: `1.0.0`
- layout id: `oli_128_v1`
- embodiment id: `oli`
- camera order: `head`, `left_wrist` (`ego2cam` shape `[2,4,4]`)
- target source: `operator` when the config confirms issued joint targets

The source `observation.state` order is the six left leg joints, the six right
leg joints, waist yaw/roll/pitch, head yaw/pitch, the 14 arm joints, and the two
hand-closed values. The 42D `action` adds `base_pos_cmd_x/y/z`,
`base_rot6d_0..5`, and uses the `joint_cmd_*` prefix for the 31 joint commands.

| Unified range | Slot | `observation.state` | `action` |
| --- | --- | --- | --- |
| `[0,14)` | arms | `[17:31]` | `[17:31]` |
| `[14,28)` | eef pose | `mask=false` | `mask=false` |
| `[28,30)` | gripper | `[31:33]` | `[40:42]` |
| `[30,62)` | left dexhand | `mask=false` | `mask=false` |
| `[62,94)` | right dexhand | `mask=false` | `mask=false` |
| `[94,98)` | waist | `[12:15]` | `[12:15]` |
| `[98,100)` | head | `[15:17]` | `[15:17]` |
| `[100,103)` | base position | `mask=false` | `[31:34]` |
| `[103,109)` | left leg | `[0:6]` | `[0:6]` |
| `[109,115)` | right leg | `[6:12]` | `[6:12]` |
| `[115,121)` | base rotation 6D | `mask=false` | `[34:40]` |
| `[121,128)` | reserved | `0.0`, `mask=false` | `0.0`, `mask=false` |

`unified[97]` is the fourth waist slot: it has no source dimension and stays
`0.0` with `mask=false`. The `action` column is the issued joint command only
when the conversion config declares `action_semantics=issued_joint_position_target`;
otherwise every target dimension stays masked. Slot names come from the
production `meta/info.json` joint names, which the adapter's preflight checks
against the recorded state joint order.
