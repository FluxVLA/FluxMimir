# FluxMimir 数据契约

> 状态：契约及 HDF→Parquet writer 已实现；raw 数据中未确认的语义仍通过 mask、placeholder 和 `training_ready=false` 表达。

规范来源为飞书文档 [Data Structure v1](https://cwjgfm21di.feishu.cn/wiki/ORYMwWwYqiJswJkeVhqcqndWnth)，并以本机只读样例
`/mnt/data/cpfs/mnt/data/liyinhao/datasets/RealRobot_AgileX_aloha_lerobot/example_canonical_107d_v3_1`
交叉核对。访问凭据不是数据契约的一部分，也不得进入仓库或发布物。

## 1. 契约身份与兼容性

FluxMimir 首期目标契约为：

- schema id：`agilex_aloha_unified`
- schema version：`1.1.0`
- canonical layout：`lingbot_vla2_dexhand32x2_107_v1`
- LeRobot codebase version：`v2.1`
- embodiment id：`agilex_aloha`
- embodiment description：`AgileX ALOHA dual-arm robot`

schema 版本与 dataset 版本必须独立记录。消费者必须同时校验 `schema_id`、`schema_version` 和 dataset version，不能依据目录名猜测格式。

版本遵循 Semantic Versioning：

- major：破坏列、layout、episode 身份、split、task 或内容语义的兼容性。
- minor：在不改变已有样本语义的前提下增加 episode、标注或 modality。
- patch：修正 metadata 或样本，但不改变 schema 契约。

已发布的数据集版本应标记为 immutable。任何内容变化都产生新版本，不原地改写发布物。

## 2. 目录布局

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

每个 episode 对应一个 Parquet。RGB 视频独立存放，不把视频字节或逐帧路径重复写入 Parquet。视频 key 的顺序固定为：

1. `observation.images.cam_high`
2. `observation.images.cam_left_wrist`
3. `observation.images.cam_right_wrist`

## 3. Parquet 物理 schema

每个 episode 的 Parquet 必须严格包含以下 12 列，不允许 null：

| 列 | Arrow/逻辑类型 | 语义 |
| --- | --- | --- |
| `observation.unified_107d` | `float32[107]` | canonical robot observation |
| `observation.unified_107d_mask` | `bool[107]` | observation 每维有效性 |
| `observation.ego2cam` | `float32[3,4,4]` | 三路相机的外参 |
| `observation.ego2cam_valid` | `bool[3]` | 每路外参有效性 |
| `target.unified_107d` | `float32[107]` | canonical issued target/action |
| `target.unified_107d_mask` | `bool[107]` | target 每维有效性 |
| `target.source` | `int8` | target 来源枚举 |
| `timestamp` | `float32` | episode 内单调时间，单位为秒 |
| `frame_index` | `int64` | episode 内从 0 开始的连续帧号 |
| `episode_index` | `int64` | 数据集内 episode 外键 |
| `index` | `int64` | 数据集内全局连续帧号 |
| `subtask_index` | `int64` | `meta/subtasks.jsonl` 外键 |

方括号表示 Arrow nested fixed-size list，不是变长 list。物理列顺序也属于契约，writer 和 validator 应按上表固定。

视频 feature 记录在 `meta/info.json`，其 shape 采用 LeRobot 的 `[channels, height, width]`，例如 `[3,480,640]`；它们不是上述 12 个 Parquet 列的一部分。

## 4. Unified 107D layout

所有 range 使用半开区间 `[start,end)`：

| Range | 维数 | 语义 | 约定 |
| --- | ---: | --- | --- |
| `[0,14)` | 14 | 双臂关节位置 | rad；每臂 6 个当前关节槽位和 1 个 reserved joint 槽位 |
| `[14,28)` | 14 | 双末端位姿 | 每臂 `x,y,z,qx,qy,qz,qw`；位置 m；四元数 `xyzw` |
| `[28,30)` | 2 | 左右夹爪位置 | m |
| `[30,62)` | 32 | 左灵巧手关节位置 | `left_hand_joint_00..31` |
| `[62,94)` | 32 | 右灵巧手关节位置 | `right_hand_joint_00..31` |
| `[94,98)` | 4 | 腰部位置 | embodiment adapter 定义具体名字和单位 |
| `[98,100)` | 2 | 头部位置 | embodiment adapter 定义具体名字和单位 |
| `[100,103)` | 3 | 移动底盘信号 | 未确认语义不得填充 |
| `[103,107)` | 4 | reserved | 必须为 0 且 mask=false |

ALOHA 的 14D `qpose` 约定为左臂 6D、左夹爪、右臂 6D、右夹爪时，映射为：

```text
unified[0:6]  <- qpose[0:6]
unified[28]   <- qpose[6]
unified[7:13] <- qpose[7:13]
unified[29]   <- qpose[13]
```

逆向解码为：

```text
qpose = unified[0:6] + unified[28] + unified[7:13] + unified[29]
eepose = unified[14:28]
```

只有确认了 raw joint order 后才能应用该映射。`unified[6]` 和 `unified[13]` 是 reserved joint 槽位，不能用夹爪值填充。

若 qpose 和 eepose 都真实可用，ALOHA observation 的有效 ranges 是 `[0,6)`、`[7,13)`、`[14,30)`。target 的常见有效 ranges 是 `[0,6)`、`[7,13)`、`[28,30)`，但仅在确认 raw action 是真实下发 target、顺序和单位一致后才可标记有效。

## 5. Mask、placeholder 与有效性

缺失的 dense tensor 使用数值 0 占位，并同时使用显式 mask 标为无效：

- 不可用的 107D 维：值为 `0.0`，对应 `*_mask=false`。
- 不可用的相机外参：4x4 矩阵为全 0，对应 camera 的 `ego2cam_valid=false`。
- 全 0 外参不是单位变换，读取时必须先检查 validity。
- 数值 0 本身不表示缺失；有效性只能由 mask/validity 判断。
- 所有标为有效的浮点值必须 finite。

统计必须按 mask 逐维计算。`episodes_stats.jsonl` 中每个 107D feature 至少包含 `min`、`max`、`mean`、`std`、`q01`、`q99` 和 `count`；无效占位 0 不参与统计，完全无效的维度 `count=0`。

### 5.1 `ego2cam`

- camera order：`cam_high`、`cam_left_wrist`、`cam_right_wrist`
- shape：`[3,4,4]`
- convention：`T_camera_from_ego`
- equation：`p_camera = T_camera_from_ego @ p_ego`
- ego frame：`base_link`

未经测量的手眼标定不能从 eepose 猜测。若标定不可用，保留全 0 placeholder 并将 validity 设为 false。

### 5.2 `target.source`

| 值 | 含义 |
| ---: | --- |
| 0 | `policy` |
| 1 | `operator` |
| 2 | `blended_policy_operator` |
| 3 | `safety_override` |
| 4 | `scripted` |
| 5 | `unknown` |

字段名为 `action` 不足以证明其来源或是否为真实下发命令。来源未确认时使用 `5`，target mask 保持 false，并令数据集 `training_ready=false`。

## 6. Metadata 契约

### 6.1 `meta/info.json`

至少描述：

- dataset/schema id 与 version
- `embodiment_id`、`embodiment_description`、整数 `fps`
- episode、frame、task、subtask、video 和 chunk 数量
- `data_path` 与 `video_path` 模板
- 12 个 Parquet feature 和三路 video feature
- 完整 `unified_layout`
- geometry、kinematics、time alignment 和 `ego2cam` 约定
- field validity、placeholder policy、storage policy
- target source enum 与 annotation contract
- 数据来源、生成器版本、warning 和 `training_ready`

`qpose`、`eepose` 不作为平行真值列重复存储；它们应从版本化 layout 解码。也不存储可由 mask 推导的标量 `target.valid`，不复制 policy target，不逐帧重复 task/subtask 文本。

### 6.2 Task 与 subtask

- `tasks.jsonl`：episode 级任务词表，至少包含 `task_index`、`task_id`、`task`、`language`。
- `episodes.jsonl[*].task_index`：episode 到 task 的外键。
- `subtasks.jsonl`：帧级原子任务词表，至少包含 `subtask_index`、`subtask_id`、`subtask`、`language`。
- Parquet `subtask_index`：每帧到 subtask 的外键。

建议的定义记录还包括 category、required effectors、object metadata、goal conditions、success evaluator、placeholder status 和 annotation provenance。自动标注必须保存 provider/model、prompt/schema version、时间、置信度与 human-review 状态，但不得保存 API secret。

如果 episode 级任务未知，必须显式记录 unavailable placeholder，不能把无序 subtask 集合拼成虚假的总体目标。可由逐帧 `subtask_index` 推导的分段不重复存储。

### 6.3 Episode 与统计

`episodes.jsonl` 每行至少包含 `episode_index`、稳定的 `episode_uid`、source episode identity、`task_index`、`length`、`data_source`、outcome 及其状态。`episodes_stats.jsonl` 每个 episode 恰有一行，并以相同 `episode_index` 关联。

## 7. Version 与内容 manifest

`meta/fluxvla_dataset_version.json` 固定 dataset/schema 版本、qualified dataset id、发布时间、release status、immutable 标记、SemVer policy，以及 `meta/artifacts.sha256.json` 的路径。

内容 manifest 应至少提供：

- manifest 格式与版本
- schema id/version 和 canonical schema-rules hash
- dataset id/version 和 content hash
- converter 名称、版本与可验证 hash
- 每个发布 artifact 的相对路径、大小和 SHA-256

manifest 覆盖 Parquet、视频和除 manifest 自身以外的 metadata。发布验证必须拒绝缺失、额外、hash 不符或路径逃逸的 artifact。

## 8. Training readiness

`training_ready` 是验证结果，不是用户随意设置的标签。以下任一项未确认时必须为 false：

- issued target/action 的语义、顺序、单位或对齐不可验证
- observation 的关键 state 不可用
- 所需相机、视频或时间对齐不完整
- task/subtask 外键无效
- schema、mask、统计或 manifest 验证失败
- 当前训练任务要求的标定、kinematics 或 episode-level task 缺失

validator 应输出结构化 blocker 列表，使“数据可浏览”和“可用于监督训练”保持明确区分。

## 9. 当前实现状态

本仓库已实现 HDF 到 canonical Parquet、可选 MP4、基础 metadata、masked
statistics、artifact manifest 以及 schema/FK/video-count/stats/hash validator。
当前不会生成 FK/eepose 或相机外参，不解释 depth，不调用自动标注 API，也不会
在 action、时间和标定语义未确认时给出训练就绪认证。
