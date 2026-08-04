# HDF 到 canonical 数据集转换规划

> 状态：HDF→canonical 107D 主链路已实现。本文同时保留设计、审计事实和仍未解决的训练就绪 blocker；实现转换不等于原始语义已经确认。

## 1. 目标与边界

转换器的目标是把采集侧 HDF episode 转换为 [`agilex_aloha_unified@1.1.0`](data_contract.md) 数据集，同时保留可追溯性并显式表达缺失信息。

核心边界：

- source adapter 只解释物理存储，不自行决定 robot/action 语义。
- embodiment adapter 负责 joint、camera、单位和 frame 映射。
- canonical transform 只接受已验证的语义输入，并生成 value + mask。
- annotation provider 与数值转换解耦。
- writer 不依赖 HDF，validator 不信任 writer 的输出。
- raw 数据只读；所有输出先进入新的 staging 目录。

首期不实现数据转换、图像分割、机械臂外观替换或在线 API 标注，只预留清晰接口。

## 2. Raw HDF 抽样事实

以下结论来自只读抽样的 3 个 episode，仅用于指导 scanner 和 adapter 设计，不代表全量 339 个一级 collection 都完全一致：

| HDF path | 抽样 shape/type |
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

抽样文件还有根 attributes `compress` 和 `sim`，但本次审计只确认 attribute 名称，未据其值建立语义。抽样的 T 为 247、258 和 324；文件约 0.9–1.2 GB，RGB/depth dataset 未启用 HDF compression，图像以单帧 chunk 存储。

重要的负面事实：

- 抽样未发现 `/observations/eepose`。
- depth key 是 camera 原名，不带 `_depth` 后缀。
- depth dtype 为 `uint8`，不能按 `uint16` 毫米深度直接解释。
- 抽样 HDF 未提供独立 timestamp dataset。
- raw 根目录下观察到 339 个一级目录，episode 位于更深层级；discoverer 不能假定只有一个 task 目录。

scanner 必须为每个文件生成 schema fingerprint，并统计异常；禁止用第一个文件代表全部 raw 数据。

## 3. FluxVLA 参考脚本审计

参考文件：

- `FluxVLA/tools/convert_hdf_to_lerobot.py`
- `FluxVLA/tools/hdf_to_lerobot_pipeline.py`
- `FluxVLA/tools/hdf_to_lerobot_direct_video.py`

### 3.1 可以借鉴的部分

- 递归发现 `episode_*.hdf5` 的基本思路。
- 同时兼容 HDF 中 raw RGB array 与编码图像 byte array 的解码分支。
- 将相机处理抽象为 camera-name 循环。
- 16DoF 到 14DoF 兼容逻辑可作为独立、显式启用的 migration transform 参考。
- LeRobot v2.1 dataset feature 创建、逐 episode Parquet 写入和统计接口的使用方式。
- PyAV 逐帧编码以避免先落地大量临时 JPEG/PNG 的思路。
- episode 完成后及时释放 buffer、记录内存和验证时间同步的意识。

借鉴表示重新封装受测试的概念，不表示复制私有 monkey-patch 或沿用其数据语义。

### 3.2 不能直接复用的部分

| 问题 | 影响 |
| --- | --- |
| 输出 `observation.state/action/eepose` | 不满足 12 列 canonical 107D schema |
| 只看首个 HDF 判断 action/eepose/depth | 混合 schema 数据会被错误解释 |
| 一次性把整集 RGB/depth 读入内存 | 大 episode 容易 OOM，不能稳定并发 |
| depth reader 查找 `{camera}_depth` | 与当前 raw 抽样 key 不一致 |
| depth feature 固定声明 `uint16` | 与当前 raw 抽样 `uint8` 不一致 |
| camera、480x640 和 30fps 硬编码 | 隐藏采集差异，可能产生错误 timestamp |
| `aloha` motor metadata 使用 right-first | 与 canonical left-first 假设存在语义冲突 |
| task 文件实际未接入，使用固定默认句子 | 产生不可追溯或错误标注 |
| 输出存在时递归删除目录 | 不符合 raw/output 安全和可恢复要求 |
| monkey-patch LeRobot 私有方法 | 对 LeRobot 版本强耦合，升级脆弱 |
| 每个 episode 后重建完整 HF dataset | 数据规模增大时开销不可控 |
| 缺少 schema/dataset version 和 manifest | 无法证明发布内容与转换器版本 |

因此，FluxMimir 应实现受版本约束的 writer adapter；若保留 direct-video 方案，也必须使用公开接口或在单独 compatibility layer 中锁定并测试 LeRobot 版本。

## 4. 中间表示

source reader 和 canonical writer 之间使用中立的流式 `EpisodeRecord` 概念，建议包含：

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

IR 中的每个字段应携带单位、顺序、frame、来源和 validity；仅有 numpy array 不足以跨模块传递语义。大图像通过 iterator/chunk reader 传递，不能把整集三路相机加载到内存。

## 5. 规划 stage

### Stage 0：Run specification

解析 declarative config，固定：source root、output root、schema/version、embodiment adapter、camera map、fps policy、annotation provider 和资源限制。生成 run id 与解析后的脱敏 config。

输出：`run_spec`。任何 credential 只从环境变量或 secret provider 注入，不得写入 config、日志或 metadata。

### Stage 1：Discovery

只读发现候选 HDF，解析稳定的 source identity，检测重复 episode，并按确定性顺序排序。目录名只作为 provenance 候选，不自动视为 task 文本。

输出：inventory，包含相对 source id、大小、mtime/etag（若适用），但不复制数据内容。

### Stage 2：Schema probe 与 preflight

读取 HDF metadata，不读取完整帧；为每个文件记录 dataset paths、shape、dtype、chunk、compression 和 attribute names。对 schema fingerprint 分组，并检查各 modality 的 T 是否一致。

输出：preflight report、schema groups、blocker/warning。未知 schema 默认失败，不静默跳过。

### Stage 3：Streaming decode

按小批帧读取 qpos/action 和相机数据；支持 raw array 与压缩 byte array。解码器只做格式转换、颜色空间标准化和损坏帧报告，不做 joint 语义重排。

输出：物理语义明确的 `EpisodeRecord` stream。

### Stage 4：Embodiment normalization

由 `agilex_aloha` adapter 校验左右臂顺序、关节/夹爪单位、camera order 和 action 对齐，再映射到 107D：

- qpos 可确认的维写入 observation 并设 mask=true。
- eepose 仅在 raw 提供或由已版本化 URDF/FK 正确计算时写入。
- `/action` 只有在 issued-target 语义、时序和单位确认后映射到 target。
- `/base_action` 不因 shape 看似匹配就直接写入 `[100,103)`。
- v1.1.0 未定义的 qvel、effort 和 depth 不偷偷增加 Parquet 列；记录为 ignored-with-reason，或通过未来 schema/versioned extension 接入。

输出：canonical values、masks 和 mapping provenance。

### Stage 5：Geometry 与时间

应用显式选择的 calibration/kinematics provider：

- FK 记录 URDF URI、hash、joint convention、base frame 和实现版本。
- ego2cam 记录标定 artifact hash、matrix convention 和 camera order。
- 缺少可信标定时写全 0 + validity=false。
- timestamp 优先使用采集时钟；若只能由 `frame_index/fps` 合成，必须先确认 fps，并记录 synthetic provenance。

输出：geometry/time fields 与 blocker。

### Stage 6：Annotation

先导入确定性 human/source annotations，再按显式配置调用自动标注 provider。provider 输出先通过 JSON schema、词表/FK 和区间校验，之后进入 review queue；不能直接覆盖人工标注。

输出：task/subtask definitions、frame assignment、provenance、confidence 和 review status。API 失败不得破坏数值转换，可将 episode 标为 annotation-pending。

### Stage 7：Write staging dataset

writer 在 `<output>.staging/<run_id>` 创建 one-Parquet-per-episode、三路视频和 metadata 草稿。每个 episode 使用临时文件 + atomic rename；支持基于 source identity 和 checksum 的安全 resume。

禁止：

- 修改 raw HDF。
- 删除已有 output root。
- 覆盖不同 run/version 的 artifact。
- 在验证前把 staging 目录标记为 release。

### Stage 8：Stats、metadata 与 manifest

按 mask 计算逐 episode 统计，生成 info、task/subtask/episode metadata 和版本文件。所有 artifact 完成后计算 SHA-256 与 schema-rules hash，最后写 manifest。

输出：完整但尚未 promote 的 release candidate。

### Stage 9：Independent validation

validator 在重新打开磁盘文件后检查：

- 恰好 12 个 Parquet 列及固定 shape/dtype，无 null/NaN/Inf。
- 107D padding、mask 和 target source 一致。
- frame/episode/global index 连续，timestamp 单调。
- Parquet rows、metadata length 和视频帧数一致。
- task/subtask/episode 外键完整。
- mask-aware stats 可复算。
- schema/dataset version 与 manifest 一致。
- 无缺失、额外或 checksum 不符的 artifact。

验证器根据目标用途计算 `training_ready` 和 blockers，不因文件能被读取就判定可训练。

### Stage 10：Promote

只有 validation 通过后，才把 staging 目录以原子方式提升为 immutable dataset version。发布动作输出审计摘要，不上传数据、不调用外部服务，除非用户另行明确授权。

## 6. 必须确认的 blocker

在实现或全量转换前，至少需要确认：

1. raw `qpos/action` 的左右臂和夹爪顺序是否为 canonical 所假设的 left-first。
2. `action` 是 observation、operator command、policy command、控制器 setpoint，还是经过延迟/shift 的标签。
3. action 与 observation 的时间对齐、首尾 padding 和 episode 终止规则。
4. 真实采集 fps、掉帧行为、是否存在未暴露的硬件 timestamp。
5. `base_action[T,2]` 的坐标系、单位和控制语义。
6. depth `uint8` 的编码、单位、无效值及其是否为真实 depth。
7. 是否存在适用的 URDF、joint limits、base frame 和 FK 约定。
8. 三路 camera intrinsics/extrinsics、手眼标定和标定版本。
9. 根 attributes `compress`、`sim` 的准确含义，以及是否存在多种 image encoding。
10. task、subtask、success/outcome 的权威来源；目录名是否仅是采集组织信息。
11. 339 个一级 collection 内是否存在更多 schema 变体、损坏文件或重复 episode。
12. annotation API 的 provider、数据出域政策、速率/成本预算和人工复核流程。

这些 blocker 应进入机器可读 preflight report。实现不得通过猜测填值来消除 blocker。

## 7. 错误、恢复与可观测性

- 默认 fail closed：schema/语义未知时停止该 episode 或整个 run，由 policy 决定。
- 每个错误包含 source identity、stage、error code 和脱敏上下文；不输出帧内容。
- run report 区分 converted、skipped、blocked、failed 和 annotation-pending。
- resume 只复用 checksum、run spec 和 converter version 均匹配的 artifact。
- 并发按内存和视频编码资源限流；不得依据 CPU 数量无限并发读取 GB 级 HDF。
- 测试从合成的最小 HDF fixture 开始，不复制真实数据进仓库。

## 8. 安全与隐私

- API key、app secret 和 access token 只能经环境变量或 secret manager 提供。
- 文档、示例配置、命令历史、日志、metadata 和异常中不得出现 credential。
- 自动标注默认不向外部 API 发送原始图像；启用数据出域必须由用户显式配置并满足数据政策。
- 日志只保留必要的 shape、dtype、hash 和匿名 source identity，不打印真实帧、标注内容或整份 HDF metadata。
- 本仓库不存放 raw 或转换后的真实数据，只存放代码、schema、合成 fixture 和脱敏报告。

## 9. 当前实现范围

当前已实现：确定性发现、逐文件 schema fingerprint、qpos/可选 eepose 到 107D
映射、显式 mask、合成 timestamp provenance、严格 12 列 Parquet、可选流式视频
编码、episode 统计、metadata、manifest、resume 和落盘复检。已经用一个真实
258 帧 episode 验证 qpos 可无损重建。

仍不包括：未经确认就启用 raw action target、从 qpos 推断 eepose、生成 ego2cam、
解释 uint8 depth、自动 annotation、宣称 `training_ready=true`，以及替换或删除
已有 immutable release。
