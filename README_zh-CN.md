# FluxMimir

FluxMimir 是 Flux 系列中负责离线数据工作的 codebase：将机器人原始采集
数据转换为版本化 canonical 数据集，生成 ego-to-robot 视觉数据，并为
FluxVLA 提供可审计的自动标注。

> **当前状态：** 已实现 AgileX ALOHA HDF 和 ReBot dual LeRobot v3 到各自
> 版本化 canonical 128D Parquet 的可执行转换，包括 MP4 物化、metadata、
> mask-aware 统计、manifest 和落盘校验；视觉合成与自动标注 backend 仍处于
> 框架阶段。

## 三条能力线

1. 将 AgileX ALOHA HDF episode 转换为
   `agilex_aloha_unified@2.0.0`，并将 ReBot dual LeRobot v3 episode 转换为
   `rebot_dual_unified@2.0.0`；两者保留各自的单位和 target provenance。
2. 按 Qwen-RobotManip 思路搭建轨迹重定向、人体区域分割、时序修复、机械臂
   IK/渲染、深度估计与遮挡合成的可替换 backend 链。
3. 通过 provider-neutral API 接口生成 task、subtask 和具身推理标注，并保留
   schema 校验、provenance、review、幂等与成本信息。

FluxMimir 只负责离线数据处理；FluxVLA 继续负责模型侧数据读取、训练和推理，
FluxThemis 负责 rollout 与评测。

## 安装、预检与转换

```bash
conda create -n fluxmimir python=3.10 -y
conda activate fluxmimir
cd /root/projects/FluxMimir
pip install -e '.[conversion,vision,api,dev]'

fluxmimir-convert \
  configs/conversion/agilex_aloha_hdf_to_canonical_128d.py --dry-run
fluxmimir-synthesize \
  configs/synthesis/qwen_robotmanip_ego_to_robot.py --dry-run
fluxmimir-annotate \
  configs/annotation/canonical_auto_annotation.py --dry-run
fluxmimir-validate configs/validation/canonical_128d.py --dry-run
```

真实转换需要显式传入 `--execute`。用户提供的 raw 根目录包含约 2.9 万个大
HDF，建议先只转换一个 episode：

```bash
fluxmimir-convert \
  configs/conversion/agilex_aloha_hdf_to_canonical_128d.py \
  --execute \
  --cfg-options \
  pipeline.source.max_episodes=1 \
  pipeline.writer.root=work_dirs/datasets/aloha_smoke
```

只校验 HDF→Parquet 时可再加 `pipeline.stages.2.enabled=False`，跳过 RGB 视频
编码。确认冒烟结果和输出盘容量后，再移除 `pipeline.source.max_episodes`。
示例配置也支持通过 `FLUXMIMIR_*` 环境变量覆盖路径。

## 文档

- [架构与职责边界](docs/architecture.md)
- [Canonical 128D 数据契约](docs/data_contract.md)
- [数据转换规划与审计](docs/conversion.md)
- [Ego-to-robot 合成规划](docs/synthesis.md)
- [自动标注规划](docs/annotation.md)
- [密钥与数据安全](docs/security.md)

未确认的语义不会被伪造成训练真值：raw action 默认输出为全零 target、全 false
mask 和 `source=unknown`；缺失的 eepose/ego2cam 同样以显式 mask 表达；缺失
timestamp 时按配置 FPS 合成并记录 provenance。因此当前输出可浏览、可审计，
但会保持 `training_ready=false`。

本项目使用 Apache License 2.0。
