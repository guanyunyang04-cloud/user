# 主脑架构

## 1. 结构目标
主脑负责把工作区组织成可治理的脑网络，而不是一堆彼此断裂的人类说明文档。

核心原则：

- `Agent 无状态，项目大脑有状态`
- 主脑管拓扑与治理
- 分脑管本项目内部认知
- 记忆分层保存，不混写
- manifest 是机器入口，脑文档是语义入口
- body 与 brain 必须镜像匹配

## 2. 主脑模块
- `brain/identity_layer.md`
  - 工作区身份、北极星、宪法级约束
- `brain/handoff_packet.md`
  - 工作区当前可接管状态摘要
- `brain/rule_memory.md`
  - 工作区级高优先级规则
- `brain/lesson_memory.md`
  - 工作区级可复用教训
- `brain/temporal_state.md`
  - 工作区级 `Past / Present / Future`
- `brain/master_brain.md`
  - 工作区身份、脑网络拓扑、全局边界
- `brain/brain_architecture.md`
  - 主脑结构、分脑 contract、写入路由
- `brain/working_memory.md`
  - 当前跨项目优先级、治理焦点、全局目标函数
- `brain/procedural_memory.md`
  - 跨项目方法学、Gemini 协同、更新顺序
- `brain/governance_layer.md`
  - 反偏移、自检、修复、交接与写回闭环
- `brain/environment_model.md`
  - 根环境、共享工具、守卫入口
- `brain/brain_manifest.json`
  - 机器可读拓扑、child_brains、brain contract

## 3. 分脑标准模块
每个分脑至少具备以下基础模块：

- `semantic_memory.md`
  - 长期稳定认知、项目身份、body_map 摘要
- `working_memory.md`
  - 当前优先级、升级 gate、停止规则
- `procedural_memory.md`
  - 可复用方法学、写入路由、协作技能
- `environment_model.md`
  - 解释器、依赖、命令口径
- `action_system.md`
  - 系统入口、操作链路、边界
- `episodic_memory.md`
  - 时间顺序实验与证据
- `brain_manifest.json`
  - parent、read_order、write_routes、body_map、modules、handoff_contract

对于生产型分脑，推荐补齐以下增强模块：

- `identity_layer.md`
  - 项目使命、北极星、成功标准、禁区
- `rule_memory.md`
  - 宪法级 / 策略级 / 经验级规则
- `lesson_memory.md`
  - 错误教训与预防资产
- `temporal_state.md`
  - `Past Ledger / Present State / Future Map`
- `handoff_packet.md`
  - 当前标准交接包
- `governance_layer.md`
  - 自检、漂移检测、修复分级、写回闭环

## 4. 主脑与分脑 contract
主脑要求每个分脑在 manifest 中至少声明：

- `brain_type`
- `brain_id`
- `parent_brain`
- `attach_status`
- `entrypoint`
- `body_root`
- `read_order`
- `write_routes`
- `body_map`
- `modules`
- `handoff_contract`

若分脑缺失这些字段，视为脑结构未完成接入。

对于生产型分脑，当前推荐补充以下可机器读取字段：

- `identity_path`
- `rule_memory_path`
- `lesson_memory_path`
- `temporal_state_path`
- `handoff_packet_path`
- `governance_path`
- `cognition_contract`

## 5. 写入路由
- 跨项目边界与治理规则：
  - `brain/master_brain.md`
- 当前全局优先级与目标函数：
  - `brain/working_memory.md`
- 跨项目方法学与交接协议：
  - `brain/procedural_memory.md`
- 项目内部稳定认知与实验：
  - 写入对应分脑

## 6. 去冗余规则
- 根级与项目级 `README` 不再作为 AI 接管入口
- 同一结论不重复写进多个脑模块
- 结构变更先改架构文档，再改 manifest，最后跑守卫
- `episodic_memory.md` 不再作为默认第一入口，而是证据库与时间证据层
- 主脑也遵守与分脑相同的“身份 / 规则 / 教训 / 时态 / 交接 / 治理”思维模型
