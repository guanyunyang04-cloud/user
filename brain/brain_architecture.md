# 主脑架构

## 1. 结构目标
主脑负责把工作区组织成可治理的脑网络，而不是一堆彼此断裂的人类说明文档。

核心原则：

- 主脑管拓扑与治理
- 分脑管本项目内部认知
- 记忆分层保存，不混写
- manifest 是机器入口，脑文档是语义入口
- body 与 brain 必须镜像匹配

## 2. 主脑模块
- `brain/master_brain.md`
  - 工作区身份、脑网络拓扑、全局边界
- `brain/brain_architecture.md`
  - 主脑结构、分脑 contract、写入路由
- `brain/working_memory.md`
  - 当前跨项目优先级、治理焦点、全局目标函数
- `brain/procedural_memory.md`
  - 跨项目方法学、Gemini 协同、更新顺序
- `brain/environment_model.md`
  - 根环境、共享工具、守卫入口
- `brain/brain_manifest.json`
  - 机器可读拓扑、child_brains、brain contract

## 3. 分脑标准模块
每个分脑都应具备以下模块：

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
