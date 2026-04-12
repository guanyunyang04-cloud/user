# 主脑架构

## 1. 结构目标
主脑负责把整个工作区组织成可治理、可接管、可审计的脑网络，而不是一组彼此断裂的说明文档。

核心原则：

- `Agent 无状态，项目大脑有状态`
- 主脑只管跨项目拓扑、统一合同和治理边界
- 分脑只管本项目事实、当前状态和进入 body 的方法
- manifest 是机器入口，脑文档是语义入口
- body 与 brain 必须保持镜像匹配

## 2. 统一六层合同
整个工作区统一采用同一套六层认知合同：

1. `Identity`
   - 回答“我是谁、追求什么、不能做什么”
2. `Memory`
   - 保存稳定事实、规则、教训和历史证据
3. `Cognition`
   - 把 `过去 / 当下 / 未来` 组织成当前判断
4. `Decision`
   - 说明当前最值得做什么、为什么做
5. `Execution`
   - 定义 body 入口、环境口径和高频动作
6. `Governance`
   - 负责自检、反偏移、修复和写回

## 3. 标准附着分脑合同
所有已接入主脑的分脑，统一遵守同一套标准模块合同。

### 3.1 必备模块
- `identity_layer.md`
- `handoff_packet.md`
- `semantic_memory.md`
- `rule_memory.md`
- `lesson_memory.md`
- `temporal_state.md`
- `brain_architecture.md`
- `working_memory.md`
- `procedural_memory.md`
- `governance_layer.md`
- `environment_model.md`
- `action_system.md`
- `episodic_memory.md`
- `brain_manifest.json`

### 3.2 可选增强模块
- `project_map.md`
  - 当项目需要显式维护双环闭环、主线地图或 body 入口图时使用
- `handoff_rules.md`
  - 当项目需要额外强调接管纪律、前台运行纪律或恢复纪律时使用

### 3.3 标准读取顺序
附着分脑默认读取顺序统一为：

1. `brain_manifest.json`
2. `identity_layer.md`
3. `handoff_packet.md`
4. `semantic_memory.md`
5. `rule_memory.md`
6. `lesson_memory.md`
7. `temporal_state.md`
8. `brain_architecture.md`
9. `project_map.md` if present
10. `working_memory.md`
11. `procedural_memory.md`
12. `handoff_rules.md` if present
13. `governance_layer.md`
14. `environment_model.md`
15. `action_system.md`
16. `episodic_memory.md`

### 3.4 标准写入原则
- 身份、北极星、禁区：
  - `identity_layer.md`
- 当前最小完备状态：
  - `handoff_packet.md`
- 稳定事实：
  - `semantic_memory.md`
- 规则与教训：
  - `rule_memory.md`
  - `lesson_memory.md`
- 过去 / 当下 / 未来：
  - `temporal_state.md`
- 当前判决与优先级：
  - `working_memory.md`
- 可复用方法：
  - `procedural_memory.md`
- 操作入口：
  - `action_system.md`
- 时间顺序证据：
  - `episodic_memory.md`

## 4. 主脑专属职责
主脑只保留跨项目共性，不复制任何分脑内部事实。

### 4.1 主脑模块分工
- `identity_layer.md`
  - 工作区身份、北极星、宪法级约束
- `handoff_packet.md`
  - 工作区当前可接管状态摘要
- `rule_memory.md`
  - 工作区级高优先级规则
- `lesson_memory.md`
  - 工作区级可复用教训
- `temporal_state.md`
  - 工作区级 `Past / Present / Future`
- `master_brain.md`
  - 工作区拓扑、分脑角色、全局边界
- `brain_architecture.md`
  - 统一结构合同、标准模块、读写顺序
- `working_memory.md`
  - 当前跨项目优先级、治理焦点、全局目标函数
- `procedural_memory.md`
  - 跨项目方法学、更新顺序、协作纪律
- `governance_layer.md`
  - 反偏移、自检、修复、交接与写回闭环
- `environment_model.md`
  - 根环境、共享工具、守卫入口
- `brain_manifest.json`
  - 机器可读拓扑、child_brains、统一合同入口

## 5. 分脑专属职责
分脑只保留本项目自己的事实、当前判断和 body 入口。

附着分脑的 `brain_architecture.md` 不再重复定义整套通用模块合同，只需要回答三件事：

- 本项目在整个脑网络中的角色是什么
- 它相对标准合同有哪些项目特有模块或特有强调
- 它的 body 应该从哪些目录进入

## 6. Manifest 合同
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

推荐额外声明：

- `identity_path`
- `rule_memory_path`
- `lesson_memory_path`
- `temporal_state_path`
- `handoff_packet_path`
- `governance_path`
- `cognition_contract`

## 7. 去冗余规则
- 主脑负责定义“共性结构”，分脑只写“项目差异”。
- 同一套模块合同不应在每个分脑 `brain_architecture.md` 重复展开。
- `master_brain.md` 只保留拓扑和边界，不重复写标准模块合同。
- `episodic_memory.md` 不是默认第一入口，而是证据库。
- 结构变更先改本文件，再改 manifest，再改分脑差异文档，最后跑守卫。
