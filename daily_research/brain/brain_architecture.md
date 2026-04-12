# Daily Research 分脑架构

## 1. 分脑定位
`daily_research/brain/` 是 `daily_research` 的项目分脑，也是当前工作区最重的生产型样板分脑。

它在整个脑网络中的角色是：

- 管理正式生产研究与执行主线
- 保存 strongest-model、learned-control、live 默认执行的当前真相
- 让任何 agent 都能先接状态，再进入 `baseline / execution / deep_alpha / tools`

## 2. 继承的标准合同
`daily_research` 继承主脑定义的标准附着分脑合同，统一结构与读写顺序以 [brain/brain_architecture.md](H:/new_tdx64/PYPlugins/user/brain/brain_architecture.md) 为准。

因此本文件不再重复展开整套通用模块定义，只保留本项目相对标准合同的特有强调。

## 3. 本项目的特有强调
- `semantic_memory.md`
  - 重点锁定 formal / recent / live / promotion 的硬协议
- `project_map.md`
  - 必须明确 strongest-model 主线、learned-control 主线和 live 主线三者关系
- `working_memory.md`
  - 必须明确 strongest-model 当前答案、learned-control 当前答案和当前主问题
- `action_system.md`
  - 必须保留 strongest-model refresh、recent protocol refresh、policy family pipeline、trade plan 与一致性检查入口
- `episodic_memory.md`
  - 允许很长，但只能当证据库，不能重新变成默认接管入口

## 4. body 进入顺序
本项目的 body_map 核心入口固定为：

- `daily_research/baseline`
- `daily_research/execution`
- `daily_research/deep_alpha`
- `daily_research/tools`
- `daily_research/output`
- `daily_research/archive`

默认进入原则：

- 研究协议与当前判决问题：
  - 先看 `semantic / project_map / working`
- 正式训练与模型实现问题：
  - 再进 `deep_alpha`
- recent / verdict / 守卫 / pipeline 问题：
  - 再进 `tools`
- live 默认执行与交易计划问题：
  - 再进 `execution`

## 5. 去冗余原则
- 通用模块合同只在主脑 `brain_architecture.md` 定义一次。
- 本分脑的 `brain_architecture.md` 只保留项目特有差异。
- `identity / handoff / rule / lesson / temporal / working / action` 是高频入口。
- `episodic_memory.md` 继续保留历史细节，但不再承担当前状态入口职责。
