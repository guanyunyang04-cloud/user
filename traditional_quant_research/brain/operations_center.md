# Traditional Quant Research 操作中枢

- 接管入口：先运行 workspace capsule，再确认路由选中 `traditional_quant_research` 后读取本脑区。
- 常用验证：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests
```

- 新实验流程：先在 `research_log/` 写明假设和证据等级，再在 `experiments/` 建配置和入口，最后把稳定结论写回 `knowledge_center` 或 `brain/references/`。
- 数据接入流程：先更新 `data/catalog.md`，确认字段、日期范围、复权口径和 survivorship bias 处理，再写研究代码。
