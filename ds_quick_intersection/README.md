# downstream

下游模块与 `mae_pretrain` 解耦，仅通过 `mae_pretrain/src/engine/pipeline.py` 调用能力。

- `configs/`: 下游任务配置
- `scripts/`: 下游任务脚本
- `src/tasks/`: 任务封装
- `src/dutils/`: 下游本地工具（不复用 pretrain utils）

## 示例

```bash
python scripts/run_pipeline_mae.py --config configs/recons.yaml
```
