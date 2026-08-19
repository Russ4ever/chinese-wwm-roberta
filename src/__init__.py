"""chinese-wwm-roberta 工程源码包。

子包职责：
- models: checkpoint 安全加载、二分类推理与 pooling
- data:   数据检查与取数脚本
- report_labeling: 单篇研报 Future Confirmation、Residual 与 Edge 纯计算逻辑

共享配置工具位于 ``config.py``（YAML + 环境变量覆盖）。
"""

__version__ = "0.1.0"
