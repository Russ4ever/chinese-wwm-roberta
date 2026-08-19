"""chinese-wwm-roberta 无数据阶段工程源码包。

子包职责：
- models:      checkpoint 安全加载、backbone/完整推理、pooling、Early-Exit 引擎
- probes:      逐层模型头（Head A–E）、逐层输出/缓存、只训练 head 的训练入口
- data:        数据协议/加载器（dataset.py）与取数脚本（fetch/）
- factors:     情绪因子输出协议与聚合骨架
- evaluation:  逐层对比分析、校准与阈值搜索

共享工具 config.py 保留在 src/ 根目录（YAML + 环境变量覆盖的配置加载）。
"""

__version__ = "0.1.0"
