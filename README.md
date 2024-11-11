# tianchi-E-commerce
移动推荐算法，以阿里巴巴移动电商平台的真实用户-商品行为数据为基础，同时提供移动时代特有的位置信息，希望你能够挖掘数据背后丰富的内涵，为移动用户在合适的时间、合适的地点精准推荐合适的内容
基于阿里巴巴移动电商平台的用户行为数据构建的推荐系统，针对天池竞赛"移动推荐算法"设计。本项目实现了完整的推荐系统流程，包括数据处理、特征工程、模型训练和预测。

## 项目结构

```
mobile_recommendation/
│
├── config/
│   └── config.yaml           # 配置文件
│
├── data/
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后的数据
│   └── output/               # 输出结果
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # 数据处理模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── model.py              # 模型定义
│   ├── trainer.py            # 训练模块
│   └── utils.py             # 工具函数
│
├── notebooks/                # Jupyter notebooks
├── tests/                    # 单元测试
├── requirements.txt          # 项目依赖
└── main.py                   # 主程序
```

## 环境要求

- Python 3.8+
- 依赖包：见 requirements.txt

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将原始数据文件放置在 `data/raw/` 目录下：
- tianchi_fresh_comp_train_user_2w.csv
- tianchi_fresh_comp_train_item_2w.csv

### 3. 运行项目

```bash
# 验证配置
python main.py validate-config

# 运行数据分析
python main.py analyze-data

# 训练模型
python main.py train

# 生成预测
python main.py predict

# 或者运行完整流程
python main.py run-all
```

## 主要功能

### 数据处理
- 数据清洗和预处理
- 类别特征编码
- 时间特征处理
- 缺失值处理

### 特征工程
- 用户行为特征
  - 各类行为计数
  - 类别偏好
  - 时间模式
- 商品特征
  - 被交互次数
  - 转化率
  - 类别分布
- 用户-商品交叉特征
  - 交互历史
  - 时间序列特征

### 模型
- 基于LightGBM的推荐模型
- 支持多时间窗口的特征
- 批量预测支持
- 模型评估和验证

### 评估指标
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数

## 配置说明

配置文件 `config/config.yaml` 包含以下主要部分：

```yaml
data:
  raw_user_data: 数据路径配置
  ...

features:
  time_windows: 特征时间窗口设置
  ...

model:
  params: 模型参数设置
  ...

training:
  train_dates: 训练日期设置
  ...
```

## 性能优化

1. 数据处理优化
- 特征缓存机制
- 内存使用优化
- 批处理支持

2. 模型优化
- 特征选择
- 参数调优
- 模型融合

## 使用示例

```python
# 示例：训练模型并生成预测
from src.trainer import ModelTrainer

# 初始化训练器
trainer = ModelTrainer('config/config.yaml')

# 运行训练
metrics = trainer.run_training()

# 生成预测
trainer.generate_submission('2014-12-19')
```

## 测试

运行单元测试：
```bash
pytest tests/
```

## 开发指南

代码风格

- 遵循PEP 8规范

- 使用类型注解

- 添加详细的文档字符串

## 常见问题

1. 内存不足
   - 减小 batch_size
   - 使用 reduce_memory_usage 优化
   - 启用特征缓存

2. 训练速度慢
   - 调整特征窗口大小
   - 减少特征数量
   - 使用增量训练

## 许可证

MIT License

## 作者

綦子宽

## 参考

- 竞赛链接：[天池移动推荐算法](https://tianchi.aliyun.com/competition/entrance/231522/information)

## 更新日志

### v1.0.0 (2024-11)
- 初始版本发布
- 基本功能实现
- 文档完善
