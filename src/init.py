"""
移动电商推荐系统
~~~~~~~~~~~~~~~

基于阿里巴巴移动电商平台的用户行为数据构建的推荐系统。

主要模块:
- data_processing: 数据处理模块
- feature_engineering: 特征工程模块
- model: 推荐模型模块
- trainer: 训练管理模块
- utils: 工具函数模块

示例:
    >>> from src.trainer import ModelTrainer
    >>> trainer = ModelTrainer('config/config.yaml')
    >>> metrics = trainer.run_training()
"""

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model import RecommendationModel
from src.trainer import ModelTrainer
from src.utils import (
    setup_logging,
    timer,
    memory_usage,
    DataFrameSerializer,
    reduce_memory_usage,
    save_dict_to_json,
    load_dict_from_json,
    BatchGenerator,
    create_submission_file,
    check_data_quality
)

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'final'
}

# 导出的主要类和函数
__all__ = [
    'DataProcessor',
    'FeatureEngineer',
    'RecommendationModel',
    'ModelTrainer',
    'setup_logging',
    'timer',
    'memory_usage',
    'DataFrameSerializer',
    'reduce_memory_usage',
    'save_dict_to_json',
    'load_dict_from_json',
    'BatchGenerator',
    'create_submission_file',
    'check_data_quality'
]

# 模块级配置
DEFAULT_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    },
    'data': {
        'random_seed': 42,
        'validation_split': 0.2,
        'shuffle': True
    },
    'model': {
        'batch_size': 1024,
        'num_epochs': 10
    }
}

def get_version():
    """获取版本信息"""
    version_str = '{major}.{minor}.{patch}'.format(**VERSION_INFO)
    if VERSION_INFO['release'] != 'final':
        version_str += '-' + VERSION_INFO['release']
    return version_str

def setup():
    """
    初始化包级配置

    设置随机种子、日志等基本配置
    """
    import numpy as np
    import random
    import torch
    
    # 设置随机种子
    random.seed(DEFAULT_CONFIG['data']['random_seed'])
    np.random.seed(DEFAULT_CONFIG['data']['random_seed'])
    torch.manual_seed(DEFAULT_CONFIG['data']['random_seed'])
    
    # 设置日志
    setup_logging(DEFAULT_CONFIG['logging'])

# 自动运行初始化配置
setup()


