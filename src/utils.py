import logging
import time
import functools
from pathlib import Path
from typing import Callable, Any, Dict, List, Union, Optional
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def timer(func):
    """A decorator that prints how long a function took to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f'{func.__name__} took {end_time - start_time:.2f} seconds to run')
        return result
    return wrapper

def setup_logging(config: Dict):
    """设置日志配置"""
    # 创建日志目录
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # 设置日志文件名
    log_file = log_dir / f"recommender_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, config.get('level', 'INFO')),
        format=config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 设置第三方库的日志级别
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")

# ... [保留原有的timer和memory_usage装饰器] ...

class TensorBatchGenerator:
    """张量批处理生成器"""
    def __init__(self, 
                 features: Dict[str, torch.Tensor], 
                 labels: Optional[torch.Tensor] = None,
                 batch_size: int = 128,
                 shuffle: bool = True):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = next(iter(features.values())).shape[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = torch.randperm(self.n_samples) if self.shuffle else torch.arange(self.n_samples)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch = {
                name: tensor[batch_indices].to(self.device)
                for name, tensor in self.features.items()
            }
            
            if self.labels is not None:
                batch['labels'] = self.labels[batch_indices].to(self.device)
            
            yield batch

class TensorDataFrameSerializer:
    """张量和DataFrame序列化工具"""
    @staticmethod
    def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                            epoch: int, loss: float, path: str):
        """保存模型检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

    @staticmethod
    def load_model_checkpoint(path: str, model: nn.Module, 
                            optimizer: Optional[torch.optim.Optimizer] = None):
        """加载模型检查点"""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

def convert_to_tensor(data: Union[np.ndarray, pd.DataFrame, pd.Series], 
                     dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """将不同类型的数据转换为PyTorch张量"""
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        tensor = torch.from_numpy(data.values)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
    if dtype:
        tensor = tensor.to(dtype)
    
    return tensor

def set_seeds(seed: int = 42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def calculate_model_size(model: nn.Module) -> float:
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

class MetricTracker:
    """指标跟踪器"""
    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
                self.history[name] = []
            self.metrics[name].append(value)

    def get_latest(self) -> Dict[str, float]:
        """获取最新指标"""
        return {name: values[-1] for name, values in self.metrics.items()}

    def get_average(self) -> Dict[str, float]:
        """获取平均指标"""
        return {name: np.mean(values) for name, values in self.metrics.items()}

    def save_history(self, path: str):
        """保存历史记录"""
        with open(path, 'w') as f:
            json.dump(self.history, f)

# ... [保留原有的其他工具函数] ...

if __name__ == "__main__":
    # 测试代码
    config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # 设置随机种子
    set_seeds(42)
    
    # 获取设备
    device = get_device()
    
    # 测试装饰器
    @timer
    @memory_usage
    def test_function():
        logger.info("Testing decorators...")
        time.sleep(1)
        return "Test completed"
    
    test_function()
