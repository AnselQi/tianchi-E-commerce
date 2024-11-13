import logging
import time
import functools
from pathlib import Path
from typing import Callable, Any, Dict
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import warnings
from tqdm import tqdm


def setup_logging(config: Dict):
    """
    设置日志配置
    Args:
        config: 日志配置字典
    """
    # 创建日志目录
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # 设置日志文件名
    log_file = log_dir / \
        f"recommender_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, config.get('level', 'INFO')),
        format=config.get(
            'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
def validate_config(self) -> bool:
    """验证配置是否完整"""
    required_keys = {
        'data': ['raw_user_data', 'raw_item_data', 'processed_data_dir', 'output_dir'],
        'features': ['time_windows', 'categorical_features', 'numerical_features'],
        'model': ['type', 'params'],
        'training': ['train_start_date', 'train_end_date', 'pred_date', 'validation_days', 'top_k'],
        'logging': ['level', 'format', 'file']
    }

    try:
        for section, keys in required_keys.items():
            if section not in self.config:
                raise ValueError(f"Missing section: {section}")
            for key in keys:
                if key not in self.config[section]:
                    raise ValueError(f"Missing key in {section}: {key}")
        return True
    except ValueError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False

    # 设置第三方库的日志级别
    logging.getLogger('lightgbm').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")


def timer(func: Callable) -> Callable:
    """
    函数执行时间装饰器
    Args:
        func: 要计时的函数
    Returns:
        包装后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__module__)
        logger.info(f"Starting {func.__name__}...")

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Finished {func.__name__} in {duration:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(
                f"Error in {func.__name__} after {duration:.2f} seconds: {str(e)}")
            raise

    return wrapper


def memory_usage(func: Callable) -> Callable:
    """
    内存使用监控装饰器
    Args:
        func: 要监控的函数
    Returns:
        包装后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        logger = logging.getLogger(func.__module__)
        logger.info(f"Memory before {func.__name__}: {mem_before:.2f} MB")

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory after {func.__name__}: {mem_after:.2f} MB")
        logger.info(
            f"Memory used by {func.__name__}: {mem_after - mem_before:.2f} MB")

        return result

    return wrapper


class DataFrameSerializer:
    """DataFrame序列化工具类"""
    @staticmethod
    def save_to_parquet(df: pd.DataFrame, path: str, compression: str = 'snappy'):
        """
        保存DataFrame为parquet格式
        Args:
            df: 要保存的DataFrame
            path: 保存路径
            compression: 压缩方式
        """
        df.to_parquet(path, compression=compression)

    @staticmethod
    def load_from_parquet(path: str) -> pd.DataFrame:
        """
        从parquet文件加载DataFrame
        Args:
            path: 文件路径
        Returns:
            加载的DataFrame
        """
        return pd.read_parquet(path)


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    减少DataFrame的内存使用
    Args:
        df: 输入DataFrame
        verbose: 是否打印信息
    Returns:
        优化后的DataFrame
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info(
            f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.2f}% reduction)')

    return df


def save_dict_to_json(dictionary: Dict, filepath: str):
    """
    保存字典到JSON文件
    Args:
        dictionary: 要保存的字典
        filepath: 文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, indent=4, ensure_ascii=False)


def load_dict_from_json(filepath: str) -> Dict:
    """
    从JSON文件加载字典
    Args:
        filepath: 文件路径
    Returns:
        加载的字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class BatchGenerator:
    """批处理数据生成器"""

    def __init__(self, data: pd.DataFrame, batch_size: int, shuffle: bool = True):
        """
        初始化批处理生成器
        Args:
            data: 输入数据
            batch_size: 批大小
            shuffle: 是否打乱数据
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)
        self.indices = np.arange(self.n_samples)

    def __len__(self):
        """返回批次数量"""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __iter__(self):
        """迭代器"""
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self.indices[start_idx:end_idx]
            yield self.data.iloc[batch_indices]


def create_submission_file(predictions: pd.DataFrame, output_path: str):
    """
    创建提交文件
    Args:
        predictions: 预测结果DataFrame
        output_path: 输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存预测结果
    predictions.to_csv(output_path, index=False, encoding='utf-8')

    logger = logging.getLogger(__name__)
    logger.info(f"Submission file created: {output_path}")
    logger.info(f"Number of recommendations: {len(predictions)}")


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    检查数据质量
    Args:
        df: 输入DataFrame
    Returns:
        数据质量报告
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': len(df) - len(df.drop_duplicates()),
        'memory_usage': df.memory_usage().sum() / 1024**2,  # MB
        'column_types': df.dtypes.to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }

    return report


if __name__ == "__main__":
    # 测试日志设置
    config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    setup_logging(config)
    logger = logging.getLogger(__name__)

    # 测试装饰器
    @timer
    @memory_usage
    def test_function():
        logger.info("Testing decorators...")
        time.sleep(1)
        return "Test completed"

    test_function()
