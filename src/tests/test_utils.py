"""
工具函数模块的单元测试
"""

import pytest
import pandas as pd
import numpy as np
import logging
import tempfile
import time
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock

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

# ==================== 日志设置测试 ====================


def test_setup_logging(temp_data_dir):
    """测试日志设置"""
    config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }

    # 设置日志
    setup_logging(config)
    logger = logging.getLogger('test_logger')

    # 测试日志级别
    assert logger.getEffectiveLevel() == logging.INFO

    # 测试日志记录
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', dir=temp_data_dir) as log_file:
        handler = logging.FileHandler(log_file.name)
        logger.addHandler(handler)

        test_message = "Test log message"
        logger.info(test_message)

        # 验证日志文件内容
        with open(log_file.name, 'r') as f:
            log_content = f.read()
            assert test_message in log_content

# ==================== 装饰器测试 ====================


def test_timer_decorator():
    """测试计时器装饰器"""
    @timer
    def slow_function():
        time.sleep(0.1)
        return "Done"

    # 捕获日志
    with patch('logging.getLogger') as mock_logger:
        result = slow_function()

        # 验证结果
        assert result == "Done"

        # 验证日志调用
        mock_logger.return_value.info.assert_called()
        calls = mock_logger.return_value.info.call_args_list
        assert any('Starting' in str(call) for call in calls)
        assert any('Finished' in str(call) for call in calls)


def test_memory_usage_decorator():
    """测试内存使用装饰器"""
    @memory_usage
    def memory_intensive_function():
        # 创建大数组来消耗内存
        array = np.zeros((1000, 1000))
        return array

    # 捕获日志
    with patch('logging.getLogger') as mock_logger:
        result = memory_intensive_function()

        # 验证结果
        assert isinstance(result, np.ndarray)

        # 验证日志调用
        mock_logger.return_value.info.assert_called()
        calls = mock_logger.return_value.info.call_args_list
        assert any('Memory before' in str(call) for call in calls)
        assert any('Memory after' in str(call) for call in calls)

# ==================== DataFrame序列化测试 ====================


def test_dataframe_serializer(temp_data_dir):
    """测试DataFrame序列化工具"""
    # 创建测试数据
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c']
    })

    file_path = Path(temp_data_dir) / 'test.parquet'

    # 测试保存
    DataFrameSerializer.save_to_parquet(df, file_path)
    assert file_path.exists()

    # 测试加载
    loaded_df = DataFrameSerializer.load_from_parquet(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)

# ==================== 内存优化测试 ====================


def test_reduce_memory_usage():
    """测试内存使用优化"""
    # 创建测试数据
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 1000),
        'float_col': np.random.random(1000),
        'small_int_col': np.random.randint(0, 5, 1000)
    })

    # 记录原始内存使用
    original_memory = df.memory_usage().sum()

    # 优化内存
    optimized_df = reduce_memory_usage(df, verbose=False)

    # 验证优化后的内存使用
    assert optimized_df.memory_usage().sum() <= original_memory

    # 验证数据一致性
    pd.testing.assert_frame_equal(df, optimized_df, check_dtype=False)

# ==================== JSON操作测试 ====================


def test_json_operations(temp_data_dir):
    """测试JSON文件操作"""
    test_dict = {
        'str_key': 'value',
        'int_key': 42,
        'list_key': [1, 2, 3],
        'nested': {'a': 1, 'b': 2}
    }

    file_path = Path(temp_data_dir) / 'test.json'

    # 测试保存
    save_dict_to_json(test_dict, file_path)
    assert file_path.exists()

    # 测试加载
    loaded_dict = load_dict_from_json(file_path)
    assert loaded_dict == test_dict

# ==================== 批处理生成器测试 ====================


def test_batch_generator():
    """测试批处理数据生成器"""
    # 创建测试数据
    df = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])
    batch_size = 10

    # 创建生成器
    generator = BatchGenerator(df, batch_size=batch_size)

    # 验证批次数量
    assert len(generator) == 10  # 100/10 = 10

    # 验证批次
    batches = list(generator)
    assert len(batches) == 10
    assert all(len(batch) == batch_size for batch in batches[:-1])  # 除了最后一个批次

    # 验证数据完整性
    all_data = pd.concat(batches)
    assert len(all_data) == len(df)

    # 测试打乱功能
    generator_shuffle = BatchGenerator(df, batch_size=batch_size, shuffle=True)
    shuffled_data = pd.concat(list(generator_shuffle))
    assert len(shuffled_data) == len(df)
    assert not shuffled_data.equals(df)  # 数据应该被打乱

# ==================== 提交文件测试 ====================


def test_create_submission_file(temp_data_dir):
    """测试创建提交文件"""
    # 创建测试预测结果
    predictions = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3'],
        'item_id': ['item1', 'item2', 'item3']
    })

    output_path = Path(temp_data_dir) / 'submission.csv'

    # 创建提交文件
    create_submission_file(predictions, output_path)

    # 验证文件
    assert output_path.exists()
    loaded_pred = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(predictions, loaded_pred)

# ==================== 数据质量检查测试 ====================


def test_check_data_quality():
    """测试数据质量检查"""
    # 创建测试数据
    df = pd.DataFrame({
        'int_col': [1, 2, None, 4],
        'float_col': [1.1, None, 3.3, 4.4],
        'str_col': ['a', 'b', 'c', 'd']
    })

    # 获取质量报告
    report = check_data_quality(df)

    # 验证报告内容
    assert 'total_rows' in report
    assert report['total_rows'] == 4
    assert 'total_columns' in report
    assert report['total_columns'] == 3
    assert 'missing_values' in report
    assert report['missing_values']['int_col'] == 1
    assert report['missing_values']['float_col'] == 1
    assert 'duplicates' in report
    assert 'column_types' in report
    assert 'unique_values' in report

# ==================== 错误处理测试 ====================


def test_error_handling():
    """测试错误处理"""
    # 测试无效的JSON文件加载
    with pytest.raises(FileNotFoundError):
        load_dict_from_json('nonexistent.json')

    # 测试无效的Parquet文件加载
    with pytest.raises(FileNotFoundError):
        DataFrameSerializer.load_from_parquet('nonexistent.parquet')

    # 测试批处理生成器的参数验证
    with pytest.raises(ValueError):
        BatchGenerator(pd.DataFrame(), batch_size=0)

# ==================== 边界情况测试 ====================


def test_edge_cases():
    """测试边界情况"""
    # 空DataFrame的内存优化
    empty_df = pd.DataFrame()
    optimized_empty_df = reduce_memory_usage(empty_df, verbose=False)
    assert len(optimized_empty_df) == 0

    # 空DataFrame的批处理
    generator = BatchGenerator(empty_df, batch_size=10)
    assert len(list(generator)) == 0

    # 空字典的JSON操作
    with tempfile.NamedTemporaryFile(suffix='.json') as tf:
        save_dict_to_json({}, tf.name)
        loaded_dict = load_dict_from_json(tf.name)
        assert loaded_dict == {}


if __name__ == '__main__':
    pytest.main([__file__])
