"""
数据处理模块的单元测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.data_processing import DataProcessor


def test_init_data_processor(test_config, temp_data_dir):
    """测试数据处理器初始化"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        processor = DataProcessor(f.name)

        assert processor.config == test_config
        assert all(encoder_name in processor.encoders
                   for encoder_name in ['user_id', 'item_id', 'item_category'])


def test_preprocess_data(test_config, sample_user_data, sample_item_data):
    """测试数据预处理"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        processor = DataProcessor(f.name)

        processed_user_data, processed_item_data = processor.preprocess_data(
            sample_user_data.copy(),
            sample_item_data.copy()
        )

        # 检查时间处理
        assert pd.api.types.is_datetime64_any_dtype(
            processed_user_data['time'])

        # 检查编码列
        assert 'user_id_encoded' in processed_user_data.columns
        assert 'item_id_encoded' in processed_user_data.columns
        assert 'category_encoded' in processed_user_data.columns

        # 检查缺失值处理
        assert not processed_user_data['user_geohash'].isna().any()
        assert not processed_item_data['item_geohash'].isna().any()


def test_save_and_load_processed_data(test_config, sample_user_data,
                                      sample_item_data, temp_data_dir):
    """测试数据保存和加载"""
    # 修改配置中的数据目录
    test_config['data']['processed_data_dir'] = temp_data_dir

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        processor = DataProcessor(f.name)

        # 处理并保存数据
        processed_user_data, processed_item_data = processor.preprocess_data(
            sample_user_data.copy(),
            sample_item_data.copy()
        )
        processor.save_processed_data(processed_user_data, processed_item_data)

        # 加载数据
        loaded_user_data, loaded_item_data, loaded_encoders = processor.load_processed_data()

        # 验证数据
        pd.testing.assert_frame_equal(processed_user_data, loaded_user_data)
        pd.testing.assert_frame_equal(processed_item_data, loaded_item_data)
        assert all(encoder_name in loaded_encoders
                   for encoder_name in ['user_id', 'item_id', 'item_category'])


def test_error_handling(test_config):
    """测试错误处理"""
    # 测试文件不存在的情况
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        processor = DataProcessor(f.name)

        with pytest.raises(FileNotFoundError):
            processor.load_processed_data()


def test_edge_cases(test_config, temp_data_dir):
    """测试边界情况"""
    # 空数据
    empty_user_data = pd.DataFrame(columns=['user_id', 'item_id', 'behavior_type',
                                            'user_geohash', 'item_category', 'time'])
    empty_item_data = pd.DataFrame(
        columns=['item_id', 'item_geohash', 'item_category'])

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        processor = DataProcessor(f.name)

        processed_user_data, processed_item_data = processor.preprocess_data(
            empty_user_data,
            empty_item_data
        )

        assert len(processed_user_data) == 0
        assert len(processed_item_data) == 0


def test_data_consistency(test_config, sample_user_data, sample_item_data):
    """测试数据一致性"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        processor = DataProcessor(f.name)

        processed_user_data, processed_item_data = processor.preprocess_data(
            sample_user_data.copy(),
            sample_item_data.copy()
        )

        # 检查编码的一致性
        user_items = set(processed_user_data['item_id'].unique())
        item_items = set(processed_item_data['item_id'].unique())
        assert user_items.issubset(item_items)

        # 检查类别的一致性
        assert all(processed_user_data['item_category'].isin(
            processed_item_data['item_category']))


if __name__ == '__main__':
    pytest.main([__file__])
