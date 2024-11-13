"""
特征工程模块的单元测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.feature_engineering import FeatureEngineer


def test_init_feature_engineer(test_config):
    """测试特征工程器初始化"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        engineer = FeatureEngineer(f.name)

        assert engineer.config == test_config
        assert 'time_windows' in engineer.config['features']


def test_generate_user_features(test_config, sample_user_data):
    """测试用户特征生成"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        engineer = FeatureEngineer(f.name)

        user_features = engineer.generate_user_features(
            sample_user_data, time_window=1)

        # 检查特征列
        expected_columns = [
            'behavior_1_count_1d',
            'behavior_2_count_1d',
            'behavior_3_count_1d',
            'behavior_4_count_1d',
            'active_days_1d',
            'active_hours_1d'
        ]

        assert all(col in user_features.columns for col in expected_columns)
        assert len(user_features) == sample_user_data['user_id'].nunique()


def test_generate_item_features(test_config, sample_user_data):
    """测试商品特征生成"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        engineer = FeatureEngineer(f.name)

        item_features = engineer.generate_item_features(
            sample_user_data, time_window=1)

        # 检查特征列
        expected_columns = [
            'item_behavior_1_count_1d',
            'item_behavior_2_count_1d',
            'item_behavior_3_count_1d',
            'item_behavior_4_count_1d',
            'item_cvr_1d',
            'item_unique_users_1d'
        ]

        assert all(col in item_features.columns for col in expected_columns)
        assert len(item_features) == sample_user_data['item_id'].nunique()


def test_generate_user_item_features(test_config, sample_user_data):
    """测试用户-商品交叉特征生成"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        engineer = FeatureEngineer(f.name)

        ui_features = engineer.generate_user_item_features(
            sample_user_data, time_window=1)

        # 检查特征列
        expected_columns = [
            'ui_behavior_count_1d',
            'ui_behavior_max_1d',
            'ui_behavior_min_1d',
            'ui_time_count_1d',
            'ui_time_max_1d',
            'ui_time_min_1d',
            'ui_time_span_1d'
        ]

        assert all(col in ui_features.columns for col in expected_columns)
        assert isinstance(ui_features.index, pd.MultiIndex)


def test_generate_all_features(test_config, sample_user_data):
    """测试生成所有特征"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        engineer = FeatureEngineer(f.name)

        all_features = engineer.generate_features(
            sample_user_data,
            end_date='2014-12-04'
        )

        # 检查特征字典
        assert all(f'user_{w}d' in all_features
                   for w in test_config['features']['time_windows'])
        assert all(f'item_{w}d' in all_features
                   for w in test_config['features']['time_windows'])
        assert all(f'user_item_{w}d' in all_features
                   for w in test_config['features']['time_windows'])


def test_save_and_load_features(test_config, sample_user_data, temp_data_dir):
    """测试特征保存和加载"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        engineer = FeatureEngineer(f.name)

        # 生成特征
        features = engineer.generate_features(
            sample_user_data,
            end_date='2014-12-04'
        )
