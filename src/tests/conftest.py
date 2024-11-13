import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_config():
    """测试配置fixture"""
    return {
        'data': {
            'raw_user_data': 'data/raw/tianchi_fresh_comp_train_user_2w.csv',
            'raw_item_data': 'data/raw/tianchi_fresh_comp_train_item_2w.csv',
            'processed_data_dir': 'data/processed',
            'output_dir': 'data/output'
        },
        'features': {
            'time_windows': [1, 3, 7],
            'categorical_features': ['user_id', 'item_id', 'item_category'],
            'numerical_features': ['behavior_count', 'category_count', 'conversion_rate']
        },
        'model': {
            'type': 'lightgbm',
            'params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05
            }
        },
        'training': {
            'train_start_date': '2014-11-18',
            'train_end_date': '2014-12-18',
            'pred_date': '2014-12-19',
            'validation_days': 1,
            'top_k': 20,
            'batch_size': 1024,
            'num_epochs': 10
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }


@pytest.fixture(scope="session")
def temp_data_dir():
    """临时数据目录fixture"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_user_data():
    """示例用户数据fixture"""
    return pd.DataFrame({
        'user_id': ['user1', 'user1', 'user2', 'user2'],
        'item_id': ['item1', 'item2', 'item1', 'item3'],
        'behavior_type': [1, 2, 3, 4],
        'user_geohash': ['abc', 'def', 'ghi', 'jkl'],
        'item_category': [101, 102, 101, 103],
        'time': pd.date_range('2014-12-01', periods=4, freq='D')
    })


@pytest.fixture
def sample_item_data():
    """示例商品数据fixture"""
    return pd.DataFrame({
        'item_id': ['item1', 'item2', 'item3'],
        'item_geohash': ['xyz', 'uvw', 'rst'],
        'item_category': [101, 102, 103]
    })


@pytest.fixture
def sample_features():
    """示例特征数据fixture"""
    user_features = pd.DataFrame({
        'behavior_1_count_1d': [1, 2],
        'behavior_2_count_1d': [2, 1]
    }, index=['user1', 'user2'])

    item_features = pd.DataFrame({
        'item_behavior_1_count_1d': [2, 1, 1],
        'item_cvr_1d': [0.5, 0.3, 0.4]
    }, index=['item1', 'item2', 'item3'])

    return {
        'user_1d': user_features,
        'item_1d': item_features
    }
