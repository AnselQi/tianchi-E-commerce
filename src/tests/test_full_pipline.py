"""
完整推荐系统流水线的集成测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml
import shutil
import logging
from datetime import datetime, timedelta

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model import RecommendationModel
from src.trainer import ModelTrainer
from src.utils import setup_logging


@pytest.fixture(scope="module")
def test_data():
    """创建测试数据"""
    # 生成用户行为数据
    user_data = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u3'] * 5,
        'item_id': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'] * 5,
        'behavior_type': [1, 2, 3, 4, 1, 2] * 5,
        'user_geohash': ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'] * 5,
        'item_category': [101, 102, 103, 104, 105, 106] * 5,
        'time': pd.date_range('2014-12-01', periods=30, freq='H')
    })

    # 生成商品数据
    item_data = pd.DataFrame({
        'item_id': [f'i{i}' for i in range(1, 7)],
        'item_geohash': [f'g{i}' for i in range(1, 7)],
        'item_category': [100 + i for i in range(1, 7)]
    })

    return user_data, item_data


@pytest.fixture(scope="module")
def test_config():
    """创建测试配置"""
    return {
        'data': {
            'raw_user_data': 'data/raw/user_data.csv',
            'raw_item_data': 'data/raw/item_data.csv',
            'processed_data_dir': 'data/processed',
            'output_dir': 'data/output'
        },
        'features': {
            'time_windows': [1, 3, 7],
            'categorical_features': ['user_id', 'item_id', 'item_category'],
            'numerical_features': ['behavior_count', 'category_count']
        },
        'model': {
            'type': 'lightgbm',
            'params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }
        },
        'training': {
            'train_start_date': '2014-12-01',
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


@pytest.fixture(scope="module")
def test_env(test_data, test_config, tmp_path_factory):
    """设置测试环境"""
    # 创建临时目录
    base_dir = tmp_path_factory.mktemp("recommendation_test")
    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    # 保存测试数据
    user_data, item_data = test_data
    user_data.to_csv(raw_dir / "user_data.csv", index=False)
    item_data.to_csv(raw_dir / "item_data.csv", index=False)

    # 更新配置路径
    config = test_config.copy()
    config['data'].update({
        'raw_user_data': str(raw_dir / "user_data.csv"),
        'raw_item_data': str(raw_dir / "item_data.csv"),
        'processed_data_dir': str(data_dir / "processed"),
        'output_dir': str(data_dir / "output")
    })

    # 保存配置
    config_path = base_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return {
        'base_dir': base_dir,
        'config_path': config_path,
        'config': config
    }


class TestFullPipeline:
    """完整推荐系统流水线测试"""

    def test_data_processing(self, test_env, caplog):
        """测试数据处理阶段"""
        with caplog.at_level(logging.INFO):
            processor = DataProcessor(str(test_env['config_path']))

            # 加载并处理数据
            user_data, item_data = processor.load_data()
            processed_user_data, processed_item_data = processor.preprocess_data(
                user_data, item_data)

            # 验证处理结果
            assert isinstance(processed_user_data, pd.DataFrame)
            assert isinstance(processed_item_data, pd.DataFrame)
            assert 'user_id_encoded' in processed_user_data.columns
            assert 'item_id_encoded' in processed_user_data.columns
            assert 'time' in processed_user_data.columns
            assert pd.api.types.is_datetime64_any_dtype(
                processed_user_data['time'])

            # 验证日志
            assert any(
                'Loading raw data' in record.message for record in caplog.records)
            assert any(
                'Preprocessing data' in record.message for record in caplog.records)

    def test_feature_engineering(self, test_env, caplog):
        """测试特征工程阶段"""
        with caplog.at_level(logging.INFO):
            # 准备数据
            processor = DataProcessor(str(test_env['config_path']))
            user_data, item_data = processor.load_data()
            processed_user_data, processed_item_data = processor.preprocess_data(
                user_data, item_data)

            # 生成特征
            engineer = FeatureEngineer(str(test_env['config_path']))
            features = engineer.generate_features(
                processed_user_data,
                test_env['config']['training']['train_end_date']
            )

            # 验证特征
            for window in test_env['config']['features']['time_windows']:
                assert f'user_{window}d' in features
                assert f'item_{window}d' in features
                assert f'user_item_{window}d' in features

            # 验证特征内容
            for feature_df in features.values():
                assert isinstance(feature_df, pd.DataFrame)
                assert not feature_df.empty

    def test_model_training(self, test_env, caplog):
        """测试模型训练阶段"""
        with caplog.at_level(logging.INFO):
            # 准备数据和特征
            processor = DataProcessor(str(test_env['config_path']))
            user_data, item_data = processor.load_data()
            processed_user_data, processed_item_data = processor.preprocess_data(
                user_data, item_data)

            engineer = FeatureEngineer(str(test_env['config_path']))
            features = engineer.generate_features(
                processed_user_data,
                test_env['config']['training']['train_end_date']
            )

            # 训练模型
            model = RecommendationModel(str(test_env['config_path']))
            X, y = model.prepare_training_data(
                features,
                processed_user_data,
                test_env['config']['training']['pred_date']
            )
            model.train(X, y)

            # 验证模型
            assert model.model is not None
            assert model.feature_importance is not None

            # 生成预测
            recommendations = model.generate_recommendations(features)
            assert len(recommendations) > 0

    def test_full_training_pipeline(self, test_env, caplog):
        """测试完整训练流水线"""
        with caplog.at_level(logging.INFO):
            trainer = ModelTrainer(str(test_env['config_path']))
            metrics = trainer.run_training()

            # 验证训练结果
            assert isinstance(metrics, dict)
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert all(0 <= v <= 1 for v in metrics.values())

            # 验证输出文件
            output_dir = Path(test_env['config']['data']['output_dir'])
            assert (output_dir / 'model.pkl').exists()
            assert (output_dir / 'feature_importance.csv').exists()
            assert (output_dir / 'training_history.csv').exists()

    def test_prediction_pipeline(self, test_env, caplog):
        """测试预测流水线"""
        with caplog.at_level(logging.INFO):
            trainer = ModelTrainer(str(test_env['config_path']))

            # 生成预测
            trainer.generate_submission(
                test_env['config']['training']['pred_date'])

            # 验证提交文件
            submission_path = Path(
                test_env['config']['data']['output_dir']) / 'submission.csv'
            assert submission_path.exists()

            # 验证提交文件格式
            submission = pd.read_csv(submission_path)
            assert 'user_id' in submission.columns
            assert 'item_id' in submission.columns
            assert not submission.duplicated().any()

    def test_error_recovery(self, test_env, caplog):
        """测试错误恢复能力"""
        with caplog.at_level(logging.INFO):
            trainer = ModelTrainer(str(test_env['config_path']))

            # 模拟中断后的恢复
            processed_dir = Path(
                test_env['config']['data']['processed_data_dir'])
            if processed_dir.exists():
                shutil.rmtree(processed_dir)

            # 重新运行应该能够恢复
            metrics = trainer.run_training()
            assert isinstance(metrics, dict)

    def test_data_consistency(self, test_env):
        """测试数据一致性"""
        trainer = ModelTrainer(str(test_env['config_path']))

        # 运行两次训练
        metrics1 = trainer.run_training()
        metrics2 = trainer.run_training()

        # 验证结果一致性
        assert metrics1['precision'] == metrics2['precision']
        assert metrics1['recall'] == metrics2['recall']
        assert metrics1['f1'] == metrics2['f1']

    def test_performance_benchmarks(self, test_env, caplog):
        """测试性能基准"""
        with caplog.at_level(logging.INFO):
            trainer = ModelTrainer(str(test_env['config_path']))

            start_time = datetime.now()
            metrics = trainer.run_training()
            end_time = datetime.now()

            # 验证运行时间
            duration = (end_time - start_time).total_seconds()
            assert duration < 300  # 假设运行时间应该小于5分钟

            # 验证内存使用
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            assert memory_usage < 2048  # 假设内存使用应该小于2GB


if __name__ == '__main__':
    pytest.main([__file__])
