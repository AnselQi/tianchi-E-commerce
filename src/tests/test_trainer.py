"""
模型训练器的单元测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml
import shutil
import logging
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.trainer import ModelTrainer


@pytest.fixture
def mock_data_processor():
    """Mock数据处理器"""
    mock = Mock()
    mock.load_processed_data.return_value = (
        pd.DataFrame({
            'user_id_encoded': [0, 0, 1, 1],
            'item_id_encoded': [0, 1, 0, 2],
            'behavior_type': [1, 2, 3, 4],
            'time': pd.date_range('2014-12-01', periods=4, freq='D')
        }),
        pd.DataFrame({
            'item_id_encoded': [0, 1, 2],
            'item_category_encoded': [0, 1, 2]
        }),
        {'user_id': Mock(), 'item_id': Mock(), 'item_category': Mock()}
    )
    return mock


@pytest.fixture
def mock_feature_engineer():
    """Mock特征工程器"""
    mock = Mock()
    mock.generate_features.return_value = {
        'user_1d': pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4]
        }, index=[0, 1]),
        'item_1d': pd.DataFrame({
            'feature3': [5, 6, 7],
            'feature4': [8, 9, 10]
        }, index=[0, 1, 2])
    }
    return mock


def test_init_trainer(test_config):
    """测试训练器初始化"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        trainer = ModelTrainer(f.name)

        assert trainer.config == test_config
        assert len(trainer.metrics_history) == 0


@patch('src.trainer.DataProcessor')
@patch('src.trainer.FeatureEngineer')
@patch('src.trainer.RecommendationModel')
def test_prepare_data(mock_model_cls, mock_engineer_cls,
                      mock_processor_cls, test_config, mock_data_processor):
    """测试数据准备"""
    mock_processor_cls.return_value = mock_data_processor

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        trainer = ModelTrainer(f.name)

        user_data, item_data, encoders = trainer.prepare_data()

        # 验证数据加载调用
        mock_data_processor.load_processed_data.assert_called_once()

        # 验证返回数据
        assert isinstance(user_data, pd.DataFrame)
        assert isinstance(item_data, pd.DataFrame)
        assert isinstance(encoders, dict)


@patch('src.trainer.DataProcessor')
@patch('src.trainer.FeatureEngineer')
@patch('src.trainer.RecommendationModel')
def test_prepare_features(mock_model_cls, mock_engineer_cls,
                          mock_processor_cls, test_config,
                          mock_feature_engineer, mock_data_processor):
    """测试特征准备"""
    mock_processor_cls.return_value = mock_data_processor
    mock_engineer_cls.return_value = mock_feature_engineer

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        trainer = ModelTrainer(f.name)

        # 准备数据
        user_data, _, _ = trainer.prepare_data()

        # 生成特征
        features = trainer.prepare_features(user_data, '2014-12-02')

        # 验证特征生成调用
        mock_feature_engineer.generate_features.assert_called_once()

        # 验证特征格式
        assert isinstance(features, dict)
        assert 'user_1d' in features
        assert 'item_1d' in features


@patch('src.trainer.DataProcessor')
@patch('src.trainer.FeatureEngineer')
@patch('src.trainer.RecommendationModel')
def test_train_and_evaluate(mock_model_cls, mock_engineer_cls,
                            mock_processor_cls, test_config,
                            mock_feature_engineer, mock_data_processor):
    """测试训练和评估"""
    mock_processor_cls.return_value = mock_data_processor
    mock_engineer_cls.return_value = mock_feature_engineer

    # 配置模型Mock
    mock_model = Mock()
    mock_model.prepare_training_data.return_value = (
        pd.DataFrame(np.random.rand(10, 4)),
        pd.Series(np.random.randint(0, 2, 10))
    )
    mock_model.generate_recommendations.return_value = [
        (0, 0, 0.9), (0, 1, 0.8), (1, 0, 0.7)
    ]
    mock_model_cls.return_value = mock_model

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        trainer = ModelTrainer(f.name)

        metrics = trainer.train_and_evaluate(
            user_data=mock_data_processor.load_processed_data()[0],
            train_end_date='2014-12-01',
            valid_end_date='2014-12-02'
        )

        # 验证训练流程
        assert mock_model.prepare_training_data.called
        assert mock_model.train.called
        assert mock_model.generate_recommendations.called

        # 验证评估指标
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())


def test_evaluate_recommendations(test_config):
    """测试推荐评估"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        trainer = ModelTrainer(f.name)

        # 创建测试数据
        user_data = pd.DataFrame({
            'user_id_encoded': [0, 0, 1, 1],
            'item_id_encoded': [0, 1, 0, 2],
            'behavior_type': [4, 4, 4, 4],  # 所有都是购买行为
            'time': pd.date_range('2014-12-02', periods=4, freq='H')
        })

        recommendations = [
            (0, 0, 0.9),  # True Positive
            (0, 2, 0.8),  # False Positive
            (1, 0, 0.7),  # True Positive
            (1, 1, 0.6)   # False Positive
        ]

        metrics = trainer.evaluate_recommendations(
            recommendations,
            user_data,
            '2014-12-02'
        )

        assert metrics['precision'] == 0.5  # 2/4
        assert metrics['recall'] > 0  # 实际购买中被预测出的比例
        assert 0 <= metrics['f1'] <= 1


@patch('src.trainer.DataProcessor')
@patch('src.trainer.FeatureEngineer')
@patch('src.trainer.RecommendationModel')
def test_run_training(mock_model_cls, mock_engineer_cls,
                      mock_processor_cls, test_config,
                      mock_feature_engineer, mock_data_processor, temp_data_dir):
    """测试完整训练流程"""
    # 设置Mock
    mock_processor_cls.return_value = mock_data_processor
    mock_engineer_cls.return_value = mock_feature_engineer

    mock_model = Mock()
    mock_model.prepare_training_data.return_value = (
        pd.DataFrame(np.random.rand(10, 4)),
        pd.Series(np.random.randint(0, 2, 10))
    )
    mock_model.generate_recommendations.return_value = [
        (0, 0, 0.9), (0, 1, 0.8), (1, 0, 0.7)
    ]
    mock_model_cls.return_value = mock_model

    # 修改配置以使用临时目录
    test_config['data']['output_dir'] = temp_data_dir

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        trainer = ModelTrainer(f.name)

        # 运行训练
        metrics = trainer.run_training()

        # 验证完整流程
        assert mock_data_processor.load_processed_data.called
        assert mock_feature_engineer.generate_features.called
        assert mock_model.train.called
        assert len(trainer.metrics_history) > 0

        # 验证输出文件
        output_dir = Path(temp_data_dir)
        assert (output_dir / 'model.pkl').exists()
        assert (output_dir / 'feature_importance.csv').exists()
        assert (output_dir / 'training_history.csv').exists()


@patch('src.trainer.DataProcessor')
@patch('src.trainer.FeatureEngineer')
@patch('src.trainer.RecommendationModel')
def test_generate_submission(mock_model_cls, mock_engineer_cls,
                             mock_processor_cls, test_config,
                             mock_feature_engineer, mock_data_processor,
                             temp_data_dir):
    """测试生成提交文件"""
    # 设置Mock
    mock_processor_cls.return_value = mock_data_processor
    mock_engineer_cls.return_value = mock_feature_engineer

    mock_model = Mock()
    mock_model.generate_recommendations.return_value = [
        (0, 0, 0.9), (0, 1, 0.8), (1, 0, 0.7)
    ]
    mock_model_cls.return_value = mock_model

    # 修改配置以使用临时目录
    test_config['data']['output_dir'] = temp_data_dir

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        trainer = ModelTrainer(f.name)

        # 生成提交文件
        trainer.generate_submission('2014-12-19')

        # 验证提交文件
        submission_path = Path(temp_data_dir) / 'submission.csv'
        assert submission_path.exists()

        submission_df = pd.read_csv(submission_path)
        assert 'user_id' in submission_df.columns
        assert 'item_id' in submission_df.columns
        assert len(submission_df) > 0


def test_error_handling(test_config):
    """测试错误处理"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        trainer = ModelTrainer(f.name)

        # 测试无效日期
        with pytest.raises(ValueError):
            trainer.train_and_evaluate(
                pd.DataFrame(),
                'invalid_date',
                '2014-12-02'
            )

        # 测试空数据
        with pytest.raises(ValueError):
            trainer.evaluate_recommendations(
                [],
                pd.DataFrame(),
                '2014-12-02'
            )


def test_logging(test_config, caplog):
    """测试日志记录"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()

        with caplog.at_level(logging.INFO):
            trainer = ModelTrainer(f.name)

            # 验证是否记录了初始化日志
            assert any('Starting' in record.message
                       for record in caplog.records)


if __name__ == '__main__':
    pytest.main([__file__])
