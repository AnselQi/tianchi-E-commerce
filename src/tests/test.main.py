"""
主程序模块的单元测试
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch, Mock
from click.testing import CliRunner
import pandas as pd
import logging

from main import cli, train, predict, analyze_data, run_all, validate_config


@pytest.fixture
def runner():
    """Click CLI测试运行器"""
    return CliRunner()


@pytest.fixture
def mock_trainer():
    """Mock训练器"""
    mock = Mock()
    mock.run_training.return_value = {
        'precision': 0.75,
        'recall': 0.80,
        'f1': 0.77
    }
    mock.generate_submission.return_value = None
    return mock


@pytest.fixture
def test_config_file():
    """测试配置文件"""
    config = {
        'data': {
            'raw_user_data': 'data/raw/tianchi_fresh_comp_train_user_2w.csv',
            'raw_item_data': 'data/raw/tianchi_fresh_comp_train_item_2w.csv',
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
                'learning_rate': 0.05
            }
        },
        'training': {
            'train_end_date': '2014-12-18',
            'pred_date': '2014-12-19',
            'top_k': 20,
            'batch_size': 1024
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }

    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name


def test_cli_help(runner):
    """测试CLI帮助信息"""
    # 测试主命令帮助
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert '移动推荐算法竞赛命令行工具' in result.output

    # 测试子命令帮助
    commands = ['train', 'predict', 'analyze-data',
                'run-all', 'validate-config']
    for cmd in commands:
        result = runner.invoke(cli, [cmd, '--help'])
        assert result.exit_code == 0
        assert cmd in result.output


@patch('main.ModelTrainer')
def test_train_command(mock_trainer_cls, runner, test_config_file, mock_trainer):
    """测试训练命令"""
    mock_trainer_cls.return_value = mock_trainer

    # 测试正常训练
    result = runner.invoke(cli, ['train', '-c', test_config_file])
    assert result.exit_code == 0
    assert mock_trainer.run_training.called

    # 测试调试模式
    result = runner.invoke(cli, ['train', '-c', test_config_file, '--debug'])
    assert result.exit_code == 0

    # 测试配置文件不存在
    result = runner.invoke(cli, ['train', '-c', 'nonexistent.yaml'])
    assert result.exit_code != 0


@patch('main.ModelTrainer')
def test_predict_command(mock_trainer_cls, runner, test_config_file, mock_trainer):
    """测试预测命令"""
    mock_trainer_cls.return_value = mock_trainer

    # 测试默认预测
    result = runner.invoke(cli, ['predict', '-c', test_config_file])
    assert result.exit_code == 0
    assert mock_trainer.generate_submission.called

    # 测试指定日期预测
    test_date = '2014-12-20'
    result = runner.invoke(
        cli, ['predict', '-c', test_config_file, '-d', test_date])
    assert result.exit_code == 0
    mock_trainer.generate_submission.assert_called_with(test_date)


@patch('main.DataProcessor')
def test_analyze_data_command(mock_processor_cls, runner, test_config_file):
    """测试数据分析命令"""
    # 配置Mock
    mock_processor = Mock()
    mock_processor.load_data.return_value = (
        pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'item_id': ['i1', 'i2'],
            'behavior_type': [1, 2],
            'time': pd.date_range('2014-12-01', periods=2)
        }),
        pd.DataFrame({
            'item_id': ['i1', 'i2'],
            'item_category': [1, 2]
        })
    )
    mock_processor_cls.return_value = mock_processor

    # 测试分析命令
    result = runner.invoke(cli, ['analyze-data', '-c', test_config_file])
    assert result.exit_code == 0
    assert mock_processor.load_data.called


@patch('main.ModelTrainer')
def test_run_all_command(mock_trainer_cls, runner, test_config_file, mock_trainer):
    """测试运行完整流程命令"""
    mock_trainer_cls.return_value = mock_trainer

    # 测试完整流程
    result = runner.invoke(cli, ['run-all', '-c', test_config_file])
    assert result.exit_code == 0
    assert mock_trainer.run_training.called
    assert mock_trainer.generate_submission.called

    # 测试调试模式
    result = runner.invoke(cli, ['run-all', '-c', test_config_file, '--debug'])
    assert result.exit_code == 0


def test_validate_config_command(runner, test_config_file):
    """测试配置验证命令"""
    # 测试有效配置
    result = runner.invoke(cli, ['validate-config', '-c', test_config_file])
    assert result.exit_code == 0
    assert "Configuration is valid!" in result.output

    # 测试无效配置
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump({'invalid': 'config'}, f)
        f.flush()
        result = runner.invoke(cli, ['validate-config', '-c', f.name])
        assert result.exit_code != 0


def test_config_loading(test_config_file):
    """测试配置加载"""
    # 测试正常加载
    config = yaml.safe_load(open(test_config_file))
    assert 'data' in config
    assert 'features' in config
    assert 'model' in config
    assert 'training' in config
    assert 'logging' in config


@patch('main.ModelTrainer')
def test_error_handling(mock_trainer_cls, runner, test_config_file):
    """测试错误处理"""
    # 模拟训练错误
    mock_trainer = Mock()
    mock_trainer.run_training.side_effect = Exception("Training failed")
    mock_trainer_cls.return_value = mock_trainer

    result = runner.invoke(cli, ['train', '-c', test_config_file])
    assert result.exit_code != 0
    assert "Training failed" in str(result.exception)


@patch('main.ModelTrainer')
def test_logging_setup(mock_trainer_cls, runner, test_config_file, caplog):
    """测试日志设置"""
    with caplog.at_level(logging.INFO):
        result = runner.invoke(cli, ['train', '-c', test_config_file])
        assert any('Starting training process' in record.message
                   for record in caplog.records)


def test_output_directory_creation(runner, test_config_file, temp_data_dir):
    """测试输出目录创建"""
    # 修改配置文件中的输出目录
    config = yaml.safe_load(open(test_config_file))
    config['data']['output_dir'] = str(temp_data_dir)

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(config, f)
        f.flush()

        # 运行命令
        result = runner.invoke(cli, ['train', '-c', f.name])
        assert Path(temp_data_dir).exists()


@patch('main.ModelTrainer')
def test_metrics_output(mock_trainer_cls, runner, test_config_file, mock_trainer, temp_data_dir):
    """测试指标输出"""
    # 设置输出目录
    config = yaml.safe_load(open(test_config_file))
    config['data']['output_dir'] = str(temp_data_dir)

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(config, f)
        f.flush()

        mock_trainer_cls.return_value = mock_trainer

        # 运行训练
        result = runner.invoke(cli, ['train', '-c', f.name])

        # 验证指标文件
        metrics_file = Path(temp_data_dir) / 'training_metrics.json'
        assert metrics_file.exists()

        with open(metrics_file) as f:
            metrics = json.load(f)
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics


if __name__ == '__main__':
    pytest.main([__file__])
