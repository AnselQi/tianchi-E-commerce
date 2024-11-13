import click
import logging
import yaml
from pathlib import Path
import warnings
from datetime import datetime
import json
import os
import pandas as pd 
import torch
import pytorch_lightning as pl

from src.trainer import DeepModelTrainer
from src.utils import setup_logging, timer
from src.data_processing import DataProcessor
from src.data.dataset import RecommendationDataset
from src.models.deep_recommender import DeepRecommender

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

@click.group()
def cli():
    """移动推荐算法竞赛命令行工具"""
    pass

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='配置文件路径')
@click.option('--debug', is_flag=True, help='是否开启调试模式')
@timer
def train(config: str, debug: bool):
    """训练模型"""
    # 加载配置
    config = load_config(config)
    if debug:
        config['logging']['level'] = 'DEBUG'
        pl.seed_everything(config['system']['seed'])

    # 设置日志
    setup_logging(config['logging'])
    logger.info("Starting training process...")

    try:
        # 加载数据
        data_processor = DataProcessor(config)
        train_data, val_data = data_processor.prepare_train_val_data()

        # 创建数据集
        train_dataset = RecommendationDataset(train_data, config)
        val_dataset = RecommendationDataset(val_data, config)

        # 初始化训练器
        trainer = DeepModelTrainer(config)

        # 运行训练
        metrics = trainer.train(train_dataset, val_dataset)

        # 保存训练结果
        output_dir = Path(config['data']['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Training completed. Metrics: {metrics}")

    except Exception as e:
        logger.exception("Error occurred during training")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='配置文件路径')
@click.option('--test-date', '-d', default=None, help='测试日期')
@timer
def predict(config: str, test_date: str):
    """生成预测结果"""
    config = load_config(config)
    setup_logging(config['logging'])
    logger.info("Starting prediction process...")

    try:
        # 加载数据
        data_processor = DataProcessor(config)
        test_data = data_processor.prepare_test_data(test_date)
        test_dataset = RecommendationDataset(test_data, config)

        # 加载训练好的模型
        trainer = DeepModelTrainer(config)
        trainer.load_checkpoint(Path(config['data']['paths']['checkpoint_dir']) / 'best.ckpt')

        # 生成预测
        predictions = trainer.predict(test_dataset)
        
        # 处理预测结果
        submission = data_processor.create_submission(predictions, test_data)
        
        # 保存预测结果
        output_dir = Path(config['data']['paths']['output_dir'])
        submission.to_csv(output_dir / 'submission.csv', index=False)
        
        logger.info("Prediction completed")

    except Exception as e:
        logger.exception("Error occurred during prediction")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='配置文件路径')
@timer
def analyze_data(config: str):
    """数据分析"""
    config_dict = load_config(config)
    setup_logging(config_dict['logging'])
    logger.info("Starting data analysis...")

    try:
        processor = DataProcessor(config_dict)
        user_data, item_data = processor.load_data()

        # 数据分析逻辑保持不变...
        analysis = {
            'user_stats': {
                'total_users': user_data['user_id'].nunique(),
                'total_interactions': len(user_data),
                'behavior_counts': user_data['behavior_type'].value_counts().to_dict(),
                'date_range': {
                    'start': str(user_data['time'].min()),
                    'end': str(user_data['time'].max()),
                    'total_days': (user_data['time'].max() - user_data['time'].min()).days + 1
                }
            },
            'item_stats': {
                'total_items': item_data['item_id'].nunique(),
                'total_categories': item_data['category'].nunique(),
                'items_per_category': item_data.groupby('category')['item_id'].nunique().to_dict()
            },
            'sequence_stats': {
                'avg_sequence_length': processor.calculate_sequence_stats(user_data)['avg_length'],
                'max_sequence_length': processor.calculate_sequence_stats(user_data)['max_length']
            }
        }

        # 保存分析结果
        output_dir = Path(config_dict['data']['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'data_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4, default=str)

        logger.info("Data analysis completed")

    except Exception as e:
        logger.exception("Error occurred during data analysis")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='配置文件路径')
@click.option('--debug', is_flag=True, help='是否开启调试模式')
@timer
def run_all(config: str, debug: bool):
    """运行完整流程"""
    config_dict = load_config(config)
    if debug:
        config_dict['logging']['level'] = 'DEBUG'
        pl.seed_everything(config_dict['system']['seed'])

    setup_logging(config_dict['logging'])
    logger.info("Starting full pipeline...")

    try:
        # 1. 数据分析
        logger.info("Step 1: Data Analysis")
        analyze_data.callback(config)

        # 2. 训练模型
        logger.info("Step 2: Model Training")
        train.callback(config, debug)

        # 3. 生成预测
        logger.info("Step 3: Generate Predictions")
        predict.callback(config, None)

        logger.info("Full pipeline completed successfully")

    except Exception as e:
        logger.exception("Error occurred during pipeline execution")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='配置文件路径')
def validate_config(config: str):
    """验证配置文件"""
    try:
        config = load_config(config)

        # 检查必要的配置项
        required_sections = ['data', 'model', 'training', 'device', 'logging']
        for section in required_sections:
            if section not in config:
                raise click.ClickException(f"Missing required section: {section}")

        # 检查数据路径
        data_dir = Path(config['data']['paths']['raw_user_data']).parent
        if not data_dir.exists():
            raise click.ClickException(f"Data directory does not exist: {data_dir}")

        # 检查GPU可用性
        if config['device']['accelerator'] == 'gpu' and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Will use CPU instead.")
            config['device']['accelerator'] = 'cpu'

        click.echo("Configuration is valid!")
        return True

    except Exception as e:
        logger.exception("Configuration validation failed")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()
