"""
移动推荐算法竞赛主程序
    训练模型：python main.py train
    生成预测：python main.py predict
    完整流程：python main.py run-all
"""

import click
import logging
import yaml
from pathlib import Path
import warnings
from datetime import datetime
import json
import os
import pandas as pd 

from src.trainer import ModelTrainer
from src.utils import setup_logging, timer
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    加载配置文件
    Args:
        config_path: 配置文件路径
    Returns:
        配置字典
    """
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

    # 设置日志
    setup_logging(config['logging'])
    logger.info("Starting training process...")

    try:
        # 初始化训练器
        trainer = ModelTrainer(config)

        # 运行训练
        metrics = trainer.run_training()

        # 保存训练结果
        output_dir = Path(config['data']['output_dir'])
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
    # 加载配置
    config = load_config(config)
    setup_logging(config['logging'])
    logger.info("Starting prediction process...")

    try:
        # 初始化训练器
        trainer = ModelTrainer(config)

        # 设置测试日期
        if test_date is None:
            test_date = config['training']['pred_date']

        # 生成预测
        trainer.generate_submission(test_date)
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

        # 首先打印数据结构信息
        logger.info("User data columns: %s", user_data.columns.tolist())
        logger.info("Item data columns: %s", item_data.columns.tolist())

        # 转换时间列
        user_data['time'] = pd.to_datetime(user_data['time'])

        # 基本统计信息
        analysis = {
            'user_stats': {
                'total_users': user_data['user_id'].nunique(),
                'total_interactions': len(user_data),
                'behavior_counts': user_data['behavior_type'].value_counts().to_dict(),
                'date_range': {
                    'start': str(user_data['time'].min()),
                    'end': str(user_data['time'].max()),
                    'total_days': (user_data['time'].max() - user_data['time'].min()).days + 1
                },
                'avg_actions_per_user': len(user_data) / user_data['user_id'].nunique()
            },
            'item_stats': {
                'total_items': item_data['item_id'].nunique() if 'item_id' in item_data.columns else 0,
                'total_categories': item_data['category'].nunique(),  # 修改为 'category'
                'items_per_category': item_data.groupby('category')['item_id'].nunique().to_dict()  # 修改为 'category'
            },
            'behavior_patterns': {
                'hourly_distribution': user_data.groupby(user_data['time'].dt.hour).size().to_dict(),
                # 将日期转换为字符串
                'daily_distribution': user_data.groupby(user_data['time'].dt.date).size().apply(lambda x: int(x)).apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)).to_dict(),
                'behavior_type_distribution': user_data['behavior_type'].value_counts().to_dict()
            }
        }

        # 保存分析结果
        output_dir = Path(config_dict['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 确保所有日期都被转换为字符串
        analysis['behavior_patterns']['daily_distribution'] = {
            k.strftime('%Y-%m-%d') if hasattr(k, 'strftime') else str(k): v 
            for k, v in analysis['behavior_patterns']['daily_distribution'].items()
        }

        with open(output_dir / 'data_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4, default=str)

        # 打印关键指标
        logger.info("\n=== 数据分析关键指标 ===")
        logger.info(f"总用户数: {analysis['user_stats']['total_users']:,}")
        logger.info(f"总交互数: {analysis['user_stats']['total_interactions']:,}")
        logger.info(f"数据时间跨度: {analysis['user_stats']['date_range']['total_days']} 天")
        logger.info(f"平均每用户行为数: {analysis['user_stats']['avg_actions_per_user']:.2f}")
        logger.info("行为类型分布:")
        for behavior_type, count in analysis['user_stats']['behavior_counts'].items():
            logger.info(f"  - 类型 {behavior_type}: {count:,}")

        logger.info("Data analysis completed")

    except Exception as e:
        logger.exception("Error occurred during data analysis")
        raise click.ClickException(str(e))

def calculate_conversion_rate(df, from_behavior, to_behavior):
    """计算行为转化率"""
    users_with_from = df[df['behavior_type'] == from_behavior]['user_id'].nunique()
    users_with_to = df[(df['behavior_type'] == to_behavior) & 
                      (df['user_id'].isin(df[df['behavior_type'] == from_behavior]['user_id']))]['user_id'].nunique()
    return users_with_to / users_with_from if users_with_from > 0 else 0

def calculate_category_conversion_rates(df):
    """计算各类别的转化率"""
    category_rates = {}
    for category in df['item_category'].unique():
        category_df = df[df['item_category'] == category]
        category_rates[str(category)] = {
            'view_to_purchase': calculate_conversion_rate(category_df, 1, 4),
            'cart_to_purchase': calculate_conversion_rate(category_df, 3, 4)
        }
    return category_rates


@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='配置文件路径')
@click.option('--debug', is_flag=True, help='是否开启调试模式')
@timer
def run_all(config: str, debug: bool):
    """运行完整流程：分析、训练和预测"""
    config_dict = load_config(config)
    if debug:
        config_dict['logging']['level'] = 'DEBUG'

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
        required_sections = ['data', 'features',
                             'model', 'training', 'logging']
        for section in required_sections:
            if section not in config:
                raise click.ClickException(
                    f"Missing required section: {section}")

        # 检查数据路径
        data_dir = Path(config['data']['raw_user_data']).parent
        if not data_dir.exists():
            raise click.ClickException(
                f"Data directory does not exist: {data_dir}")

        click.echo("Configuration is valid!")
        return True

    except Exception as e:
        logger.exception("Configuration validation failed")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    cli()
