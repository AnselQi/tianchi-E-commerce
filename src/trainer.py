import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models.deep_recommender import DeepRecommender
from src.data.dataset import RecommendationDataset
from src.utils import setup_logging, timer

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: Union[str, dict]):
        """初始化训练器"""
        # 处理配置
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be either a path (str) or a dictionary")

        setup_logging(self.config['logging'])

        # 初始化组件
        self.data_processor = DataProcessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model = DeepRecommender(self.config)

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        pl.seed_everything(self.config['system']['seed'])

    @timer
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """准备数据"""
        try:
            logger.info("Loading processed data...")
            user_data, item_data, encoders = self.data_processor.load_processed_data()
        except FileNotFoundError:
            logger.info("Processed data not found, processing raw data...")
            user_data, item_data = self.data_processor.load_data()
            user_data, item_data = self.data_processor.preprocess_data(
                user_data, item_data)
            self.data_processor.save_processed_data(user_data, item_data)
            _, _, encoders = self.data_processor.load_processed_data()

        return user_data, item_data, encoders

    def create_datasets(self, user_data: pd.DataFrame, train_end_date: str, valid_end_date: str) -> Tuple[RecommendationDataset, RecommendationDataset]:
        """创建训练和验证数据集"""
        # 生成特征
        train_features = self.feature_engineer.generate_features(user_data, train_end_date)
        valid_features = self.feature_engineer.generate_features(user_data, valid_end_date)

        # 创建数据集
        train_dataset = RecommendationDataset(train_features, user_data, train_end_date, self.config)
        valid_dataset = RecommendationDataset(valid_features, user_data, valid_end_date, self.config)

        return train_dataset, valid_dataset

    def create_dataloaders(self, train_dataset: RecommendationDataset, valid_dataset: RecommendationDataset) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['model']['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config['model']['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )

        return train_loader, valid_loader

    def setup_training(self) -> Tuple[pl.Trainer, List]:
        """设置训练器和回调"""
        # 设置检查点回调
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.config['data']['paths']['checkpoint_dir'],
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        )

        # 设置早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['model']['training']['early_stopping']['patience'],
            min_delta=self.config['model']['training']['early_stopping']['min_delta'],
            mode='min'
        )

        # 设置TensorBoard日志
        logger = TensorBoardLogger(
            save_dir=self.config['data']['paths']['log_dir'],
            name='lightning_logs'
        )

        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=self.config['model']['training']['num_epochs'],
            accelerator=self.config['device']['accelerator'],
            devices=self.config['device']['devices'],
            strategy=self.config['device']['strategy'],
            precision=self.config['device']['precision'],
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            log_every_n_steps=50
        )

        return trainer, [checkpoint_callback, early_stopping]

    @timer
    def train(self, train_dataset: RecommendationDataset, valid_dataset: RecommendationDataset) -> Dict[str, float]:
        """训练模型"""
        # 创建数据加载器
        train_loader, valid_loader = self.create_dataloaders(train_dataset, valid_dataset)

        # 设置训练器
        trainer, callbacks = self.setup_training()

        # 训练模型
        trainer.fit(
            self.model,
            train_loader,
            valid_loader
        )

        # 获取最佳模型
        best_model_path = callbacks[0].best_model_path
        self.model = DeepRecommender.load_from_checkpoint(best_model_path)

        # 返回最佳指标
        return {
            'best_val_loss': callbacks[0].best_model_score.item(),
            'best_epoch': callbacks[0].best_epoch
        }

    def predict(self, test_dataset: RecommendationDataset) -> List[Tuple]:
        """生成预测"""
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['model']['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )

        return self.model.generate_recommendations(test_loader)

    @timer
    def run_training(self) -> Dict[str, float]:
        """运行完整训练流程"""
        # 准备数据
        user_data, item_data, encoders = self.prepare_data()

        # 创建数据集
        train_dataset, valid_dataset = self.create_datasets(
            user_data,
            self.config['training']['train_end_date'],
            self.config['training']['pred_date']
        )

        # 训练模型
        metrics = self.train(train_dataset, valid_dataset)

        # 保存模型
        self.save_model_artifacts()

        return metrics

    def save_model_artifacts(self):
        """保存模型相关文件"""
        output_dir = Path(self.config['data']['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save_model(output_dir / 'model.pt')

        # 保存配置
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        logger.info(f"Saved model artifacts to {output_dir}")

    def generate_submission(self, test_date: str):
        """生成提交文件"""
        # 准备数据
        user_data, item_data, encoders = self.prepare_data()

        # 创建测试数据集
        test_features = self.feature_engineer.generate_features(user_data, test_date)
        test_dataset = RecommendationDataset(test_features, user_data, test_date, self.config)

        # 生成预测
        recommendations = self.predict(test_dataset)

        # 转换ID并保存
        submission = []
        for user_encoded, item_encoded, _ in recommendations:
            user_id = encoders['user_id'].inverse_transform([user_encoded])[0]
            item_id = encoders['item_id'].inverse_transform([item_encoded])[0]
            submission.append((str(user_id), str(item_id)))

        # 保存提交文件
        submission_df = pd.DataFrame(submission, columns=['user_id', 'item_id'])
        output_path = Path(self.config['data']['paths']['output_dir']) / 'submission.csv'
        submission_df.to_csv(output_path, index=False)

        logger.info(f"Generated submission file with {len(submission_df)} recommendations")

if __name__ == "__main__":
    # 运行训练流程
    trainer = ModelTrainer('config/config.yaml')
    metrics = trainer.run_training()
    trainer.generate_submission('2024-11-14')
