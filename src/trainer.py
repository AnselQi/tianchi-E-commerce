import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, List, Tuple, Union
import numpy as np
from pathlib import Path
import yaml
import gc
import psutil
import os
from tqdm import tqdm
import functools
from torch.utils.data import DataLoader

from src.data.dataset import RecommendationDataset
from src.data_processing import DataProcessor
from src.models.deep_recommender import DeepRecommender
from src.utils import setup_logging, timer
logger = logging.getLogger(__name__)

# 首先定义装饰器
def memory_optimize(func):
    """内存优化装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return wrapper

class ModelTrainer:
    def __init__(self, config: Union[str, dict]):
        """初始化训练器"""
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be either a path (str) or a dictionary")

        setup_logging(self.config['logging'])
        self.check_gpu_and_optimize()
        self.data_processor = DataProcessor(self.config)
        self.model = DeepRecommender(self.config)

    def monitor_memory_usage(self):
        """监控内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        logger.info(f"Memory Usage: {memory_info.rss / 1024 / 1024 / 1024:.2f}GB")
        logger.info(f"Virtual Memory: {memory_info.vms / 1024 / 1024 / 1024:.2f}GB")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")

    def check_gpu_and_optimize(self):
        """检查 GPU 并进行优化设置"""
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available!")
        
        gpu_name = torch.cuda.get_device_name(0)
        if 'A100' not in gpu_name:
            logger.warning(f"Expected NVIDIA A100, but found {gpu_name}")
        
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1024**3
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        logger.info(f"Using GPU: {gpu_name} with {total_memory:.2f}GB memory")
        return total_memory

    def setup_training(self) -> Tuple[pl.Trainer, List]:
        """设置训练器"""
        gpu_memory = self.check_gpu_and_optimize()
        
        if hasattr(self, 'dynamic_batch_size'):
            batch_size = self.calculate_optimal_batch_size(gpu_memory)
            self.config['model']['training']['batch_size'] = batch_size
            logger.info(f"Dynamically set batch size to: {batch_size}")
        
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[0],
            precision=16,
            strategy='auto',
            callbacks=[
                ModelCheckpoint(
                    dirpath=self.config['data']['paths']['checkpoint_dir'],
                    filename='model-{epoch:02d}-{val_loss:.2f}',
                    save_top_k=3,
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['model']['training']['early_stopping']['patience'],
                    mode='min'
                )
            ],
            logger=TensorBoardLogger(
                save_dir=self.config['data']['paths']['log_dir'],
                name='lightning_logs'
            ),
            gradient_clip_val=1.0,
            log_every_n_steps=50,
            max_epochs=self.config['model']['training']['num_epochs']
        )
        
        return trainer

    def create_dataloaders(self, train_dataset: RecommendationDataset, valid_dataset: RecommendationDataset) -> Tuple[DataLoader, DataLoader]:
        """创建优化的数据加载器"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['model']['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['system']['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.config['system']['prefetch_factor'],
            drop_last=True,
            generator=torch.Generator(device='cuda')
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config['model']['training']['batch_size'],
            shuffle=False,
            num_workers=max(1, self.config['system']['num_workers'] // 2),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        return train_loader, valid_loader

    @memory_optimize  # 现在可以使用这个装饰器了
    def train(self):
        """训练模型"""
        try:
            self.monitor_memory_usage()
            
            logger.info("Preparing training data...")
            train_data, val_data = self.data_processor.prepare_train_val_data()
            
            self.monitor_memory_usage()
            
            train_loader, val_loader = self.create_dataloaders(train_data, val_data)
            trainer = self.setup_training()
            
            logger.info("Starting model training...")
            trainer.fit(self.model, train_loader, val_loader)
            
            self.monitor_memory_usage()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def save_model(self):
        """保存模型和配置"""
        output_dir = Path(self.config['data']['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / 'model.pt'
        torch.save(self.model.state_dict(), model_path)
        
        config_path = output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        logger.info(f"Model and config saved to {output_dir}")

if __name__ == "__main__":
    trainer = ModelTrainer('config/config.yaml')
    trainer.train()
