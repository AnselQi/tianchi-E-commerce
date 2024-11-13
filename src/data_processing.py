import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
from typing import Tuple, Dict, Union, List
import logging
from src.utils import setup_logging
import torch
from torch.utils.data import random_split

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理类"""

    def __init__(self, config: Union[str, dict]):
        """
        初始化数据处理器
        Args:
            config: 配置文件路径或配置字典
        """
        # 处理配置输入
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be either a path (str) or a dictionary")

        setup_logging(self.config['logging'])

        # 编码器和标准化器
        self.encoders = {
            'user_id': LabelEncoder(),
            'item_id': LabelEncoder(),
            'category': LabelEncoder(),
            'user_geohash': LabelEncoder(),
            'item_geohash': LabelEncoder()
        }
        
        self.scalers = {
            'numerical_features': StandardScaler()
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载原始数据"""
        logger.info("Loading raw data...")

        user_data = pd.read_csv(self.config['data']['paths']['raw_user_data'])
        item_data = pd.read_csv(self.config['data']['paths']['raw_item_data'])

        if 'item_category' in user_data.columns:
            user_data = user_data.rename(columns={'item_category': 'category'})
        if 'item_category' in item_data.columns:
            item_data = item_data.rename(columns={'item_category': 'category'})

        logger.info(f"Loaded {len(user_data)} user records and {len(item_data)} item records")
        return user_data, item_data

    def preprocess_data(self, user_data: pd.DataFrame, item_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """数据预处理"""
        logger.info("Preprocessing data...")

        try:
            # 时间处理
            user_data['time'] = pd.to_datetime(user_data['time'])
            user_data['hour'] = user_data['time'].dt.hour
            user_data['day'] = user_data['time'].dt.day
            user_data['weekday'] = user_data['time'].dt.weekday

            # 编码分类特征
            categorical_features = self.config['data']['features']['categorical']
            for feature in categorical_features:
                if feature in user_data.columns:
                    user_data[f'{feature}_encoded'] = self.encoders[feature].fit_transform(user_data[feature])
                if feature in item_data.columns:
                    if hasattr(self.encoders[feature], 'classes_'):
                        item_data[f'{feature}_encoded'] = self.encoders[feature].transform(item_data[feature])
                    else:
                        item_data[f'{feature}_encoded'] = self.encoders[feature].fit_transform(item_data[feature])

            # 处理缺失值
            user_data['user_geohash'] = user_data['user_geohash'].fillna('unknown')
            item_data['item_geohash'] = item_data['item_geohash'].fillna('unknown')

            # 生成序列特征
            user_data = self.generate_sequence_features(user_data)

            # 标准化数值特征
            numerical_features = ['hour', 'day', 'weekday']
            user_data[numerical_features] = self.scalers['numerical_features'].fit_transform(user_data[numerical_features])

            logger.info("Data preprocessing completed")
            return user_data, item_data

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def generate_sequence_features(self, user_data: pd.DataFrame) -> pd.DataFrame:
        """生成序列特征"""
        max_seq_length = self.config['data']['features']['sequence']['max_length']
        
        # 按用户和时间排序
        user_data = user_data.sort_values(['user_id', 'time'])
        
        # 生成行为序列
        sequences = user_data.groupby('user_id').agg({
            'item_id_encoded': lambda x: list(x)[-max_seq_length:],
            'behavior_type': lambda x: list(x)[-max_seq_length:],
            'time': lambda x: list(x)[-max_seq_length:]
        }).reset_index()
        
        # 填充序列
        sequences['sequence_length'] = sequences['item_id_encoded'].apply(len)
        sequences['item_sequence'] = sequences['item_id_encoded'].apply(
            lambda x: x + [0] * (max_seq_length - len(x)))
        sequences['behavior_sequence'] = sequences['behavior_type'].apply(
            lambda x: x + [0] * (max_seq_length - len(x)))
        
        return pd.merge(user_data, sequences[['user_id', 'item_sequence', 'behavior_sequence', 'sequence_length']], 
                       on='user_id', how='left')

    def prepare_train_val_data(self) -> Tuple[Dict, Dict]:
        """准备训练和验证数据"""
        user_data, item_data = self.load_data()
        processed_user_data, processed_item_data = self.preprocess_data(user_data, item_data)
        
        # 分割训练和验证数据
        train_end_date = pd.to_datetime(self.config['training']['train_end_date'])
        val_date = pd.to_datetime(self.config['training']['pred_date'])
        
        train_data = processed_user_data[processed_user_data['time'] <= train_end_date]
        val_data = processed_user_data[processed_user_data['time'].dt.date == val_date.date()]
        
        # 创建特征字典
        train_features = self.create_feature_dict(train_data, processed_item_data)
        val_features = self.create_feature_dict(val_data, processed_item_data)
        
        return train_features, val_features

    def create_feature_dict(self, user_data: pd.DataFrame, item_data: pd.DataFrame) -> Dict:
        """创建特征字典"""
        return {
            'user_features': {
                'categorical': {
                    feat: torch.tensor(user_data[f'{feat}_encoded'].values)
                    for feat in self.config['data']['features']['user']['categorical']
                },
                'numerical': torch.tensor(
                    user_data[self.config['data']['features']['user']['numerical']].values
                )
            },
            'item_features': {
                'categorical': {
                    feat: torch.tensor(item_data[f'{feat}_encoded'].values)
                    for feat in self.config['data']['features']['item']['categorical']
                },
                'numerical': torch.tensor(
                    item_data[self.config['data']['features']['item']['numerical']].values
                )
            },
            'sequence_features': {
                'items': torch.tensor(user_data['item_sequence'].tolist()),
                'behaviors': torch.tensor(user_data['behavior_sequence'].tolist()),
                'lengths': torch.tensor(user_data['sequence_length'].values)
            },
            'labels': torch.tensor(
                (user_data['behavior_type'] == 4).astype(float).values
            )
        }

    def prepare_test_data(self, test_date: str = None) -> Dict:
        """准备测试数据"""
        if test_date is None:
            test_date = self.config['training']['pred_date']
            
        user_data, item_data = self.load_data()
        processed_user_data, processed_item_data = self.preprocess_data(user_data, item_data)
        
        test_data = processed_user_data[
            processed_user_data['time'].dt.date == pd.to_datetime(test_date).date()
        ]
        
        return self.create_feature_dict(test_data, processed_item_data)

    def calculate_sequence_stats(self, user_data: pd.DataFrame) -> Dict:
        """计算序列统计信息"""
        sequence_lengths = user_data.groupby('user_id').size()
        return {
            'avg_length': sequence_lengths.mean(),
            'max_length': sequence_lengths.max(),
            'min_length': sequence_lengths.min()
        }

    def create_submission(self, predictions: np.ndarray, test_data: Dict) -> pd.DataFrame:
        """创建提交文件"""
        user_ids = self.encoders['user_id'].inverse_transform(test_data['user_features']['categorical']['user_id'])
        item_ids = self.encoders['item_id'].inverse_transform(test_data['item_features']['categorical']['item_id'])
        
        submission = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'score': predictions
        })
        
        # 筛选top-k推荐
        top_k = self.config['training']['top_k']
        submission = submission.groupby('user_id').apply(
            lambda x: x.nlargest(top_k, 'score')
        ).reset_index(drop=True)
        
        return submission[['user_id', 'item_id']]

if __name__ == "__main__":
    processor = DataProcessor('config/config.yaml')
    train_features, val_features = processor.prepare_train_val_data()
    logger.info("Data preparation completed successfully")
