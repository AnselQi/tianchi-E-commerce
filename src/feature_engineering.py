import gc
import os
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config: Union[str, dict]):
        # 加载配置
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be either a path (str) or a dictionary")
            
        # 特征配置
        self.feature_config = self.config['data']['features']
        self.sequence_config = self.feature_config['sequence']
        self.max_seq_length = self.sequence_config['max_length']
        
        # 内存配置
        self.chunk_size = self.feature_config.get('chunk_size', 100000)
        self.memory_optimize = self.feature_config.get('memory_optimize', True)
        
        # 初始化标准化器
        self.scalers = {
            'user_numerical': StandardScaler(),
            'item_numerical': StandardScaler(),
            'sequence_numerical': StandardScaler()
        }

    def generate_features(self, df: pd.DataFrame, end_date: str) -> Dict[str, torch.Tensor]:
        """生成所有特征"""
        logger.info("Starting feature generation...")
        end_date = pd.to_datetime(end_date)
        
        try:
            # 1. 生成用户特征
            user_features = self._generate_user_features(df)
            self._check_memory_usage("After user features")

            # 2. 生成商品特征
            item_features = self._generate_item_features(df)
            self._check_memory_usage("After item features")

            # 3. 生成序列特征
            sequence_features = self._generate_sequence_features(df)
            self._check_memory_usage("After sequence features")

            # 4. 生成时序特征
            temporal_features = self._generate_temporal_features(df)
            self._check_memory_usage("After temporal features")

            # 合并所有特征
            features = {
                'user': user_features,
                'item': item_features,
                'sequence': sequence_features,
                'temporal': temporal_features
            }

            return self._convert_to_tensors(features)

        except Exception as e:
            logger.error(f"Error in feature generation: {str(e)}")
            raise

    def _generate_user_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """生成用户特征"""
        features = {}
        
        # 分批处理用户
        unique_users = df['user_id_encoded'].unique()
        user_features_list = []
        
        for chunk_start in range(0, len(unique_users), self.chunk_size):
            chunk_users = unique_users[chunk_start:chunk_start + self.chunk_size]
            chunk = df[df['user_id_encoded'].isin(chunk_users)]
            
            # 基础统计特征
            user_stats = chunk.groupby('user_id_encoded').agg({
                'behavior_type': ['count', 'nunique'],
                'item_id_encoded': 'nunique',
                'category_encoded': 'nunique'
            })
            
            # 用户行为分布
            behavior_features = pd.get_dummies(chunk['behavior_type'])\
                .groupby(chunk['user_id_encoded']).mean()
            
            # 类别偏好
            category_features = pd.get_dummies(chunk['category_encoded'])\
                .groupby(chunk['user_id_encoded']).mean()
            
            # 时间特征
            time_features = chunk.groupby('user_id_encoded').agg({
                'time': [
                    lambda x: x.dt.hour.mean(),
                    lambda x: x.dt.weekday.mean(),
                    lambda x: len(x.dt.date.unique())
                ]
            })
            
            # 合并特征
            chunk_features = pd.concat([
                user_stats, behavior_features, category_features, time_features
            ], axis=1)
            
            user_features_list.append(chunk_features)
            self._check_memory_usage()
        
        # 合并所有批次
        user_features = pd.concat(user_features_list)
        
        # 标准化数值特征
        numerical_cols = user_features.select_dtypes(include=['float64', 'int64']).columns
        user_features[numerical_cols] = self.scalers['user_numerical'].fit_transform(
            user_features[numerical_cols]
        )
        
        return user_features.to_dict('series')

    def _generate_item_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """生成商品特征"""
        features = {}
        
        # 分批处理商品
        unique_items = df['item_id_encoded'].unique()
        item_features_list = []
        
        for chunk_start in range(0, len(unique_items), self.chunk_size):
            chunk_items = unique_items[chunk_start:chunk_start + self.chunk_size]
            chunk = df[df['item_id_encoded'].isin(chunk_items)]
            
            # 商品统计特征
            item_stats = chunk.groupby('item_id_encoded').agg({
                'user_id_encoded': ['count', 'nunique'],
                'behavior_type': ['nunique', 'mean'],
                'time': lambda x: len(pd.to_datetime(x).dt.date.unique())
            })
            
            # 商品行为分布
            behavior_features = pd.get_dummies(chunk['behavior_type'])\
                .groupby(chunk['item_id_encoded']).mean()
            
            # 用户多样性
            user_diversity = chunk.groupby('item_id_encoded')\
                .agg({'user_id_encoded': lambda x: len(x.unique()) / len(x)})
            
            # 合并特征
            chunk_features = pd.concat([
                item_stats, behavior_features, user_diversity
            ], axis=1)
            
            item_features_list.append(chunk_features)
            self._check_memory_usage()
        
        # 合并所有批次
        item_features = pd.concat(item_features_list)
        
        # 标准化数值特征
        numerical_cols = item_features.select_dtypes(include=['float64', 'int64']).columns
        item_features[numerical_cols] = self.scalers['item_numerical'].fit_transform(
            item_features[numerical_cols]
        )
        
        return item_features.to_dict('series')

    def _generate_sequence_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """生成序列特征"""
        # 按用户和时间排序
        df = df.sort_values(['user_id_encoded', 'time'])
        
        # 初始化序列特征
        sequences = {
            'item_seq': [],
            'behavior_seq': [],
            'time_seq': [],
            'category_seq': [],
            'mask': []
        }
        
        # 分批处理用户
        unique_users = df['user_id_encoded'].unique()
        
        for chunk_start in range(0, len(unique_users), self.chunk_size):
            chunk_users = unique_users[chunk_start:chunk_start + self.chunk_size]
            chunk = df[df['user_id_encoded'].isin(chunk_users)]
            
            for user_id, user_data in chunk.groupby('user_id_encoded'):
                # 获取最近的行为序列
                recent_actions = user_data.iloc[-self.max_seq_length:]
                seq_length = len(recent_actions)
                
                # 填充序列
                item_seq = np.zeros(self.max_seq_length, dtype=np.int32)
                behavior_seq = np.zeros(self.max_seq_length, dtype=np.int32)
                category_seq = np.zeros(self.max_seq_length, dtype=np.int32)
                time_seq = np.zeros(self.max_seq_length, dtype=np.float32)
                mask = np.ones(self.max_seq_length, dtype=np.bool_)
                
                # 填充实际数据
                item_seq[:seq_length] = recent_actions['item_id_encoded'].values
                behavior_seq[:seq_length] = recent_actions['behavior_type'].values
                category_seq[:seq_length] = recent_actions['category_encoded'].values
                time_seq[:seq_length] = recent_actions['time'].astype(np.int64) // 10**9
                mask[:seq_length] = False
                
                # 添加到序列集合
                sequences['item_seq'].append(item_seq)
                sequences['behavior_seq'].append(behavior_seq)
                sequences['category_seq'].append(category_seq)
                sequences['time_seq'].append(time_seq)
                sequences['mask'].append(mask)
            
            self._check_memory_usage()
        
        return {k: np.array(v) for k, v in sequences.items()}

    def _generate_temporal_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """生成时间特征"""
        # 提取时间特征
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['weekday'] = df['time'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # 计算时间差特征
        df['time_diff'] = df.groupby('user_id_encoded')['time'].diff().dt.total_seconds()
        
        temporal_features = df.groupby('user_id_encoded').agg({
            'hour': ['mean', 'std'],
            'weekday': ['mean', 'std'],
            'is_weekend': 'mean',
            'time_diff': ['mean', 'std', 'max', 'min']
        }).fillna(0)
        
        # 标准化
        temporal_features = pd.DataFrame(
            self.scalers['sequence_numerical'].fit_transform(temporal_features),
            index=temporal_features.index,
            columns=temporal_features.columns
        )
        
        return temporal_features.to_dict('series')

    def _convert_to_tensors(self, features: Dict) -> Dict[str, torch.Tensor]:
        """将特征转换为PyTorch张量"""
        tensor_features = {}
        
        for feature_type, feature_dict in features.items():
            if feature_type == 'sequence':
                # 序列特征需要特殊处理
                tensor_features[feature_type] = {
                    k: torch.tensor(v, dtype=torch.long if k != 'time_seq' else torch.float)
                    for k, v in feature_dict.items()
                }
            else:
                # 其他特征直接转换为张量
                tensor_features[feature_type] = {
                    k: torch.tensor(v.values, dtype=torch.float)
                    for k, v in feature_dict.items()
                }
                
        return tensor_features

    def _check_memory_usage(self, stage: str = ""):
        """检查内存使用情况"""
        if self.memory_optimize:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            current_memory = memory_info.rss / 1024 / 1024  # MB
            
            logger.info(f"Memory usage {stage}: {current_memory:.2f} MB")
            
            if current_memory > self.config.get('system', {}).get('memory_optimize', {}).get('max_memory_gb', 16) * 1024:
                logger.warning(f"High memory usage detected. Triggering garbage collection...")
                gc.collect()

if __name__ == "__main__":
    # 测试特征工程
    from data_processing import DataProcessor
    
    processor = DataProcessor('config/config.yaml')
    user_data, item_data, _ = processor.load_processed_data()
    
    engineer = FeatureEngineer('config/config.yaml')
    features = engineer.generate_features(user_data, '2014-12-18')
    
    # 打印特征信息
    for feature_type, feature_dict in features.items():
        if isinstance(feature_dict, dict):
            for name, tensor in feature_dict.items():
                print(f"{feature_type} - {name}: {tensor.shape}")
        else:
            print(f"{feature_type}: {feature_dict.shape}")
