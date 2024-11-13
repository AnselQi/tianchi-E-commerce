import gc
import os
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Union

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

        # 提取配置参数
        self.feature_config = self.config['features']
        self.time_windows = self.feature_config['time_windows']
        self.chunk_size = self.feature_config.get('chunk_size', 1000000)
        self.memory_optimize = self.feature_config.get('memory_optimize', True)
        
        # 系统配置
        self.system_config = self.config.get('system', {})
        self.max_memory = self.system_config.get('memory_optimize', {}).get('max_memory_gb', 16) * 1024 * 1024 * 1024
        
        # 设置日志
        self.enable_memory_logging = self.config['logging'].get('enable_memory_logging', True)
        self.memory_log_interval = self.config['logging'].get('memory_log_interval', 1000)
        
        # 解禁内存
        self.memory_limit = self.config.get('system', {}).get('memory_limit_gb', 8) * 1024 * 1024 * 1024
        self.chunk_size = self.config.get('features', {}).get('chunk_size', 500000)


    def generate_features(self, df: pd.DataFrame, end_date: str) -> Dict[str, pd.DataFrame]:
        """生成所有特征（内存优化版本）"""
        features = {}
        end_date = pd.to_datetime(end_date)
        
        # 只保留必要的列
        needed_columns = ['user_id_encoded', 'item_id_encoded', 'category_encoded', 
                         'behavior_type', 'time']
        df = df[needed_columns].copy()
        
        # 强制转换数据类型以节省内存
        df['user_id_encoded'] = df['user_id_encoded'].astype(np.int32)
        df['item_id_encoded'] = df['item_id_encoded'].astype(np.int32)
        df['category_encoded'] = df['category_encoded'].astype(np.int16)
        df['behavior_type'] = df['behavior_type'].astype(np.int8)
        
        logger.info(f"Data time range: {df['time'].min()} to {df['time'].max()}")
        
        for window in tqdm(self.time_windows, desc="Generating features"):
            try:
                start_date = end_date - pd.Timedelta(days=window)
                window_data = df[
                    (df['time'] >= start_date) & 
                    (df['time'] < end_date)
                ].copy()

                logger.info(f"Generating features for {window}-day window")
                logger.info(f"Window date range: {start_date} to {end_date}")
                logger.info(f"Window data shape: {window_data.shape}")

                if len(window_data) == 0:
                    logger.warning(f"No data in {window}-day window")
                    empty_features = self.generate_empty_features(df, window)
                    features[f'user_{window}d'] = empty_features['user_features']
                    features[f'item_{window}d'] = empty_features['item_features']
                    features[f'user_item_{window}d'] = empty_features['user_item_features']
                    continue

                # 分批处理特征生成
                user_features = self._generate_user_features_chunked(window_data, window)
                del window_data  # 立即释放内存
                gc.collect()
                
                window_data = df[
                    (df['time'] >= start_date) & 
                    (df['time'] < end_date)
                ].copy()
                item_features = self._generate_item_features_chunked(window_data, window)
                del window_data
                gc.collect()
                
                window_data = df[
                    (df['time'] >= start_date) & 
                    (df['time'] < end_date)
                ].copy()
                ui_features = self._generate_user_item_features_chunked(window_data, window)
                del window_data
                gc.collect()

                features[f'user_{window}d'] = user_features
                features[f'item_{window}d'] = item_features
                features[f'user_item_{window}d'] = ui_features

                # 检查内存使用
                self._check_memory_usage()

            except Exception as e:
                logger.error(f"Error generating features for {window}-day window: {str(e)}")
                raise

        return features

    def _check_memory_usage(self):
        """检查内存使用情况并在需要时清理"""
        process = psutil.Process(os.getpid())
        memory_used = process.memory_info().rss
        logger.info(f"Current memory usage: {memory_used / 1024 / 1024:.2f} MB")
        
        if memory_used > self.memory_limit:
            logger.warning(f"Memory usage ({memory_used / 1024 / 1024:.2f} MB) exceeds limit "
                         f"({self.memory_limit / 1024 / 1024:.2f} MB). Triggering cleanup...")
            gc.collect()
            memory_used = process.memory_info().rss
            logger.info(f"Memory usage after cleanup: {memory_used / 1024 / 1024:.2f} MB")
    
    def _generate_user_features_chunked(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """分块生成用户特征"""
        user_features_list = []
        unique_users = df['user_id_encoded'].unique()
    
        for chunk_start in range(0, len(unique_users), self.chunk_size):
            chunk_users = unique_users[chunk_start:chunk_start + self.chunk_size]
            chunk = df[df['user_id_encoded'].isin(chunk_users)]
        
            # 基础统计
            user_stats = chunk.groupby('user_id_encoded').agg({
                'behavior_type': ['count', 'nunique'],
                'item_id_encoded': 'nunique',
                'category_encoded': 'nunique',
                'time': lambda x: len(pd.to_datetime(x).dt.date.unique())
            })
        
            # 行为类型统计
            behavior_counts = pd.get_dummies(chunk['behavior_type'], prefix='behavior_type')\
                .groupby(chunk['user_id_encoded']).sum()
        
            # 合并特征
            chunk_features = pd.concat([user_stats, behavior_counts], axis=1)
            user_features_list.append(chunk_features)
        
            del chunk, user_stats, behavior_counts
            self._check_memory_usage()
    
        user_features = pd.concat(user_features_list, axis=0)
        user_features.columns = [f'user_{col}_{window}d' for col in user_features.columns]
        return user_features
    

    def _generate_item_features_chunked(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """分块生成商品特征"""
        item_features_list = []
        
        for chunk_start in range(0, len(df), self.chunk_size):
            chunk = df.iloc[chunk_start:chunk_start + self.chunk_size]
            
            # 基础商品特征
            item_stats = chunk.groupby('item_id_encoded').agg({
                'user_id_encoded': ['count', 'nunique'],
                'behavior_type': ['nunique', 'mean'],
                'time': lambda x: len(pd.to_datetime(x).dt.date.unique())
            })
            
            # 行为类型统计
            behavior_counts = pd.get_dummies(chunk['behavior_type'], prefix='behavior_type').groupby(chunk['item_id_encoded']).sum()
            
            # 合并特征
            chunk_features = pd.concat([item_stats, behavior_counts], axis=1)
            item_features_list.append(chunk_features)
            
            self._check_memory_usage()

        # 合并所有分块结果
        item_features = pd.concat(item_features_list)
        item_features = item_features.groupby(level=0).sum()
        
        # 重命名列
        item_features.columns = [f'item_{col}_{window}d' for col in item_features.columns]
        
        return item_features

    def _generate_user_item_features_chunked(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """分块生成用户-商品交叉特征"""
        ui_features_list = []
        
        for chunk_start in range(0, len(df), self.chunk_size):
            chunk = df.iloc[chunk_start:chunk_start + self.chunk_size]
            
            # 用户-商品交互统计
            ui_stats = chunk.groupby(['user_id_encoded', 'item_id_encoded']).agg({
                'behavior_type': ['count', 'nunique', 'mean'],
                'time': lambda x: (x.max() - x.min()).total_seconds() / 3600  # 转换为小时
            })
            
            ui_features_list.append(ui_stats)
            
            self._check_memory_usage()

        # 合并所有分块结果
        ui_features = pd.concat(ui_features_list)
        ui_features = ui_features.groupby(level=[0,1]).last()  # 使用last而不是sum
        
        # 重命名列
        ui_features.columns = [f'ui_{col}_{window}d' for col in ui_features.columns]
        
        return ui_features


    def generate_empty_features(self, df: pd.DataFrame, time_window: int) -> Dict[str, pd.DataFrame]:
        """
        生成空特征DataFrame（当没有数据时使用）
        Args:
            df: 原始数据（用于获取用户和商品ID）
            time_window: 时间窗口
        Returns:
            空特征字典
        """
        # 获取所有唯一的用户和商品ID
        user_ids = df['user_id_encoded'].unique()
        item_ids = df['item_id_encoded'].unique()
    
        # 用户特征
        user_columns = [
            'behavior_count', 'behavior_types', 'unique_items', 'unique_categories', 
            'active_days'
        ] + [f'behavior_type_{i}' for i in range(1, 5)]
        user_features = pd.DataFrame(0, index=user_ids, 
                                    columns=[f'user_{col}_{time_window}d' for col in user_columns])
    
        # 商品特征
        item_columns = [
            'user_count', 'unique_users', 'behavior_types', 'mean_behavior',
            'active_days'
        ] + [f'behavior_type_{i}' for i in range(1, 5)]
        item_features = pd.DataFrame(0, index=item_ids, 
                                    columns=[f'item_{col}_{time_window}d' for col in item_columns])
    
        # 用户-商品交叉特征
        ui_columns = ['interaction_count', 'behavior_types', 'mean_behavior', 'time_span']
        ui_features = pd.DataFrame(columns=[f'ui_{col}_{time_window}d' for col in ui_columns])
    
        return {
            'user_features': user_features,
            'item_features': item_features,
            'user_item_features': ui_features
        }
    
    def save_features(self, features: Dict[str, pd.DataFrame], output_dir: str):
        """
        保存特征
        Args:
            features: 特征字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, feature_df in features.items():
            feature_df.to_pickle(output_path / f'{name}_features.pkl')

        logger.info(f"Saved features to {output_path}")

    def load_features(self, input_dir: str) -> Dict[str, pd.DataFrame]:
        """
        加载特征
        Args:
            input_dir: 输入目录
        Returns:
            特征字典
        """
        input_path = Path(input_dir)
        features = {}

        for feature_file in input_path.glob('*_features.pkl'):
            name = feature_file.stem.replace('_features', '')
            features[name] = pd.read_pickle(feature_file)

        return features


if __name__ == "__main__":
    # 测试特征工程
    from data_processing import DataProcessor

    processor = DataProcessor('config/config.yaml')
    user_data, item_data, _ = processor.load_processed_data()

    engineer = FeatureEngineer('config/config.yaml')
    features = engineer.generate_features(user_data, '2014-12-18')
    engineer.save_features(features, 'data/processed/features')