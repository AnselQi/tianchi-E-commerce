import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import yaml
from typing import Tuple, Dict, Union
import logging
from src.utils import setup_logging

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

        # 更新编码器字典中的键名，使其与特征工程中的名称一致
        self.encoders = {
            'user_id': LabelEncoder(),
            'item_id': LabelEncoder(),
            'category': LabelEncoder()  # 改为 'category' 以匹配特征工程中的命名
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载原始数据
        Returns:
            用户数据和商品数据
        """
        logger.info("Loading raw data...")

        user_data = pd.read_csv(self.config['data']['raw_user_data'])
        item_data = pd.read_csv(self.config['data']['raw_item_data'])

        # 确保category列名一致
        if 'item_category' in user_data.columns:
            user_data = user_data.rename(columns={'item_category': 'category'})
        if 'item_category' in item_data.columns:
            item_data = item_data.rename(columns={'item_category': 'category'})

        logger.info(
            f"Loaded {len(user_data)} user records and {len(item_data)} item records")
        return user_data, item_data

    def preprocess_data(self, user_data: pd.DataFrame, item_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        数据预处理
        Args:
            user_data: 用户数据
            item_data: 商品数据
        Returns:
            处理后的用户数据和商品数据
        """
        logger.info("Preprocessing data...")

        try:
            # 时间处理
            user_data['time'] = pd.to_datetime(user_data['time'])

            # 确保category列存在
            if 'category' not in user_data.columns:
                raise ValueError("'category' column not found in user_data")

            # 编码分类特征
            encoded_columns = {
                'user_id': 'user_id_encoded',
                'item_id': 'item_id_encoded',
                'category': 'category_encoded'
            }

            for orig_col, enc_col in encoded_columns.items():
                if orig_col in user_data.columns:
                    user_data[enc_col] = self.encoders[orig_col].fit_transform(user_data[orig_col])
                if orig_col in item_data.columns:
                    if hasattr(self.encoders[orig_col], 'classes_'):
                        item_data[enc_col] = self.encoders[orig_col].transform(item_data[orig_col])
                    else:
                        item_data[enc_col] = self.encoders[orig_col].fit_transform(item_data[orig_col])

            # 处理缺失值
            user_data['user_geohash'] = user_data['user_geohash'].fillna('unknown')
            item_data['item_geohash'] = item_data['item_geohash'].fillna('unknown')

            # 验证必要的列是否存在
            required_columns = ['category_encoded', 'user_id_encoded', 'item_id_encoded']
            missing_columns = [col for col in required_columns if col not in user_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns after preprocessing: {missing_columns}")

            logger.info("Data preprocessing completed")
            logger.info(f"Available columns in user_data: {user_data.columns.tolist()}")
            return user_data, item_data

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        完整的数据处理流程
        Returns:
            处理后的用户数据、商品数据和编码器
        """
        logger.info("Starting complete data processing...")
        user_data, item_data = self.load_data()
        processed_user_data, processed_item_data = self.preprocess_data(user_data, item_data)
        self.save_processed_data(processed_user_data, processed_item_data)
        return processed_user_data, processed_item_data, self.encoders

    def save_processed_data(self, user_data: pd.DataFrame, item_data: pd.DataFrame):
        """
        保存处理后的数据
        Args:
            user_data: 处理后的用户数据
            item_data: 处理后的商品数据
        """
        try:
            processed_dir = Path(self.config['data']['processed_data_dir'])
            processed_dir.mkdir(parents=True, exist_ok=True)

            user_data.to_pickle(processed_dir / 'processed_user_data.pkl')
            item_data.to_pickle(processed_dir / 'processed_item_data.pkl')

            # 保存编码器
            np.save(processed_dir / 'encoders.npy', self.encoders)

            logger.info(f"Saved processed data to {processed_dir}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        加载处理后的数据，如果数据无效则重新处理
        Returns:
            处理后的用户数据、商品数据和编码器
        """
        try:
            processed_dir = Path(self.config['data']['processed_data_dir'])
            
            # 检查所有必要的文件是否存在
            required_files = ['processed_user_data.pkl', 'processed_item_data.pkl', 'encoders.npy']
            if not all((processed_dir / file).exists() for file in required_files):
                logger.info("Some processed files missing, starting reprocessing...")
                return self.process_all()

            # 尝试加载数据
            user_data = pd.read_pickle(processed_dir / 'processed_user_data.pkl')
            item_data = pd.read_pickle(processed_dir / 'processed_item_data.pkl')
            encoders = np.load(processed_dir / 'encoders.npy', allow_pickle=True).item()

            # 验证必要的列是否存在
            required_columns = ['category_encoded', 'user_id_encoded', 'item_id_encoded']
            if not all(col in user_data.columns for col in required_columns):
                logger.info("Processed data missing required columns, starting reprocessing...")
                return self.process_all()

            logger.info("Successfully loaded processed data")
            return user_data, item_data, encoders

        except Exception as e:
            logger.warning(f"Error loading processed data: {str(e)}. Starting reprocessing...")
            return self.process_all()


if __name__ == "__main__":
    # 测试数据处理
    processor = DataProcessor('config/config.yaml')
    
    # 直接调用load_processed_data，它会在需要时自动重新处理数据
    user_data, item_data, encoders = processor.load_processed_data()