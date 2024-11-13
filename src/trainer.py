import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Union

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model import RecommendationModel
from src.utils import setup_logging, timer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, config: Union[str, dict]):
        """
        初始化训练器
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

        # 初始化组件时传递配置字典
        self.data_processor = DataProcessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model = RecommendationModel(self.config)

        self.metrics_history = []
    
    @timer
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        准备数据
        Returns:
            处理后的用户数据、商品数据和编码器
        """
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

    def prepare_features(self, user_data: pd.DataFrame, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        准备特征数据
        Args:
            user_data: 用户行为数据
            end_date: 结束日期
        Returns:
            特征字典
        """
        try:
            logger.info(f"Generating features for {end_date}...")
            features = self.feature_engineer.generate_features(user_data, end_date)
            
            # 验证特征不为空
            if not features:
                raise ValueError("No features were generated")
            
            # 验证特征维度
            for feature_name, feature_df in features.items():
                if feature_df.empty:
                    logger.warning(f"Empty feature DataFrame for {feature_name}")
                else:
                    logger.info(f"Feature {feature_name} shape: {feature_df.shape}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in prepare_features: {str(e)}")
            raise

    def prepare_training_data(self, features: Dict[str, pd.DataFrame], labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        Args:
            features: 特征字典
            labels: 标签数据
        Returns:
            特征矩阵和标签向量
        """
        try:
            # 合并所有特征
            all_features = []
            for feature_df in features.values():
                if not feature_df.empty:
                    all_features.append(feature_df)
            
            if not all_features:
                raise ValueError("No valid features available for training")
            
            X = pd.concat(all_features, axis=1)
            
            # 确保有共同的索引
            common_index = X.index.intersection(labels.index)
            if common_index.empty:
                raise ValueError("No common samples between features and labels")
            
            X = X.loc[common_index]
            y = labels.loc[common_index]
            
            logger.info(f"Prepared training data with shape: {X.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in prepare_training_data: {str(e)}")
            raise

    @timer
    def train_and_evaluate(self, user_data: pd.DataFrame, train_end_date: str, valid_end_date: str) -> Dict[str, float]:
        """
        训练并评估模型
        Args:
            user_data: 用户行为数据
            train_end_date: 训练集截止日期
            valid_end_date: 验证集截止日期
        Returns:
            评估指标
        """
        try:
            # 准备训练数据
            logger.info("Preparing training features...")
            train_features = self.prepare_features(user_data, train_end_date)
            if not train_features:
                raise ValueError("Failed to generate training features")

            # 准备训练标签
            X_train, y_train = self.model.prepare_training_data(
                train_features, user_data, train_end_date)

            # 训练模型
            logger.info("Training model...")
            self.model.train(X_train, y_train)

            # 准备验证数据
            logger.info("Preparing validation features...")
            valid_features = self.prepare_features(user_data, valid_end_date)
            if not valid_features:
                raise ValueError("Failed to generate validation features")

            # 生成推荐
            logger.info("Generating recommendations...")
            recommendations = self.model.generate_recommendations(valid_features)

            # 评估
            if recommendations:
                metrics = self.model.evaluate(recommendations, user_data, valid_end_date)
            else:
                logger.warning("No recommendations generated")
                metrics = {
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'coverage': 0
                }

            return metrics

        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise

    def evaluate_recommendations(self,
                                 recommendations: List[Tuple[int, int, float]],
                                 user_data: pd.DataFrame,
                                 eval_date: str) -> Dict[str, float]:
        """
        评估推荐结果
        Args:
            recommendations: 推荐列表
            user_data: 用户行为数据
            eval_date: 评估日期
        Returns:
            评估指标
        """
        eval_date = pd.to_datetime(eval_date)
        actual_purchases = set(
            user_data[
                (user_data['time'].dt.date == eval_date.date()) &
                (user_data['behavior_type'] == 4)
            ][['user_id_encoded', 'item_id_encoded']].itertuples(index=False)
        )

        pred_purchases = set((user, item) for user, item, _ in recommendations)

        # 计算评估指标
        try:
            precision = len(pred_purchases & actual_purchases) / \
                len(pred_purchases)
        except ZeroDivisionError:
            precision = 0

        try:
            recall = len(pred_purchases & actual_purchases) / \
                len(actual_purchases)
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    @timer
    def run_training(self):
        """运行完整的训练流程"""
        # 准备数据
        user_data, item_data, encoders = self.prepare_data()

        # 设置时间范围
        train_end_date = self.config['training']['train_end_date']
        valid_end_date = self.config['training']['pred_date']

        # 训练并评估模型
        metrics = self.train_and_evaluate(
            user_data,
            train_end_date,
            valid_end_date
        )

        # 保存模型和特征重要性
        self.save_model_artifacts()

        return metrics

    def save_model_artifacts(self):
        """保存模型相关文件"""
        output_dir = Path(self.config['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save_model(output_dir / 'model.pkl')

        # 保存特征重要性
        pd.DataFrame({
            'feature': self.model.feature_importance.index,
            'importance': self.model.feature_importance.values
        }).to_csv(output_dir / 'feature_importance.csv', index=False)

        # 保存训练历史
        pd.DataFrame(self.metrics_history).to_csv(
            output_dir / 'training_history.csv',
            index=False
        )

        logger.info(f"Saved model artifacts to {output_dir}")

    def generate_submission(self, test_date: str):
        """
        生成预测提交文件
        Args:
            test_date: 预测日期
        """
        logger.info(f"Generating predictions for {test_date}")
    
        try:
            # 准备数据
            user_data, item_data, encoders = self.prepare_data()
        
            # 准备特征
            test_features = self.prepare_features(user_data, test_date)
            if not test_features:
                raise ValueError("Failed to generate test features")
            
            # 生成推荐
            recommendations = self.model.generate_recommendations(test_features)
            if not recommendations:
                raise ValueError("No recommendations generated")
            
            # 转换回原始ID
            submission = []
            for user_encoded, item_encoded, _ in recommendations:
                try:
                    user_id = encoders['user_id'].inverse_transform([user_encoded])[0]
                    item_id = encoders['item_id'].inverse_transform([item_encoded])[0]
                    # 转换为字符串类型
                    submission.append((str(user_id), str(item_id)))
                except Exception as e:
                    logger.warning(f"Error converting IDs: {e}")
                    continue
        
            # 创建DataFrame并去重
            submission_df = pd.DataFrame(
                submission,
                columns=['user_id', 'item_id']
            ).drop_duplicates()
        
            # 保存为CSV文件
            output_path = Path(self.config['data']['output_dir']) / 'tianchi_mobile_recommendation_predict.csv'
            submission_df.to_csv(output_path, index=False, encoding='utf-8')
        
            logger.info(f"Generated submission file with {len(submission_df)} predictions")
            logger.info(f"Submission file saved to: {output_path}")
        
            # 输出一些统计信息
            logger.info(f"Number of unique users: {submission_df['user_id'].nunique()}")
            logger.info(f"Number of unique items: {submission_df['item_id'].nunique()}")
        
        except Exception as e:
            logger.error(f"Error generating submission: {str(e)}")
            raise

if __name__ == "__main__":
    # 运行训练流程
    trainer = ModelTrainer('config/config.yaml')
    metrics = trainer.run_training()

    # 生成提交文件
    trainer.generate_submission('2024-11-14')
