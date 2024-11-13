import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml
import logging
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from typing import Union
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)


class RecommendationModel:
    def __init__(self, config: Union[str, dict]):
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be either a path (str) or a dictionary")

    def prepare_training_data(self, features: Dict[str, pd.DataFrame],
                              user_data: pd.DataFrame, pred_date: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        Args:
            features: 特征字典
            user_data: 用户行为数据
            pred_date: 预测日期
        Returns:
            特征矩阵和标签
        """
        pred_date = pd.to_datetime(pred_date)
        label_data = user_data[
            (user_data['time'].dt.date == pred_date.date()) &
            (user_data['behavior_type'] == 4)
        ]

        logger.info("Preparing training data...")
        train_data = []
        labels = []

        # 创建正样本
        positive_pairs = set(
            zip(label_data['user_id_encoded'], label_data['item_id_encoded']))

        # 采样负样本
        users = features['user_1d'].index
        items = features['item_1d'].index

        for user_id in users:
            # 添加正样本
            pos_items = [item for user,
                         item in positive_pairs if user == user_id]
            for item_id in pos_items:
                sample = self._combine_features(user_id, item_id, features)
                train_data.append(sample)
                labels.append(1)

            # 采样负样本（采样比例1:3）
            neg_items = np.random.choice(
                [i for i in items if (user_id, i) not in positive_pairs],
                size=min(len(pos_items) * 3, len(items)),
                replace=False
            )
            for item_id in neg_items:
                sample = self._combine_features(user_id, item_id, features)
                train_data.append(sample)
                labels.append(0)

        return pd.DataFrame(train_data), pd.Series(labels)

    def _combine_features(self, user_id: int, item_id: int, features: Dict[str, pd.DataFrame]) -> List[float]:
        """
        组合所有特征
        Args:
            user_id: 用户ID
            item_id: 商品ID
            features: 特征字典
        Returns:
            组合后的特征向量
        """
        combined_features = []

        for window in self.config['features']['time_windows']:
            # 用户特征
            user_feats = features[f'user_{window}d']
            if user_id in user_feats.index:
                combined_features.extend(user_feats.loc[user_id].values)
            else:
                combined_features.extend([0] * len(user_feats.columns))

            # 商品特征
            item_feats = features[f'item_{window}d']
            if item_id in item_feats.index:
                combined_features.extend(item_feats.loc[item_id].values)
            else:
                combined_features.extend([0] * len(item_feats.columns))

            # 用户-商品交叉特征
            ui_feats = features[f'user_item_{window}d']
            if (user_id, item_id) in ui_feats.index:
                combined_features.extend(
                    ui_feats.loc[(user_id, item_id)].values)
            else:
                combined_features.extend([0] * len(ui_feats.columns))

        return combined_features

    def train(self, train_features: pd.DataFrame, train_labels: pd.Series):
        """
        训练模型
        Args:
            train_features: 训练特征
            train_labels: 训练标签
        """
        logger.info("Training model...")

        train_data = lgb.Dataset(train_features, label=train_labels)

        # 训练模型
        self.model = lgb.train(
            self.config['model']['params'],
            train_data,
            num_boost_round=self.config['training']['num_epochs']
        )

        # 记录特征重要性
        self.feature_importance = pd.Series(
            self.model.feature_importance(),
            index=train_features.columns
        ).sort_values(ascending=False)

        logger.info("Model training completed")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        预测
        Args:
            features: 特征矩阵
        Returns:
            预测概率
        """
        return self.model.predict(features)

    def generate_recommendations(self, features: Dict[str, pd.DataFrame]) -> List[Tuple]:
        """
        生成推荐
        Args:
            features: 特征字典
        Returns:
            推荐列表 [(user_id, item_id, score), ...]
        """
        logger.info("Starting recommendation generation...")

        # 使用时间窗口最大的特征
        max_window = max(self.config['features']['time_windows'])
    
        # 检查特征是否存在
        user_features = features.get(f'user_{max_window}d')
        item_features = features.get(f'item_{max_window}d')
    
        # 检查特征是否有效
        if user_features is None or item_features is None:
            logger.warning("Missing required features")
            return []
        
        if user_features.empty or item_features.empty:
            logger.warning("Empty feature DataFrames")
            return []
        
        # 获取所有用户和商品
        users = user_features.index
        items = item_features.index
    
        logger.info(f"Generating recommendations for {len(users)} users and {len(items)} items")
    
        all_predictions = []
        # 使用 tqdm 显示进度
        for user in tqdm(users, desc="Generating recommendations"):
            try:
                user_vec = user_features.loc[user]
            
                # 计算用户-商品分数
                scores = []
                for item in items:
                    try:
                        item_vec = item_features.loc[item]
                        # 如果向量中包含任何 NaN 值，跳过这个组合
                        if np.isnan(user_vec).any() or np.isnan(item_vec).any():
                            continue
                        score = self._calculate_score(user_vec, item_vec)
                        scores.append((user, item, score))
                    except Exception as e:
                        logger.debug(f"Error calculating score for user {user}, item {item}: {str(e)}")
                        continue
            
                # 取每个用户的top-k推荐
                if scores:  # 只在有分数时排序
                    top_k = self.config['training']['top_k']
                    user_top_k = sorted(scores, key=lambda x: x[2], reverse=True)[:top_k]
                    all_predictions.extend(user_top_k)
                
            except Exception as e:
                logger.warning(f"Error processing user {user}: {str(e)}")
                continue
    
        logger.info(f"Generated {len(all_predictions)} recommendations")
        return all_predictions

    def _calculate_score(self, user_vec: pd.Series, item_vec: pd.Series) -> float:
        """
        计算用户-商品分数
        Args:
            user_vec: 用户特征向量
            item_vec: 商品特征向量
        Returns:
            相似度分数
        """
        try:
            # 确保向量不包含 NaN 值
            if np.isnan(user_vec).any() or np.isnan(item_vec).any():
                return 0.0
            
            # 使用点积计算相似度
            numerator = np.dot(user_vec, item_vec)
            user_norm = np.linalg.norm(user_vec)
            item_norm = np.linalg.norm(item_vec)
        
            # 避免除零错误
            if user_norm == 0 or item_norm == 0:
                return 0.0
            
            return numerator / (user_norm * item_norm)
        except Exception as e:
            logger.debug(f"Error in score calculation: {str(e)}")
            return 0.0

    def evaluate(self, recommendations: List[Tuple[int, int, float]], 
            user_data: pd.DataFrame, 
            eval_date: str) -> Dict[str, float]:
        """
        评估推荐结果
        Args:
            recommendations: 推荐列表，格式为[(user_id, item_id, score),...]
            user_data: 用户行为数据
            eval_date: 评估日期
        Returns:
            评估指标字典
        """
        logger.info(f"Evaluating recommendations for date: {eval_date}")
    
        # 获取实际购买数据
        eval_date = pd.to_datetime(eval_date)
        actual_purchases = set(
            user_data[
                (user_data['time'].dt.date == eval_date.date()) & 
                (user_data['behavior_type'] == 4)
            ][['user_id_encoded', 'item_id_encoded']].itertuples(index=False)
            )
    
        # 获取推荐结果
        pred_purchases = set((user, item) for user, item, _ in recommendations)
    
        # 计算评估指标
        metrics = self._calculate_metrics(pred_purchases, actual_purchases, user_data)
    
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def _calculate_metrics(self, pred_purchases: set, actual_purchases: set, user_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算评估指标
        """
        # 基础指标
        try:
            precision = len(pred_purchases & actual_purchases) / len(pred_purchases)
        except ZeroDivisionError:
            precision = 0
        
        try:
            recall = len(pred_purchases & actual_purchases) / len(actual_purchases)
        except ZeroDivisionError:
            recall = 0
        
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        
        # 用户级别指标
        user_metrics = self._calculate_user_metrics(pred_purchases, actual_purchases)
    
        # 计算覆盖率
        total_possible_pairs = len(user_data['user_id_encoded'].unique()) * \
                          len(user_data['item_id_encoded'].unique())
        coverage = len(pred_purchases) / total_possible_pairs if total_possible_pairs > 0 else 0
    
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_user_precision': user_metrics['avg_precision'],
            'avg_user_recall': user_metrics['avg_recall'],
            'coverage': coverage,
            'num_recommendations': len(pred_purchases),
            'num_actual_purchases': len(actual_purchases)
        }

    def _calculate_user_metrics(self, pred_purchases: set, actual_purchases: set) -> Dict[str, float]:
        """
        计算用户级别的指标
        """
        user_metrics = []
        users = set(uid for uid, _ in pred_purchases | actual_purchases)
    
        for user in users:
            user_preds = set(item for uid, item in pred_purchases if uid == user)
            user_actuals = set(item for uid, item in actual_purchases if uid == user)
        
            if user_preds and user_actuals:
                user_precision = len(user_preds & user_actuals) / len(user_preds)
                user_recall = len(user_preds & user_actuals) / len(user_actuals)
                user_metrics.append({
                    'precision': user_precision,
                    'recall': user_recall
                })
    
        if user_metrics:
            avg_precision = np.mean([m['precision'] for m in user_metrics])
            avg_recall = np.mean([m['recall'] for m in user_metrics])
        else:
            avg_precision = avg_recall = 0
    
        return {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall
        }
    
    
    
    def save_model(self, model_path: str):
        """
        保存模型
        Args:
            model_path: 模型保存路径
        """
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_model(str(model_path))
        
        # 保存特征重要性
        if self.feature_importance is not None:
            importance_path = model_dir / 'feature_importance.csv'
            self.feature_importance.to_csv(importance_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        加载模型
        Args:
            model_path: 模型加载路径
        """
        self.model = lgb.Booster(model_file=str(model_path))
        
        # 尝试加载特征重要性
        importance_path = Path(model_path).parent / 'feature_importance.csv'
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path, index_col=0).squeeze()
        
        logger.info(f"Model loaded from {model_path}")