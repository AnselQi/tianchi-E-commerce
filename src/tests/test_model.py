"""
推荐模型模块的单元测试
"""

import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
import tempfile
import yaml
from pathlib import Path
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

from src.model import RecommendationModel


def test_init_model(test_config):
    """测试模型初始化"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        assert model.config == test_config
        assert model.model is None
        assert model.feature_importance is None


def test_prepare_training_data(test_config, sample_features, sample_user_data):
    """测试训练数据准备"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 准备训练数据
        X, y = model.prepare_training_data(
            features=sample_features,
            user_data=sample_user_data,
            pred_date='2014-12-02'
        )

        # 验证特征矩阵
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert y.isin([0, 1]).all()  # 验证标签是二值的

        # 验证采样比例
        pos_samples = y.sum()
        neg_samples = len(y) - pos_samples
        assert neg_samples <= pos_samples * 3  # 验证负采样比例不超过1:3


def test_combine_features(test_config, sample_features):
    """测试特征组合"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 选择测试用户和商品
        user_id = sample_features['user_1d'].index[0]
        item_id = sample_features['item_1d'].index[0]

        # 组合特征
        combined_features = model._combine_features(
            user_id, item_id, sample_features)

        # 验证特征向量
        assert isinstance(combined_features, list)
        assert len(combined_features) > 0
        assert all(isinstance(x, (int, float)) for x in combined_features)


def test_train_model(test_config):
    """测试模型训练"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 创建简单的训练数据
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))

        # 训练模型
        model.train(X_train, y_train)

        # 验证模型
        assert isinstance(model.model, lgb.Booster)
        assert model.feature_importance is not None
        assert len(model.feature_importance) == X_train.shape[1]


def test_predict(test_config):
    """测试模型预测"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 训练一个简单的模型
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        model.train(X_train, y_train)

        # 进行预测
        X_test = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10)
        })
        predictions = model.predict(X_test)

        # 验证预测结果
        assert len(predictions) == len(X_test)
        assert all(0 <= p <= 1 for p in predictions)  # 验证概率值


def test_generate_recommendations(test_config, sample_features):
    """测试推荐生成"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 训练一个简单的模型
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        model.train(X_train, y_train)

        # 生成推荐
        recommendations = model.generate_recommendations(
            features=sample_features,
            top_k=5
        )

        # 验证推荐结果
        assert len(recommendations) > 0
        # (user_id, item_id, score)
        assert all(len(rec) == 3 for rec in recommendations)
        assert all(0 <= rec[2] <= 1 for rec in recommendations)  # 验证分数范围


def test_save_and_load_model(test_config, temp_data_dir):
    """测试模型保存和加载"""
    model_path = Path(temp_data_dir) / 'model.pkl'

    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 训练模型
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        model.train(X_train, y_train)

        # 保存模型
        model.save_model(model_path)

        # 创建新的模型实例并加载
        new_model = RecommendationModel(f.name)
        new_model.load_model(model_path)

        # 验证预测结果一致性
        X_test = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10)
        })
        pred1 = model.predict(X_test)
        pred2 = new_model.predict(X_test)
        np.testing.assert_array_almost_equal(pred1, pred2)


def test_feature_importance(test_config):
    """测试特征重要性"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 训练模型
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        model.train(X_train, y_train)

        # 验证特征重要性
        assert isinstance(model.feature_importance, pd.Series)
        assert list(model.feature_importance.index) == ['feature1', 'feature2']
        assert all(
            importance >= 0 for importance in model.feature_importance.values)


def test_error_handling(test_config):
    """测试错误处理"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 测试未训练就预测
        X_test = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10)
        })
        with pytest.raises(ValueError, match="Model has not been trained"):
            model.predict(X_test)

        # 测试特征不匹配
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        model.train(X_train, y_train)

        X_test_invalid = pd.DataFrame({
            'invalid_feature': np.random.rand(10)
        })
        with pytest.raises(ValueError, match="Feature names do not match"):
            model.predict(X_test_invalid)


def test_model_evaluation(test_config, sample_features, sample_user_data):
    """测试模型评估"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        model = RecommendationModel(f.name)

        # 准备训练数据
        X, y = model.prepare_training_data(
            features=sample_features,
            user_data=sample_user_data,
            pred_date='2014-12-02'
        )

        # 训练模型
        model.train(X, y)

        # 生成推荐
        recommendations = model.generate_recommendations(
            features=sample_features,
            top_k=5
        )

        # 评估结果
        y_true = y
        y_pred = [1 if (u, i) in [(r[0], r[1]) for r in recommendations] else 0
                  for u, i in zip(X.index.get_level_values(0), X.index.get_level_values(1))]

        metrics = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

        assert all(0 <= score <= 1 for score in metrics.values())


if __name__ == '__main__':
    pytest.main([__file__])
