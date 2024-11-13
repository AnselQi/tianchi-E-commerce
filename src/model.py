import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)

class RecommendationModel(pl.LightningModule):
    def __init__(self, config: Union[str, dict]):
        """初始化推荐模型"""
        super().__init__()
        
        # 加载配置
        if isinstance(config, (str, Path)):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config must be either a path (str) or a dictionary")
            
        self.save_hyperparameters()
        
        # 获取模型配置
        model_config = self.config['model']['architecture']
        self.embedding_dim = model_config['embedding_dim']
        self.hidden_dims = model_config['hidden_dims']
        self.dropout = model_config['dropout']
        
        # 初始化模型组件
        self._init_embeddings()
        self._init_feature_layers()
        self._init_sequence_encoder()
        self._init_prediction_layers()
        
        # 记录特征重要性
        self.feature_importance = None

    def _init_embeddings(self):
        """初始化嵌入层"""
        # 用户特征嵌入
        self.user_embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings, self.embedding_dim)
            for feat, num_embeddings in self.config['data']['features']['user']['categorical_dims'].items()
        })
        
        # 商品特征嵌入
        self.item_embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings, self.embedding_dim)
            for feat, num_embeddings in self.config['data']['features']['item']['categorical_dims'].items()
        })

    def _init_feature_layers(self):
        """初始化特征处理层"""
        # 用户特征处理
        self.user_mlp = self._create_mlp(
            input_dim=self._get_user_feature_dim(),
            hidden_dims=self.hidden_dims
        )
        
        # 商品特征处理
        self.item_mlp = self._create_mlp(
            input_dim=self._get_item_feature_dim(),
            hidden_dims=self.hidden_dims
        )

    def _init_sequence_encoder(self):
        """初始化序列编码器"""
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=self.embedding_dim * 4,
                dropout=self.dropout
            ),
            num_layers=2
        )

    def _init_prediction_layers(self):
        """初始化预测层"""
        total_dim = self.hidden_dims[-1] * 3  # user + item + sequence
        self.prediction_layers = self._create_mlp(
            input_dim=total_dim,
            hidden_dims=[total_dim // 2, total_dim // 4, 1],
            output_activation=nn.Sigmoid()
        )

    def _create_mlp(self, input_dim: int, hidden_dims: List[int], 
                   output_activation: nn.Module = None) -> nn.Sequential:
        """创建多层感知机"""
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(self.dropout)
            ])
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        if output_activation:
            layers.append(output_activation)
            
        return nn.Sequential(*layers)

    def _get_user_feature_dim(self) -> int:
        """获取用户特征维度"""
        categorical_dim = len(self.user_embeddings) * self.embedding_dim
        numerical_dim = len(self.config['data']['features']['user']['numerical'])
        return categorical_dim + numerical_dim

    def _get_item_feature_dim(self) -> int:
        """获取商品特征维度"""
        categorical_dim = len(self.item_embeddings) * self.embedding_dim
        numerical_dim = len(self.config['data']['features']['item']['numerical'])
        return categorical_dim + numerical_dim

    def forward(self, batch: Dict) -> torch.Tensor:
        """前向传播"""
        # 用户特征
        user_categorical = batch['user_features']['categorical']
        user_numerical = batch['user_features']['numerical']
        user_emb = [emb(user_categorical[feat]) for feat, emb in self.user_embeddings.items()]
        user_features = torch.cat([*user_emb, user_numerical], dim=1)
        user_repr = self.user_mlp(user_features)
        
        # 商品特征
        item_categorical = batch['item_features']['categorical']
        item_numerical = batch['item_features']['numerical']
        item_emb = [emb(item_categorical[feat]) for feat, emb in self.item_embeddings.items()]
        item_features = torch.cat([*item_emb, item_numerical], dim=1)
        item_repr = self.item_mlp(item_features)
        
        # 序列特征
        sequence = batch['sequence_features']
        mask = batch['sequence_mask']
        sequence_repr = self.sequence_encoder(sequence, src_key_padding_mask=mask)
        sequence_repr = sequence_repr.mean(dim=1)  # 平均池化
        
        # 特征融合
        combined = torch.cat([user_repr, item_repr, sequence_repr], dim=1)
        
        # 预测
        return self.prediction_layers(combined)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        y_hat = self(batch)
        loss = F.binary_cross_entropy(y_hat, batch['labels'])
        
        # 记录指标
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """验证步骤"""
        y_hat = self(batch)
        loss = F.binary_cross_entropy(y_hat, batch['labels'])
        
        # 计算指标
        preds = (y_hat > 0.5).float()
        acc = (preds == batch['labels']).float().mean()
        
        # 记录指标
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['model']['training']['learning_rate'],
            weight_decay=self.config['model']['training']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['model']['training']['scheduler']['T_max'],
            eta_min=self.config['model']['training']['scheduler']['eta_min']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def generate_recommendations(self, dataloader: DataLoader) -> List[Tuple]:
        """生成推荐"""
        self.eval()
        recommendations = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating recommendations"):
                scores = self(batch)
                user_ids = batch['user_features']['categorical']['user_id']
                item_ids = batch['item_features']['categorical']['item_id']
                
                # 获取top-k推荐
                top_k = self.config['training']['top_k']
                values, indices = torch.topk(scores, k=min(top_k, len(scores)))
                
                for user_id, item_id, score in zip(
                    user_ids[indices], item_ids[indices], values
                ):
                    recommendations.append((
                        user_id.item(),
                        item_id.item(),
                        score.item()
                    ))
        
        return recommendations

    def save_model(self, save_path: str):
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path)
        self.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Model loaded from {load_path}")
