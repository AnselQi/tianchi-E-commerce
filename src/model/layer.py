import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

class UserEncoder(nn.Module):
    """用户编码器"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        feature_dim = config['model']['architecture']['user_feature_dim']
        hidden_dims = config['model']['architecture']['user_hidden_dims']
        
        # 特征编码层
        self.embedding_layers = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings, config['data']['features']['user']['embedding_dim'])
            for feat, num_embeddings in config['data']['features']['user']['categorical_dims'].items()
        })
        
        # MLP层
        layers = []
        prev_dim = feature_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(config['model']['architecture']['dropout'])
            ])
            prev_dim = dim
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 处理类别特征
        embeddings = [
            self.embedding_layers[feat](features[f'{feat}_encoded'])
            for feat in self.config['data']['features']['user']['categorical_features']
        ]
        
        # 处理数值特征
        numerical = features['numerical']
        
        # 连接所有特征
        x = torch.cat([*embeddings, numerical], dim=1)
        
        # MLP处理
        return self.mlp(x)

class ItemEncoder(nn.Module):
    """商品编码器"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        feature_dim = config['model']['architecture']['item_feature_dim']
        hidden_dims = config['model']['architecture']['item_hidden_dims']
        
        # 特征编码层
        self.embedding_layers = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings, config['data']['features']['item']['embedding_dim'])
            for feat, num_embeddings in config['data']['features']['item']['categorical_dims'].items()
        })
        
        # MLP层
        layers = []
        prev_dim = feature_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(config['model']['architecture']['dropout'])
            ])
            prev_dim = dim
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 处理类别特征
        embeddings = [
            self.embedding_layers[feat](features[f'{feat}_encoded'])
            for feat in self.config['data']['features']['item']['categorical_features']
        ]
        
        # 处理数值特征
        numerical = features['numerical']
        
        # 连接所有特征
        x = torch.cat([*embeddings, numerical], dim=1)
        
        # MLP处理
        return self.mlp(x)

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """多头注意力层"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config['model']['architecture']['sequence_dim']
        nhead = config['model']['architecture']['num_attention_heads']
        
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=config['model']['architecture']['dropout']
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        return self.multihead_attn(query, key, value, key_padding_mask=mask)

class SequenceEncoder(nn.Module):
    """序列编码器"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config['model']['architecture']['sequence_dim']
        
        # 特征编码
        self.item_encoder = ItemEncoder(config)
        self.behavior_embedding = nn.Embedding(
            config['data']['features']['sequence']['num_behaviors'],
            config['data']['features']['sequence']['behavior_dim']
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['model']['architecture']['num_attention_heads'],
            dim_feedforward=config['model']['architecture']['transformer_ff_dim'],
            dropout=config['model']['architecture']['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['architecture']['num_transformer_layers']
        )
        
    def forward(self, sequence: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 获取序列数据
        items = sequence['items']
        behaviors = sequence['behaviors']
        mask = sequence['mask']
        
        # 编码商品和行为
