import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Optional

class RecommendationDataset(Dataset):
    """推荐系统数据集类"""
    
    def __init__(self, 
                 features: Dict[str, torch.Tensor],
                 user_data: pd.DataFrame,
                 end_date: str,
                 config: dict,
                 labels: Optional[torch.Tensor] = None):
        """
        初始化数据集
        Args:
            features: 特征字典
            user_data: 用户行为数据
            end_date: 结束日期
            config: 配置字典
            labels: 标签数据
        """
        self.features = features
        self.config = config
        self.labels = labels
        
        # 处理序列特征
        self.sequence_features = self._process_sequences(user_data, end_date)
        
        # 获取用户和商品列表
        self.users = torch.tensor(list(features['user_features'].keys()))
        self.items = torch.tensor(list(features['item_features'].keys()))
        
    def _process_sequences(self, user_data: pd.DataFrame, end_date: str) -> Dict[str, torch.Tensor]:
        """处理行为序列"""
        max_len = self.config['data']['features']['sequence']['max_length']
        
        # 按用户分组并获取序列
        sequences = {}
        for user_id, group in user_data.groupby('user_id_encoded'):
            # 获取时间截止前的行为
            group = group[group['time'] <= pd.to_datetime(end_date)]
            group = group.sort_values('time')
            
            # 提取最近的行为序列
            recent_actions = group.iloc[-max_len:]
            
            # 创建序列特征
            seq_len = len(recent_actions)
            item_seq = np.zeros(max_len, dtype=np.int64)
            behavior_seq = np.zeros(max_len, dtype=np.int64)
            time_seq = np.zeros(max_len, dtype=np.float32)
            
            # 填充序列
            item_seq[:seq_len] = recent_actions['item_id_encoded'].values
            behavior_seq[:seq_len] = recent_actions['behavior_type'].values
            time_seq[:seq_len] = recent_actions['time'].astype(np.int64).values
            
            # 创建掩码
            mask = np.ones(max_len, dtype=np.bool_)
            mask[:seq_len] = False
            
            sequences[user_id] = {
                'items': torch.tensor(item_seq),
                'behaviors': torch.tensor(behavior_seq),
                'times': torch.tensor(time_seq),
                'mask': torch.tensor(mask),
                'length': torch.tensor(seq_len)
            }
            
        return sequences

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        user_id = self.users[idx]
        
        # 获取用户特征
        user_features = self.features['user_features'][user_id]
        
        # 获取序列特征
        sequence = self.sequence_features.get(user_id.item(), {
            'items': torch.zeros(self.config['data']['features']['sequence']['max_length'], dtype=torch.long),
            'behaviors': torch.zeros(self.config['data']['features']['sequence']['max_length'], dtype=torch.long),
            'times': torch.zeros(self.config['data']['features']['sequence']['max_length']),
            'mask': torch.ones(self.config['data']['features']['sequence']['max_length'], dtype=torch.bool),
            'length': torch.tensor(0)
        })
        
        # 组装样本
        sample = {
            'user_id': user_id,
            'user_features': user_features,
            'sequence': sequence
        }
        
        if self.labels is not None:
            sample['labels'] = self.labels[idx]
            
        return sample
