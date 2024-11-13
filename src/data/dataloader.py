import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import numpy as np

class RecommendationDataLoader:
    """推荐系统数据加载器"""
    
    def __init__(self, 
                 dataset: 'RecommendationDataset',
                 batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        初始化数据加载器
        Args:
            dataset: 数据集
            batch_size: 批大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            pin_memory: 是否将数据固定在内存中
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 创建DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )
        
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批处理整理函数
        Args:
            batch: 批数据列表
        Returns:
            整理后的批数据
        """
        # 收集所有样本的字段
        batch_dict = {
            'user_id': [],
            'user_features': [],
            'sequence': {
                'items': [],
                'behaviors': [],
                'times': [],
                'mask': [],
                'length': []
            }
        }
        
        if 'labels' in batch[0]:
            batch_dict['labels'] = []
            
        # 整理数据
        for sample in batch:
            batch_dict['user_id'].append(sample['user_id'])
            batch_dict['user_features'].append(sample['user_features'])
            
            # 序列数据
            seq = sample['sequence']
            batch_dict['sequence']['items'].append(seq['items'])
            batch_dict['sequence']['behaviors'].append(seq['behaviors'])
            batch_dict['sequence']['times'].append(seq['times'])
            batch_dict['sequence']['mask'].append(seq['mask'])
            batch_dict['sequence']['length'].append(seq['length'])
            
            if 'labels' in sample:
                batch_dict['labels'].append(sample['labels'])
                
        # 转换为张量
        batch_dict['user_id'] = torch.stack(batch_dict['user_id'])
        batch_dict['user_features'] = torch.stack(batch_dict['user_features'])
        
        batch_dict['sequence'] = {
            'items': torch.stack(batch_dict['sequence']['items']),
            'behaviors': torch.stack(batch_dict['sequence']['behaviors']),
            'times': torch.stack(batch_dict['sequence']['times']),
            'mask': torch.stack(batch_dict['sequence']['mask']),
            'length': torch.stack(batch_dict['sequence']['length'])
        }
        
        if 'labels' in batch_dict:
            batch_dict['labels'] = torch.stack(batch_dict['labels'])
            
        return batch_dict
    
    def __iter__(self):
        """返回数据加载器迭代器"""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """返回批次数量"""
        return len(self.dataloader)
