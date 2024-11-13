import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F

class UserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.user_embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings, config['features']['user']['embedding_size'])
            for feat, num_embeddings in config['features']['user']['categorical_dims'].items()
        })
        
        num_numerical = len(config['features']['user']['numerical'])
        total_dims = (num_numerical + 
                     len(self.user_embeddings) * config['features']['user']['embedding_size'])
        
        self.fc = nn.Linear(total_dims, config['model']['architecture']['hidden_dims'][0])
        
    def forward(self, user_features):
        numerical = user_features['numerical']
        categorical = user_features['categorical']
        
        embeddings = []
        for feat, embedding_layer in self.user_embeddings.items():
            embeddings.append(embedding_layer(categorical[feat]))
            
        x = torch.cat([numerical] + embeddings, dim=1)
        return self.fc(x)

class ItemEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.item_embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings, config['features']['item']['embedding_size'])
            for feat, num_embeddings in config['features']['item']['categorical_dims'].items()
        })
        
        num_numerical = len(config['features']['item']['numerical'])
        total_dims = (num_numerical + 
                     len(self.item_embeddings) * config['features']['item']['embedding_size'])
        
        self.fc = nn.Linear(total_dims, config['model']['architecture']['hidden_dims'][0])
        
    def forward(self, item_features):
        numerical = item_features['numerical']
        categorical = item_features['categorical']
        
        embeddings = []
        for feat, embedding_layer in self.item_embeddings.items():
            embeddings.append(embedding_layer(categorical[feat]))
            
        x = torch.cat([numerical] + embeddings, dim=1)
        return self.fc(x)

class SequenceEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.item_encoder = ItemEncoder(config)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['model']['architecture']['hidden_dims'][0],
                nhead=8,
                dim_feedforward=config['model']['architecture']['hidden_dims'][0] * 4,
                dropout=config['model']['architecture']['dropout']
            ),
            num_layers=2
        )
        
    def forward(self, sequence_features, mask=None):
        # sequence_features shape: [batch_size, seq_len, feature_dim]
        item_embeddings = self.item_encoder(sequence_features)
        
        if mask is None:
            mask = torch.zeros(item_embeddings.size(0), item_embeddings.size(1)).bool()
            
        return self.transformer(item_embeddings.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)

class DeepRecommender(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Encoders
        self.user_encoder = UserEncoder(config)
        self.item_encoder = ItemEncoder(config)
        self.sequence_encoder = SequenceEncoder(config)
        
        # Fusion layers
        hidden_dims = config['model']['architecture']['hidden_dims']
        input_dim = hidden_dims[0] * 3  # user + item + sequence
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(config['model']['architecture']['dropout'])
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.fusion = nn.Sequential(*layers)
        
    def forward(self, batch):
        user_features = batch['user_features']
        item_features = batch['item_features']
        sequence_features = batch['sequence_features']
        sequence_mask = batch['sequence_mask']
        
        user_embedding = self.user_encoder(user_features)
        item_embedding = self.item_encoder(item_features)
        sequence_embedding = self.sequence_encoder(sequence_features, sequence_mask)
        sequence_embedding = sequence_embedding.mean(dim=1)  # pool sequence
        
        # Concatenate all features
        combined = torch.cat([
            user_embedding,
            item_embedding,
            sequence_embedding
        ], dim=1)
        
        return self.fusion(combined)
    
    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.binary_cross_entropy(y_hat, batch['labels'])
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.binary_cross_entropy(y_hat, batch['labels'])
        
        # Calculate metrics
        preds = (y_hat > 0.5).float()
        acc = (preds == batch['labels']).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
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
