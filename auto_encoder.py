import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import RobustScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutoEncoderProjector(TransformerMixin, BaseEstimator):
    def __init__(self, seq_length: int = 50, embed_dim: int = 4, lr: int = 0.001):
        self.embed_dim = embed_dim
        self.pre_processor = AutoEncoderProcessor(seq_length)
        self.model = AutoEncoderModel(seq_length, self.embed_dim)
        self.model.apply(self.model.weights_init)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.criterion = nn.MSELoss()

    def fit(self, X: pd.Series, n_epoch: int = 3, n_batchs: int = 128):
        self.pre_processor.fit(X)
        dataset = AdyenDataset(X, self.pre_processor)
        data_loader = DataLoader(dataset, batch_size=n_batchs, shuffle=True, num_workers=0)

        for epoch in tqdm(range(n_epoch), desc='Train AutoEncoder'):
            self.model.train()
            for batch_idx, feats in enumerate(data_loader):
                inputs = feats.to(device).type(torch.float32)
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, inputs)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        embedding = [0] * len(X)
        dataset = AdyenDataset(X, self.pre_processor)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            self.model.eval()
            for idx, feats in enumerate(data_loader):
                inputs = feats.to(device).type(torch.float32)
                _, embed = self.model(inputs)
                embedding[idx] = embed.squeeze().detach().numpy()

        embedding = pd.DataFrame(
            embedding,
            columns=[f'{X.name}_emb{i}' for i in range(self.embed_dim)],
            index=X.index
        )

        return embedding


class AutoEncoderModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(AutoEncoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ELU(),
            nn.Linear(input_dim // 2, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim // 2),
            nn.ELU(),
            nn.Linear(input_dim // 2, input_dim),
        )

    def forward(self, x):
        emb = self.encoder(x)
        x = self.decoder(emb)

        return x, emb.detach()

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


class AutoEncoderProcessor(TransformerMixin, BaseEstimator):
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.scaler = RobustScaler()

    def fit(self, X: pd.Series):
        X = self._pad_list(X, self.max_length)
        self.scaler.fit(X.tolist())

        return self

    def transform(self, X: pd.Series) -> pd.Series:
        series_name = X.name
        X = self._pad_list(X, self.max_length)
        X = self.scaler.transform(X.tolist())

        return pd.Series(list(X), name=series_name)

    @staticmethod
    def _pad_list(X: pd.Series, max_length: int) -> pd.Series:
        """Fill np.nan to make every sequence have `self.max_length` size"""
        X = X.apply(lambda x: x[-max_length:])

        return X.apply(lambda x: [0] * (max_length - len(x)) + x)


class AdyenDataset(Dataset):
    def __init__(self, X: pd.Series, processor: AutoEncoderProcessor):
        self.X = processor.transform(X)

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)
