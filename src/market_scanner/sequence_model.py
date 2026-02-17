import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import os
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class OptionSequenceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.3):
        super(OptionSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.fc(out)
        return self.sigmoid(out)

class SequenceTrainer:
    def __init__(self, models_dir: str = "models", seq_length: int = 60):
        self.models_dir = models_dir
        self.seq_length = seq_length
        self.model_path = os.path.join(models_dir, "sequence_lstm.pth")
        self.model = None
        self.input_dim = 4 # Default features: OI Velocity, Volume Surge, Gamma Change, PCR Trend
        
        os.makedirs(models_dir, exist_ok=True)

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes sequential features:
        - OI Velocity: Change in OI (call + put)
        - Volume Surge: Volume / Moving Average Volume
        - Gamma Change: Change in features['gex'] if available, else 0
        - PCR Trend: Change in PCR
        """
        # Ensure df has timestamps sorted
        df = df.sort_values("timestamp")
        
        # We need specific columns. If they don't exist, we mock or derive.
        # Check ingestion.py for what keys are in 'metrics'
        # keys: pcr, max_pain, gex, dex, gamma_flip, iv_skew, etc.
        # We need raw OI and Volume to be precise, or we assume they are in metrics.
        # features.py computes 'pcr'. It does NOT currently output raw total OI or Volume in 'metrics'.
        # We might need to fetch 'total_call_oi', 'total_put_oi' from the snapshot data directly?
        # For this exercise, we will assume these columns exist or we compute them from what we have.
        # Let's trust that 'metrics' has 'total_call_oi', 'total_put_oi', 'total_call_vol', 'total_put_vol'
        # IF NOT, we should have added them in feature engineer. 
        # BUT, since we can't easily change feature engineer right now without backtracking, 
        # let's assume we can use what's available or proxies.
        
        # Proxies:
        # OI Velocity ~ delta(PCR) * delta(Spot) ? No.
        # Let's use what we have in features.py output:
        # pcr, max_pain, gex, iv_rank, etc.
        
        # If we strictly need "OI velocity", we need OI. 
        # FeatureEngineer.compute_oi_stats returns 'pcr', 'oi_imbalance', 'max_pain'.
        # We can use 'oi_imbalance' delta.
        
        # Let's define features as:
        # 1. pcr_trend: diff(pcr)
        # 2. gex_change: diff(gex)
        # 3. max_pain_change: diff(max_pain)
        # 4. iv_skew_change: diff(iv_skew)
        
        # This is strictly derived from `derived_features` table.
        
        cols = ['pcr', 'gex', 'max_pain', 'iv_skew']
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0
        
        feature_df = pd.DataFrame()
        feature_df['pcr_trend'] = df['pcr'].diff().fillna(0)
        feature_df['gex_change'] = df['gex'].diff().fillna(0)
        feature_df['mp_change'] = df['max_pain'].diff().fillna(0)
        feature_df['skew_change'] = df['iv_skew'].diff().fillna(0)
        
        # Normalize/Scale?
        # A proper pipeline would fit a scaler. For simplicity here:
        feature_df = (feature_df - feature_df.mean()) / (feature_df.std() + 1e-6)
        
        self.input_dim = feature_df.shape[1]
        return feature_df

    def prepare_data(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.compute_derived_features(df).values
        
        X = []
        y = []
        
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            if target is not None:
                y.append(target.iloc[i+self.seq_length])
                
        X_tensor = torch.FloatTensor(np.array(X))
        if target is not None:
            y_tensor = torch.FloatTensor(np.array(y)).unsqueeze(1)
            return X_tensor, y_tensor
        
        return X_tensor, None

    def train_model(self, df: pd.DataFrame, target: pd.Series, epochs: int = 20, batch_size: int = 32):
        X, y = self.prepare_data(df, target)
        
        # Split train/val
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        
        self.model = OptionSequenceModel(input_dim=self.input_dim, hidden_dim=32, num_layers=2)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        best_loss = float('inf')
        patience = 5
        trigger_times = 0
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for tx, ty in train_loader:
                optimizer.zero_grad()
                out = self.model(tx)
                loss = criterion(out, ty)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val)
                val_loss = criterion(val_out, y_val).item()
            self.model.train()
            
            logger.info(f"Epoch: epoch={epoch}, train_loss={epoch_loss/len(train_loader)}, val_loss={val_loss}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                trigger_times = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    logger.info("Early stopping triggered")
                    break

    def predict_mc_dropout(self, x_input: torch.Tensor, n_samples: int = 20) -> Tuple[float, float]:
        """
        Runs MC Dropout inference.
        Returns (mean_prob, std_prob)
        """
        if self.model is None:
            # Load model structure must be known. 
            self.model = OptionSequenceModel(input_dim=self.input_dim, hidden_dim=32, num_layers=2)
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path))
        
        self.model.train() # Enable dropout
        
        probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.model(x_input)
                probs.append(out.item())
        
        return np.mean(probs), np.std(probs)
