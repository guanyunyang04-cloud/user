import torch
import torch.nn as pd_nn # alias just to avoid shadowing
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor for Stable-Baselines3 that uses an LSTM network
    to process time-series market data before passing it to the PPO Actor/Critic networks.
    
    This solves the "amnesia" problem of standard MLPs by maintaining a hidden state
    across the window of historical bars provided in the observation.
    """
    def __init__(self, observation_space: gym.spaces.Box, window_size: int, num_market_features: int, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.num_market_features = num_market_features
        # Account features are whatever is left after extracting the market history matrix
        self.num_account_features = observation_space.shape[0] - (window_size * num_market_features)
        
        # 1. 1D-CNN to extract local patterns from market features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.num_market_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 2. The LSTM network for processing the market time series
        # input_size: matches CNN out_channels
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True
        )
        
        # 2. A linear layer to process the private account state (balance, inventory, etc.)
        self.account_net = nn.Sequential(
            nn.Linear(self.num_account_features, 32),
            nn.ReLU()
        )
        
        # 3. Final projection layer that combines LSTM output and account state output
        # 64 (from LSTM) + 32 (from account net) = 96
        self.linear = nn.Sequential(
            nn.Linear(64 + 32, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Processes the flat observation vector back into its components, passes the
        market data through the LSTM, and concatenates it with the account state.
        """
        # Split the flattened observation tensor
        # observations shape: (batch_size, window_size * market_features + account_features)
        market_flat = observations[:, :self.window_size * self.num_market_features]
        account_data = observations[:, self.window_size * self.num_market_features:]
        
        # Reshape market data for CNN: (batch_size, channels, sequence_length)
        # Sequence length is window_size (e.g., 30 bars)
        market_seq = market_flat.view(-1, self.window_size, self.num_market_features)
        market_seq = market_seq.permute(0, 2, 1) # Shape: (batch, features, seq)
        
        # Pass through CNN
        cnn_out = self.cnn(market_seq)
        
        # Reshape back for LSTM: (batch, seq, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # Pass through LSTM
        # We only care about the last output of the sequence (the summary of the past)
        lstm_out, (hn, cn) = self.lstm(cnn_out)
        # lstm_out shape: (batch, seq, hidden_size). We take the last timestep `[:, -1, :]`
        last_lstm_out = lstm_out[:, -1, :] 
        
        # Pass account data through its small network
        account_out = self.account_net(account_data)
        
        # Concatenate temporal market features with current account state
        combined = torch.cat((last_lstm_out, account_out), dim=1)
        
        # Final projection to the features_dim expected by SB3 (default 128)
        return self.linear(combined)


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    A non-recurrent feature extractor for RecurrentPPO.
    Uses CNN over the sliding window, then pools to a fixed vector.
    This avoids "double memory" when the policy is already recurrent.
    """
    def __init__(self, observation_space: gym.spaces.Box, window_size: int, num_market_features: int, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.window_size = window_size
        self.num_market_features = num_market_features
        self.num_account_features = observation_space.shape[0] - (window_size * num_market_features)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.num_market_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.account_net = nn.Sequential(
            nn.Linear(self.num_account_features, 32),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(32 + 32, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, window_size * market_features + account_features)
        market_flat = observations[:, :self.window_size * self.num_market_features]
        account_data = observations[:, self.window_size * self.num_market_features:]

        # Reshape market data for CNN: (batch_size, channels, sequence_length)
        market_seq = market_flat.view(-1, self.window_size, self.num_market_features)
        market_seq = market_seq.permute(0, 2, 1)
        cnn_out = self.cnn(market_seq)

        # Global average pool over time axis
        pooled = torch.mean(cnn_out, dim=2)

        account_out = self.account_net(account_data)
        combined = torch.cat((pooled, account_out), dim=1)
        return self.linear(combined)
