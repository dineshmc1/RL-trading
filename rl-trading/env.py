import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Gymnasium
    Actions: 0 (Hold), 1 (Buy/Long), 2 (Sell/Short)
    Supports Multi-Asset training with Asset Embeddings.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dict, defined_assets=None, window_size=30, initial_balance=10000, 
                 transaction_cost=0.001, reward_type='sortino', 
                 random_start=True, randomize_cost=False):
        super(TradingEnv, self).__init__()

        # Handle both single asset (dict or not) and multi-asset dicts
        if not isinstance(data_dict, dict) or 'train' in data_dict: 
             pass 

        self.data_dict = data_dict # {Symbol: (df, scaled_data)}
        
        # Consistent Asset Universe (for Embeddings)
        if defined_assets is not None:
            self.assets = defined_assets
        else:
            self.assets = list(data_dict.keys())
            
        # Assets available for active sampling in this env instance
        self.active_assets = [a for a in self.data_dict.keys() if a in self.assets]
        if not self.active_assets:
            raise ValueError("No active assets found in data_dict matching defined_assets.")

        self.num_assets = len(self.assets)
        self.asset_mapping = {symbol: i for i, symbol in enumerate(self.assets)}
        
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.base_transaction_cost = transaction_cost
        self.reward_type = reward_type 
        self.random_start = random_start
        self.randomize_cost = randomize_cost

        self.action_space = spaces.Discrete(3)

        # Observation space: 
        # [Features (N) | Portfolio State (4) | Asset Embedding (num_assets)]
        # We assume all active assets have same number of features
        first_asset = self.active_assets[0]
        _, first_scaled = self.data_dict[first_asset]
        num_features = first_scaled.shape[1]
        
        # Total observation size per timestep
        self.obs_dim = num_features + 4 + self.num_assets
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, self.obs_dim), 
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Select asset
        if options and 'asset' in options:
            self.current_asset_symbol = options['asset']
            if self.current_asset_symbol not in self.active_assets:
                raise ValueError(f"Asset {self.current_asset_symbol} not in active assets.")
        else:
            self.current_asset_symbol = random.choice(self.active_assets)
            
        self.current_asset_idx = self.asset_mapping[self.current_asset_symbol]
        self.df, self.scaled_data = self.data_dict[self.current_asset_symbol]
        
        self.balance = self.initial_balance
        self.position = 0 
        self.shares = 0
        
        # Random start step
        if self.random_start and len(self.df) > self.window_size + 100:
            max_start = len(self.df) - 100 
            self.current_step = random.randint(self.window_size, max_start)
        else:
            self.current_step = self.window_size
            
        # Randomized Transaction Cost (Gaussian) for this episode or step?
        # User asked for "Randomized transaction costs". Let's vary it per episode or constant?
        # Usually per episode is good to test robustness.
        if self.randomize_cost:
            # Mean = base_cost, Std = base_cost * 0.2 (20% variablity)
            self.current_transaction_cost = max(0, np.random.normal(self.base_transaction_cost, self.base_transaction_cost * 0.2))
        else:
            self.current_transaction_cost = self.base_transaction_cost

        self.history = []
        self.total_equity_history = [self.initial_balance]
        self.returns_history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 1. Market Features
        window_start = self.current_step - self.window_size
        obs_features = self.scaled_data[window_start:self.current_step].copy()
        
        # 2. Portfolio State
        p_state = np.array([
            self.position,
            self.balance / self.initial_balance,
            self._get_unrealized_pnl() / self.initial_balance,
            self._get_total_equity() / self.initial_balance
        ])
        
        # 3. Asset Embedding (One-Hot)
        embedding = np.zeros(self.num_assets)
        embedding[self.current_asset_idx] = 1.0
        
        # Combine [Features, Portfolio, Embedding]
        # p_state and embedding need to be tiled to match window size?
        # Typically LSTM/Transformers need sequence. MLP policies flatten anyway.
        # But `env` usually returns (Window, Channels/Features).
        # So we repeat static info across the window.
        
        static_info = np.concatenate([p_state, embedding])
        static_window = np.tile(static_info, (self.window_size, 1))
        
        full_obs = np.hstack([obs_features, static_window])
        return full_obs.astype(np.float32)

    def _get_unrealized_pnl(self):
        current_price = self.df['Close'].iloc[self.current_step]
        if self.position == 1: # Long
            return self.shares * current_price - (self.initial_balance - self.balance) 
        return 0 

    def _get_total_equity(self):
        current_price = self.df['Close'].iloc[self.current_step]
        if self.position == 0:
            return self.balance
        elif self.position == 1:
            return self.balance + self.shares * current_price
        else: # Short
            # Simplified short logic: Inverse of Long
            return self.balance + self.shares * current_price

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        prev_equity = self._get_total_equity()

        # Action logic
        if action == 1 and self.position != 1: # Go Long
            self.balance = self._get_total_equity()
            cost = self.balance * self.current_transaction_cost
            self.balance -= cost
            self.shares = self.balance / current_price
            self.balance = 0
            self.position = 1
        elif action == 2 and self.position != 0: # Go Flat
            self.balance = self._get_total_equity()
            cost = self.balance * self.current_transaction_cost
            self.balance -= cost
            self.shares = 0
            self.position = 0

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        current_equity = self._get_total_equity()
        self.total_equity_history.append(current_equity)

        # Reward Calculation
        step_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        self.returns_history.append(step_return)
        
        reward = 0
        if self.reward_type == 'return':
            log_return = np.log(current_equity / prev_equity) if prev_equity > 0 else 0
            peak = max(self.total_equity_history)
            drawdown = (peak - current_equity) / peak
            reward = log_return - (drawdown * 0.1)
            
        elif self.reward_type == 'sharpe':
            if len(self.returns_history) < self.window_size:
                reward = step_return
            else:
                window_returns = self.returns_history[-self.window_size:]
                std = np.std(window_returns)
                reward = np.mean(window_returns) / (std + 1e-6)
                
        elif self.reward_type == 'sortino':
            if len(self.returns_history) < self.window_size:
                reward = step_return
            else:
                window_returns = self.returns_history[-self.window_size:]
                downside_returns = [r for r in window_returns if r < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-6
                reward = np.mean(window_returns) / (downside_std + 1e-6)

        truncated = False
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, truncated, {"equity": current_equity, "position": self.position}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Asset: {self.current_asset_symbol}, Equity: {self._get_total_equity():.2f}")
