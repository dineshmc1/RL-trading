import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Gymnasium
    Actions: Continuous [-1, 1] for Position Scaling.
    Supports Multi-Asset training, Asset Embeddings, Regime-Weighted Rewards, and Execution Noise.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dict, defined_assets=None, window_size=30, initial_balance=10000, 
                 transaction_cost=0.001, reward_type='sortino', 
                 random_start=True, randomize_cost=False,
                 random_slippage=0.0, execution_delay=0, price_noise=0.0):
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
        
        # New Evaluation Noise Parameters
        self.random_slippage = random_slippage 
        self.execution_delay = execution_delay 
        self.price_noise = price_noise 

        # Action space: Continuous (-1 to 1) for Position Scaling
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: 
        # [Features (N) | Portfolio State (4) | Asset Embedding (num_assets)]
        first_asset = self.active_assets[0]
        _, first_scaled = self.data_dict[first_asset]
        num_features = first_scaled.shape[1]
        
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
        self.position = 0.0 # Continuous position -1.0 to 1.0
        self.shares = 0
        
        # Random start step
        if self.random_start and len(self.df) > self.window_size + 100:
            max_start = len(self.df) - 100 
            self.current_step = random.randint(self.window_size, max_start)
        else:
            self.current_step = self.window_size
            
        # Randomized Transaction Cost
        if self.randomize_cost:
            # Mean = base_cost, Std = base_cost * 0.2
            self.current_transaction_cost = max(0, np.random.normal(self.base_transaction_cost, self.base_transaction_cost * 0.2))
        else:
            self.current_transaction_cost = self.base_transaction_cost

        # Slippage for this episode
        self.current_slippage = np.random.uniform(0, self.random_slippage) if self.random_slippage > 0 else 0

        self.history = []
        self.total_equity_history = [self.initial_balance]
        self.returns_history = []
        
        # Action Delay Queue
        self.action_queue = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 1. Market Features
        window_start = self.current_step - self.window_size
        obs_features = self.scaled_data[window_start:self.current_step].copy()
        
        # Noise injection if price_noise > 0
        if self.price_noise > 0:
            noise = np.random.normal(0, self.price_noise, obs_features.shape)
            obs_features += noise

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
        
        static_info = np.concatenate([p_state, embedding])
        static_window = np.tile(static_info, (self.window_size, 1))
        
        full_obs = np.hstack([obs_features, static_window])
        return full_obs.astype(np.float32)

    def _get_unrealized_pnl(self):
        current_price = self.df['Close'].iloc[self.current_step]
        # PnL approximation for scaling
        value_of_shares = self.shares * current_price
        return value_of_shares - (self.initial_balance - self.balance) if self.position != 0 else 0

    def _get_total_equity(self):
        current_price = self.df['Close'].iloc[self.current_step]
        return self.balance + self.shares * current_price

    def step(self, action):
        # 1. Handle Execution Delay
        # Push action to queue
        self.action_queue.append(action)
        
        # Check if we can execute (queue length > delay)
        current_delay = np.random.randint(0, self.execution_delay + 1) if self.execution_delay > 0 else 0
        
        if len(self.action_queue) > current_delay:
            executed_action = self.action_queue.pop(0)
        else:
            # No action yet (during initial delay), hold previous position
            executed_action = np.array([self.position]) 

        target_position = np.clip(executed_action[0], -1, 1)
        
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Apply Slippage
        buy_price = current_price * (1 + self.current_slippage)
        sell_price = current_price * (1 - self.current_slippage)
        
        prev_equity = self._get_total_equity()

        # Rebalancing Logic
        current_value = self.shares * current_price
        
        # Target value based on total equity
        total_equity = prev_equity
        target_value = total_equity * target_position
        
        # Difference to trade
        # Note: self.position tracks fraction, but due to price moves, actual fraction drifts.
        # We rebalance towards target_position fraction of TOTAL EQUITY.
        
        diff_value = target_value - current_value
        
        if diff_value > 0: # Buy
            cost = diff_value * self.current_transaction_cost
            if self.balance >= diff_value + cost: # Can afford?
                self.balance -= (diff_value + cost)
                self.shares += diff_value / buy_price
        elif diff_value < 0: # Sell
            sell_val = abs(diff_value)
            cost = sell_val * self.current_transaction_cost
            self.balance += (sell_val - cost)
            self.shares -= sell_val / sell_price 
            
        self.position = target_position

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        current_equity = self._get_total_equity()
        self.total_equity_history.append(current_equity)

        # Advanced Reward Calculation
        # Base: log return
        log_return = np.log(current_equity / prev_equity) if prev_equity > 0 else 0
        
        # Drawdown Penalty
        peak = max(self.total_equity_history)
        drawdown = (peak - current_equity) / peak
        dd_penalty = 0.5 * max(0, drawdown - 0.05) # 5% limit
        
        # Realized Vol Penalty
        self.returns_history.append(log_return)
        if len(self.returns_history) > 20:
             realized_vol = np.std(self.returns_history[-20:])
        else:
             realized_vol = 0
             
        vol_penalty = 0.2 * realized_vol
        
        reward = log_return - dd_penalty - vol_penalty
        
        # Regime Conditional
        # Extract Regime from df. 'Regime' column.
        # Assuming HMM Regimes are available in df.
        regime = self.df['Regime'].iloc[self.current_step]
        # Heuristic: Let's assume Regime 1 is Bear.
        if regime == 1: 
            reward *= 1.5
            
        # Trend Bonus (Bull Market)
        # Using feature 'Trend_Slope' if available.
        # Assuming Trend_Slope is column 5 in features, but better to access from df via index.
        # However, df has the raw column.
        trend_slope = self.df['Trend_Slope'].iloc[self.current_step]
        if self.position > 0.5 and trend_slope > 0.05:
             reward += 0.001 
             
        truncated = False
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, truncated, {"equity": current_equity, "position": self.position}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Asset: {self.current_asset_symbol}, Equity: {self._get_total_equity():.2f}")
