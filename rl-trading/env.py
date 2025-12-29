import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Gymnasium
    Actions: 0 (Hold), 1 (Buy/Long), 2 (Sell/Short)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, scaled_data, window_size=30, initial_balance=10000, transaction_cost=0.001, reward_type='sortino'):
        super(TradingEnv, self).__init__()

        self.df = df
        self.scaled_data = scaled_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type # 'return', 'sharpe', or 'sortino'

        # Action space: 0: Hold, 1: Long, 2: Short (optional, but requested as "Sell/Exit/Short")
        # Let's implement Long, Flat, Short logic
        self.action_space = spaces.Discrete(3)

        # Observation space: Window of features + portfolio state
        # Portfolio state: [position, balance, unrealized_pnl, total_equity]
        # features: scaled_data.shape[1]
        num_features = scaled_data.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, num_features + 4), 
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0  # 0: flat, 1: long, -1: short
        self.shares = 0
        self.current_step = self.window_size
        self.history = []
        self.total_equity_history = [self.initial_balance]
        self.returns_history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Get window of scaled features
        window_start = self.current_step - self.window_size
        obs = self.scaled_data[window_start:self.current_step].copy()
        
        # Add portfolio state to each timestep in the window
        # (normalized by initial balance for scaling consistency where applicable)
        p_state = np.array([
            self.position,
            self.balance / self.initial_balance,
            self._get_unrealized_pnl() / self.initial_balance,
            self._get_total_equity() / self.initial_balance
        ])
        
        # Broadcast p_state to match window size
        p_state_window = np.tile(p_state, (self.window_size, 1))
        
        full_obs = np.hstack([obs, p_state_window])
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
            return self.balance + self.shares * current_price

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        prev_equity = self._get_total_equity()

        # Action logic
        if action == 1 and self.position != 1: # Go Long
            self.balance = self._get_total_equity()
            cost = self.balance * self.transaction_cost
            self.balance -= cost
            self.shares = self.balance / current_price
            self.balance = 0
            self.position = 1
        elif action == 2 and self.position != 0: # Go Flat
            self.balance = self._get_total_equity()
            cost = self.balance * self.transaction_cost
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
            # 1. Standard approach: log return - drawdown penalty
            log_return = np.log(current_equity / prev_equity) if prev_equity > 0 else 0
            peak = max(self.total_equity_history)
            drawdown = (peak - current_equity) / peak
            reward = log_return - (drawdown * 0.1)
            
        elif self.reward_type == 'sharpe':
            # 2. Rolling Sharpe Ratio (approximate over window)
            if len(self.returns_history) < self.window_size:
                reward = step_return
            else:
                window_returns = self.returns_history[-self.window_size:]
                std = np.std(window_returns)
                reward = np.mean(window_returns) / (std + 1e-6)
                
        elif self.reward_type == 'sortino':
            # 3. Rolling Sortino Ratio (only penalize downside deviations)
            if len(self.returns_history) < self.window_size:
                reward = step_return
            else:
                window_returns = self.returns_history[-self.window_size:]
                downside_returns = [r for r in window_returns if r < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 1 else (abs(min(window_returns)) if any(r < 0 for r in window_returns) else 1e-6)
                reward = np.mean(window_returns) / (downside_std + 1e-6)

        # Truncated is used in new Gymnasium API
        truncated = False
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, truncated, {"equity": current_equity}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Equity: {self._get_total_equity():.2f}, Position: {self.position}")
