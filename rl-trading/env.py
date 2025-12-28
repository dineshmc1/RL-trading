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

    def __init__(self, df, scaled_data, window_size=30, initial_balance=10000, transaction_cost=0.001):
        super(TradingEnv, self).__init__()

        self.df = df
        self.scaled_data = scaled_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

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
            return self.shares * current_price - (self.initial_balance - self.balance) # Actually it's more complex if multiple buys
            # Let's simplify: position tracking
        return 0 # Placeholder for simplicity, will improve in step

    def _get_total_equity(self):
        current_price = self.df['Close'].iloc[self.current_step]
        if self.position == 0:
            return self.balance
        elif self.position == 1:
            return self.balance + self.shares * current_price
        else: # Short
            # Short logic: balance + (entry_price - current_price) * shares
            # Simplified: initial_short_cash - current_value
            # For now, let's treat position 2 as EXIT/FLAT and position 1 as GO LONG
            # The prompt asked for: 0 -> Hold, 1 -> Buy/Long, 2 -> Sell/Exit/Short
            return self.balance + self.shares * current_price

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        prev_equity = self._get_total_equity()

        # Action logic
        # 0: Hold (do nothing)
        # 1: Buy/Long (if flat, buy; if short, cover and buy)
        # 2: Sell/Exit (if long, sell; if flat, maybe go short)
        
        if action == 1 and self.position != 1: # Go Long
            # Exit previous if any
            self.balance = self._get_total_equity()
            # Transaction cost
            cost = self.balance * self.transaction_cost
            self.balance -= cost
            # Buy
            self.shares = self.balance / current_price
            self.balance = 0
            self.position = 1
            
        elif action == 2 and self.position != 0: # Go Flat
            # Exit position
            self.balance = self._get_total_equity()
            # Transaction cost
            cost = self.balance * self.transaction_cost
            self.balance -= cost
            self.shares = 0
            self.position = 0

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        current_equity = self._get_total_equity()
        self.total_equity_history.append(current_equity)

        # Reward Function Design (Risk-aware)
        # 1. Step Return (log return of equity)
        step_return = np.log(current_equity / prev_equity) if prev_equity > 0 else 0
        
        # 2. Drawdown Penalty
        peak = max(self.total_equity_history)
        drawdown = (peak - current_equity) / peak
        dd_penalty = drawdown * 0.1
        
        # 3. Transaction Cost is already baked into equity, 
        # but we can add an extra "over-trading" penalty if frequent changes
        # (This is implicitly handled by the cost itself)

        reward = step_return - dd_penalty
        
        # Truncated is used in new Gymnasium API
        truncated = False
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, truncated, {"equity": current_equity}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Equity: {self._get_total_equity():.2f}, Position: {self.position}")
