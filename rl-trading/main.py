import os
import argparse
import pandas as pd
from data_loader import DataLoader
from env import TradingEnv
from agent import RLAgent
from utils import set_all_seeds, create_dirs, Logger

# Re-importing evaluation logic
import numpy as np
import matplotlib.pyplot as plt

def train_asset(symbol, start_date, end_date, timesteps=20000, reward_type='sortino'):
    Logger.info(f"--- Training Agent for {symbol} (Reward: {reward_type}) ---")
    loader = DataLoader([symbol], start_date, end_date)
    data = loader.prepare_data(symbol)
    train_df, train_scaled = data['train']
    
    env = TradingEnv(train_df, train_scaled, window_size=30, reward_type=reward_type)
    agent = RLAgent(env)
    agent.train(total_timesteps=timesteps)
    
    model_dir = "models"
    create_dirs([model_dir])
    model_path = os.path.join(model_dir, f"ppo_{symbol.replace('-', '_')}")
    agent.save(model_path)
    Logger.info(f"Model saved to {model_path}")

def compute_metrics(equity_curve, buy_and_hold_equity, df):
    # Duration in years
    start_date = df.index[0]
    end_date = df.index[-1]
    duration_days = (end_date - start_date).days
    trading_years = duration_days / 365.25
    
    total_return = (equity_curve[-1] - equity_curve[0]) / (equity_curve[0] + 1e-6)
    bh_return = (buy_and_hold_equity[-1] - buy_and_hold_equity[0]) / (buy_and_hold_equity[0] + 1e-6)
    
    # Annualized Return (Geometric)
    annualized_return = (1 + total_return)**(1 / trading_years) - 1 if trading_years > 0 else 0
    bh_annualized_return = (1 + bh_return)**(1 / trading_years) - 1 if trading_years > 0 else 0
    
    daily_returns = pd.Series(equity_curve).pct_change().dropna()
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-6) * np.sqrt(252))
    
    peak = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - peak) / (peak + 1e-6)
    max_dd = drawdown.min()
    
    return {
        "Trading Years": trading_years,
        "Total Return (%)": total_return * 100,
        "CAGR (%)": annualized_return * 100,
        "B&H Return (%)": bh_return * 100,
        "B&H CAGR (%)": bh_annualized_return * 100,
        "Annualized Vol": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

def run_evaluation(symbol, start_date, end_date):
    Logger.info(f"--- Evaluating {symbol} ---")
    loader = DataLoader([symbol], start_date, end_date)
    data = loader.prepare_data(symbol)
    test_df, test_scaled = data['test']
    
    model_path = os.path.join("models", f"ppo_{symbol.replace('-', '_')}.zip")
    if not os.path.exists(model_path):
        Logger.error(f"Model not found for {symbol}. Run with --mode train first.")
        return None
    
    env = TradingEnv(test_df, test_scaled, window_size=30)
    agent = RLAgent(env, model_path=model_path)
    
    obs, _ = env.reset()
    done = False
    truncated = False
    actions = []
    equities = [env.initial_balance]
    positions = []
    
    while not (done or truncated):
        action = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        equities.append(info['equity'])
        positions.append(info['position'])
    
    bh_shares = env.initial_balance / (test_df['Close'].iloc[0] + 1e-6)
    bh_equity = test_df['Close'] * bh_shares
    
    metrics = compute_metrics(equities, bh_equity.values, test_df)
    
    # Track positions for plotting
    # Already tracked in the loop above
        
    plot_results(symbol, test_df, equities, bh_equity, positions)
    return metrics

def plot_results(symbol, df, equities, bh_equity, positions):
    create_dirs(["plots"])
    plt.figure(figsize=(15, 12))
    
    # 1. Equity Curve
    plt.subplot(3, 1, 1)
    plt.plot(equities, label='RL Agent Equity', color='blue', linewidth=2)
    plt.plot(bh_equity.values, label='Buy & Hold Equity', color='orange', linestyle='--')
    plt.title(f"Equity Curve - {symbol}")
    plt.ylabel("Portfolio Value")
    plt.legend(); plt.grid(True, alpha=0.3)
    
    # 2. Price and Markers (Markers for significant position changes)
    plt.subplot(3, 1, 2)
    plt.plot(df['Close'].values, label='Price', color='black', alpha=0.7)
    plt.title(f"Price Action - {symbol}")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # 3. Target Position (Continuous)
    plt.subplot(3, 1, 3)
    plt.fill_between(range(len(positions)), positions, 0, 
                     where=np.array(positions) >= 0, color='green', alpha=0.3, label='Long')
    plt.fill_between(range(len(positions)), positions, 0, 
                     where=np.array(positions) < 0, color='red', alpha=0.3, label='Short')
    plt.plot(positions, color='purple', linewidth=1, label='Target Position')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"RL Agent Position - {symbol}")
    plt.ylabel("Position (-1 to 1)")
    plt.ylim(-1.1, 1.1)
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"plots/results_{symbol.replace('-', '_')}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="RL Trading System")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], default="both", help="Mode: train, eval, or both")
    parser.add_argument("--reward", type=str, choices=["return", "sharpe", "sortino"], default="sortino", help="Reward function type")
    parser.add_argument("--assets", nargs="+", default=["BTC-USD", "ETH-USD", "AAPL", "NVDA", "GOOGL"], help="Assets to process")
    parser.add_argument("--timesteps", type=int, default=20000, help="Training timesteps")
    args = parser.parse_args()

    set_all_seeds(42)
    start_date = "2018-01-01"
    end_date = "2023-12-31"

    if args.mode in ["train", "both"]:
        for asset in args.assets:
            train_asset(asset, start_date, end_date, timesteps=args.timesteps, reward_type=args.reward)

    if args.mode in ["eval", "both"]:
        results = []
        for asset in args.assets:
            m = run_evaluation(asset, start_date, end_date)
            if m:
                m['Symbol'] = asset
                results.append(m)
        
        if results:
            df_results = pd.DataFrame(results)
            print("\nSummary Metrics Table:")
            print(df_results.to_string())

if __name__ == "__main__":
    main()
