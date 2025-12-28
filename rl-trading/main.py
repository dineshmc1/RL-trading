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

def train_asset(symbol, start_date, end_date, timesteps=20000):
    Logger.info(f"--- Training Agent for {symbol} ---")
    loader = DataLoader([symbol], start_date, end_date)
    data = loader.prepare_data(symbol)
    train_df, train_scaled = data['train']
    
    env = TradingEnv(train_df, train_scaled, window_size=30)
    agent = RLAgent(env)
    agent.train(total_timesteps=timesteps)
    
    model_dir = "models"
    create_dirs([model_dir])
    model_path = os.path.join(model_dir, f"ppo_{symbol.replace('-', '_')}")
    agent.save(model_path)
    Logger.info(f"Model saved to {model_path}")

def compute_metrics(equity_curve, buy_and_hold_equity):
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    bh_return = (buy_and_hold_equity[-1] - buy_and_hold_equity[0]) / buy_and_hold_equity[0]
    
    daily_returns = pd.Series(equity_curve).pct_change().dropna()
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
    
    peak = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - peak) / peak
    max_dd = drawdown.min()
    
    return {
        "Total Return (%)": total_return * 100,
        "B&H Return (%)": bh_return * 100,
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
    
    while not (done or truncated):
        action = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        actions.append(action)
        equities.append(info['equity'])
    
    bh_shares = env.initial_balance / test_df['Close'].iloc[0]
    bh_equity = test_df['Close'] * bh_shares
    
    metrics = compute_metrics(equities, bh_equity.values)
    plot_results(symbol, test_df, equities, bh_equity, actions)
    return metrics

def plot_results(symbol, df, equities, bh_equity, actions):
    create_dirs(["plots"])
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(equities, label='RL Agent Equity', color='blue')
    plt.plot(bh_equity.values, label='Buy & Hold Equity', color='orange', linestyle='--')
    plt.title(f"Equity Curve - {symbol}")
    plt.legend(); plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['Close'].values, label='Price', color='gray', alpha=0.5)
    buy_indices = [i for i, a in enumerate(actions) if a == 1]
    sell_indices = [i for i, a in enumerate(actions) if a == 2]
    plt.scatter(buy_indices, df['Close'].iloc[buy_indices], marker='^', color='green', label='Buy')
    plt.scatter(sell_indices, df['Close'].iloc[sell_indices], marker='v', color='red', label='Sell')
    plt.title(f"Trading Actions - {symbol}")
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"plots/results_{symbol.replace('-', '_')}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="RL Trading System")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], default="both", help="Mode: train, eval, or both")
    parser.add_argument("--assets", nargs="+", default=["BTC-USD", "ETH-USD", "AAPL", "NVDA", "GOOGL"], help="Assets to process")
    parser.add_argument("--timesteps", type=int, default=20000, help="Training timesteps")
    args = parser.parse_args()

    set_all_seeds(42)
    start_date = "2018-01-01"
    end_date = "2023-12-31"

    if args.mode in ["train", "both"]:
        for asset in args.assets:
            train_asset(asset, start_date, end_date, timesteps=args.timesteps)

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
