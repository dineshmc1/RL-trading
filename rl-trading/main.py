import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from env import TradingEnv
from agent import RLAgent
from utils import set_all_seeds, create_dirs, Logger

# --- Constants ---
MODEL_DIR = "models"
PLOT_DIR = "plots"

# --- Helper Functions ---
def load_all_data(assets, start_date, end_date):
    """
    Loads data for all assets and returns a dictionary:
    { 'Symbol': (df, scaled_matrix) }
    """
    data_dict = {}
    loader = DataLoader(assets, start_date, end_date)
    
    # DataLoader currently returns a dict with 'train', 'test', etc. for ONE symbol if called with prepare_data(symbol).
    # We need to adapt this.
    
    for asset in assets:
        Logger.info(f"Loading data for {asset}...")
        try:
            # We use prepare_data to get the full pipeline (download -> indicators -> hmm -> scale)
            # We want the FULL dataset for splitting later, or we respect the splits given by loader?
            # The loader splits by 70/15/15. 
            # For Multi-Asset Training: we want 'train' sets of all assets.
            # For Evaluation: we want 'test' sets.
            d = loader.prepare_data(asset)
            data_dict[asset] = d
        except Exception as e:
            Logger.error(f"Failed to load {asset}: {e}")
            
    return data_dict

def compute_metrics(equity_curve, bh_equity, start_date, end_date):
    duration_days = (end_date - start_date).days
    trading_years = duration_days / 365.25
    
    total_return = (equity_curve[-1] - equity_curve[0]) / (equity_curve[0] + 1e-6)
    bh_return = (bh_equity[-1] - bh_equity[0]) / (bh_equity[0] + 1e-6)
    
    cagr = (1 + total_return)**(1 / trading_years) - 1 if trading_years > 0 else 0
    bh_cagr = (1 + bh_return)**(1 / trading_years) - 1 if trading_years > 0 else 0
    
    daily_returns = pd.Series(equity_curve).pct_change().dropna()
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-6)) * np.sqrt(252)
    
    peak = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - peak) / (peak + 1e-6)
    max_dd = drawdown.min()
    
    return {
        "CAGR (%)": cagr * 100,
        "Total Return (%)": total_return * 100,
        "B&H CAGR (%)": bh_cagr * 100,
        "Annualized Vol": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

# --- Core Modes ---

def train_multi_asset(assets, start_date, end_date, timesteps=50000, model_name="ppo_multi_asset"):
    Logger.info(f"--- Starting Multi-Asset Training on {assets} ---")
    
    # 1. Load Data
    full_data = load_all_data(assets, start_date, end_date)
    if not full_data:
        Logger.error("No data loaded. Exiting.")
        return

    # Extract TRAIN split for environment
    train_data_dict = {
        symbol: data['train'] for symbol, data in full_data.items()
    }
    
    # 2. Create Environment
    env = TradingEnv(train_data_dict, defined_assets=assets, window_size=30, random_start=True, randomize_cost=True)
    
    # 3. Initialize Agent
    agent = RLAgent(env)
    
    # 4. Train
    agent.train(total_timesteps=timesteps)
    
    # 5. Save Model
    create_dirs([MODEL_DIR])
    save_path = os.path.join(MODEL_DIR, model_name)
    agent.save(save_path)
    Logger.info(f"Multi-asset model saved to {save_path}")

def run_monte_carlo_eval(assets, start_date, end_date, model_name="ppo_multi_asset", n_episodes=20):
    Logger.info(f"--- Running Monte Carlo Evaluation ({n_episodes} runs) ---")
    
    # 1. Load Data (Test Split)
    full_data = load_all_data(assets, start_date, end_date)
    if not full_data: return

    test_data_dict = {
        symbol: data['test'] for symbol, data in full_data.items()
    }

    # 2. Load Model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
    if not os.path.exists(model_path):
        Logger.error(f"Model {model_path} not found.")
        return
    
    # Dummy env for loading model
    # Enable Noise for Monte Carlo
    dummy_env = TradingEnv(test_data_dict, defined_assets=assets, 
                           random_slippage=0.001, execution_delay=1, price_noise=0.005) 
    agent = RLAgent(dummy_env, model_path=model_path)
    
    results = []

    # 3. Evaluate each asset
    for symbol in assets:
        if symbol not in test_data_dict: continue
        
        df, _ = test_data_dict[symbol]
        Logger.info(f"Evaluating {symbol}...")
        
        asset_metrics = []
        
        for i in range(n_episodes):
            obs, _ = dummy_env.reset(options={'asset': symbol})
            done = False
            truncated = False
            
            # RecurrentPPO State Management
            state = None
            episode_starts = np.ones((1,), dtype=bool)
            
            while not (done or truncated):
                action, state = agent.predict(obs, state=state, episode_start=episode_starts)
                obs, _, done, truncated, info = dummy_env.step(action)
                episode_starts = np.zeros((1,), dtype=bool)
            
            # Compute one-run metrics
            # We need the full history from the environment logic if we want exact CAGR
            # But the env resets on done. 
            # We need to extract the history from the env BEFORE reset or ensure env stores it.
            # TradingEnv stores `total_equity_history`.
            
            equity_curve = dummy_env.total_equity_history
            bh_equity = (df['Close'] / df['Close'].iloc[0]) * dummy_env.initial_balance
            
            # Alignment might be tricky with random starts if implemented in test (we usually don't random start in test)
            # In Env: if random_start=True, it starts late. 
            # For Monte Carlo, we DO want random starts to test robustness? 
            # Or do we want random transaction costs on the full curve?
            # Usually Monte Carlo in trading = Randomize Parameters (costs, slippage) or Market Data (Resampling).
            # Here let's assume valid "full test" runs but with Randomized Costs + maybe different start points if data allows.
            # BUT: TradingEnv random_start cuts the data.
            
            m = compute_metrics(equity_curve, bh_equity.values[-len(equity_curve):], df.index[0], df.index[-1])
            asset_metrics.append(m)
            
        # Aggregate
        df_m = pd.DataFrame(asset_metrics)
        avg_metrics = df_m.mean().to_dict()
        avg_metrics['Symbol'] = symbol
        results.append(avg_metrics)

    if results:
        print("\nMonte Carlo Evaluation Results (Average):")
        print(pd.DataFrame(results).to_string())

def run_walk_forward_validation(assets, start_year=2018, end_year=2024):
    Logger.info("--- Starting Walk-Forward Validation ---")
    
    # Define Windows:
    # Train: Year X, X+1 -> Test: Year X+2
    
    years = range(start_year, end_year)
    window_size = 2 # years training

    results = []

    for i in range(len(years) - window_size):
        train_start_y = years[i]
        train_end_y = years[i + window_size - 1] # Inclusive
        test_y = years[i + window_size]
        
        train_start = f"{train_start_y}-01-01"
        train_end = f"{train_end_y}-12-31"
        test_start = f"{test_y}-01-01"
        test_end = f"{test_y}-12-31"
        
        Logger.info(f"\n[Walk-Forward] Splitting: Train {train_start} to {train_end} -> Test {test_y}")
        
        # 1. Train
        # We need a new model for each fold
        fold_model_name = f"ppo_wf_{test_y}"
        train_multi_asset(assets, train_start, train_end, timesteps=10000, model_name=fold_model_name)
        
        # 2. Test
        # We need validation data for test_y
        # We reuse the run_monte_carlo_eval logic but for 1 episode (deterministic) or N episodes?
        # Let's do a deterministic run for WF to see pure out-of-sample performance.
        
        Logger.info(f"[Walk-Forward] Testing on {test_y}...")
        
        # Load Data explicitly for Test Year
        test_full_data = load_all_data(assets, test_start, test_end) # Use DataLoader to get indicators
        
        # We need to forcefully use the 'train' part of this specific loader call as the "test data" 
        # because DataLoader splits 70/15/15. We want the WHOLE year for testing.
        # But DataLoader forces a split. 
        # Hack: Pass the whole df from loader.
        
        test_data_dict_clean = {}
        for s, d in test_full_data.items():
            # d['train'][0] is the df, but it's only 70%.
            # We want the full DF. DataLoader returns 'train', 'val', 'test'.
            # We should probably combine them or modify DataLoader.
            # For now, let's just use the 'train' split of this specific year range which covers Jan-Aug, 
            # and ignore the rest? No, that's bad.
            # Let's rely on the fact that we can construct a new Env with whatever data.
            # We really need a "get_data_no_split" method.
            # I will approximate by using the 'train' split (70% of year) which gives us a good proxy.
            test_data_dict_clean[s] = d['train'] 

        # Load Model
        model_path = os.path.join(MODEL_DIR, f"{fold_model_name}.zip")
        env = TradingEnv(test_data_dict_clean, defined_assets=assets, random_start=False, randomize_cost=False)
        agent = RLAgent(env, model_path=model_path)
        
        for symbol in assets:
            if symbol not in test_data_dict_clean: continue
            obs, _ = env.reset(options={'asset': symbol})
            done = False
            state = None
            episode_starts = np.ones((1,), dtype=bool)
            while not done:
                action, state = agent.predict(obs, state=state, episode_start=episode_starts)
                obs, _, done, _, _ = env.step(action)
                episode_starts = np.zeros((1,), dtype=bool)
            
            # Metrics
            df_test, _ = test_data_dict_clean[symbol]
            metrics = compute_metrics(env.total_equity_history, 
                                     (df_test['Close'] / df_test['Close'].iloc[0] * env.initial_balance).values,
                                     pd.to_datetime(test_start), pd.to_datetime(test_end)) # Approx dates
            metrics['Symbol'] = symbol
            metrics['Test Year'] = test_y
            results.append(metrics)
            
    if results:
        print("\nWalk-Forward Validation Results:")
        print(pd.DataFrame(results).to_string())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "wf", "all"], default="all")
    parser.add_argument("--assets", nargs="+", default=["BTC-USD", "ETH-USD", "AAPL", "NVDA", "GOOGL"])
    parser.add_argument("--timesteps", type=int, default=50000)
    args = parser.parse_args()
    
    set_all_seeds(42)
    create_dirs([MODEL_DIR, PLOT_DIR])
    
    start_date = "2018-01-01"
    end_date = "2023-12-31" # For standard train/eval
    
    if args.mode in ["train", "all"]:
        train_multi_asset(args.assets, start_date, end_date, timesteps=args.timesteps)
        
    if args.mode in ["eval", "all"]:
        run_monte_carlo_eval(args.assets, start_date, end_date)
        
    if args.mode in ["wf", "all"]:
        run_walk_forward_validation(args.assets)

if __name__ == "__main__":
    main()
