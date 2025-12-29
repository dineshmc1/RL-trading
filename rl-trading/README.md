# RL Trading System with HMM Regime Detection

A sophisticated end-to-end Reinforcement Learning (RL) trading framework equipped with Hidden Markov Model (HMM) for market regime detection. This system trains autonomous agents to navigate financial markets by adapting their strategies to different market conditions.

## Overview

This project implements a custom Gymnasium-based trading environment where RL agents (using Stable-Baselines3) learn to trade various assets (Cryptocurrencies and Stocks). The system leverages advanced technical analysis and machine learning techniques to provide the agent with a comprehensive view of market dynamics.

### Key Capabilities
- **Multi-Asset Training**: Supports training across diverse symbols including BTC-USD, ETH-USD, AAPL, NVDA, and GOOGL.
- **HMM Regime Detection**: Automatically identifies market regimes (e.g., Bull, Bear, Sideways) using Gaussian Hidden Markov Models, providing the agent with context-aware features.
- **Flexible Reward Functions**: Support for various risk-adjusted reward metrics including Sharpe Ratio, Sortino Ratio, and Log Returns.
- **Automated Pipeline**: End-to-end workflow from data acquisition to feature engineering, model training, and performance evaluation.

---

## Technology Stack

- **Core Logic**: Python 3.12+
- **Reinforcement Learning**: `Stable-Baselines3` (PPO), `Gymnasium`
- **Machine Learning**: `hmmlearn` (Regime Detection), `scikit-learn`
- **Data & Indicators**: `yfinance`, `pandas-ta`
- **Analysis & Viz**: `matplotlib`, `pandas`, `numpy`

---

## Project Structure

```text
rl-trading/
├── agent.py          # RL Agent wrapper (SB3 integration)
├── data_loader.py    # Data acquisition, indicators, and HMM regimes
├── env.py            # Custom Gymnasium trading environment
├── main.py           # Entry point for training and evaluation
├── utils.py          # Logging and helper utilities
├── models/           # Persistent storage for trained models
└── plots/            # Generated performance visualizations
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rl-trading
   ```

2. **Set up the environment**:
   It is recommended to use a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # OR if using the provided pyproject.toml
   pip install .
   ```

---

## Usage

The system is controlled via a command-line interface in `main.py`.

### 1. Training and Evaluation (Default)
To train agents on all default assets (BTC, ETH, AAPL, NVDA, GOOGL) and evaluate them immediately:
```bash
python main.py --mode both --reward sortino
```

### 2. Specific Asset Training
To train only on specific assets with a set number of timesteps:
```bash
python main.py --mode train --assets BTC-USD AAPL --timesteps 50000
```

### 3. Evaluation Only
To evaluate existing models and generate performance plots:
```bash
python main.py --mode eval --assets BTC-USD
```

---

## Performance Metrics

The system generates a comprehensive performance report for each evaluation run, including:
- **Total Return**: Cumulative P&L of the RL strategy.
- **Buy & Hold Comparison**: Benchmark against a simple holding strategy.
- **Sharpe/Sortino Ratio**: Risk-adjusted performance metrics.
- **Max Drawdown**: Maximum peak-to-trough decline.

Visual results are saved in the `plots/` directory, showing the equity curve and specific trading actions (Buy/Sell) overlaid on the price chart.

---

## Disclaimer

*This project is for educational purposes only. Trading involves significant risk. The authors are not responsible for any financial losses incurred from using this software.*
