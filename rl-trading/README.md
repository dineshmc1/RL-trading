# Advanced RL Trading System with Probabilistic Regime Detection

An professional-grade Reinforcement Learning (RL) trading framework featuring continuous action spaces, leak-proof Hidden Markov Models (HMM) for regime detection, and risk-aware logarithmic reward functions.

## Overview

This system leverages State-of-the-Art (SOTA) techniques in Reinforcement Learning to navigate financial markets. Unlike traditional discrete trading bots, this agent learns a continuous policy, allowing for precise position sizing and dynamic exposure management across Cryptocurrencies and Equities.

### Core Upgrades & Features
- **Continuous Action Space**: The agent operates in a continuous domain `[-1, 1]`, enabling smooth transitions between Fully Short (-1.0), Flat (0.0), and Fully Long (+1.0) positions.
- **Probabilistic Regime Detection**: Utilizes Gaussian Hidden Markov Models (HMM) to output probability distributions across 3 market regimes (Bull, Bear, Sideways).
- **Leak-Proof Training Architecture**: The HMM is trained exclusively on historical training data to eliminate look-ahead bias and ensure robust out-of-sample performance.
- **Logarithmic Reward Function**: Implements $R_t = \log(1 + r_t) - \lambda \cdot DrawdownPenalty$, providing a geometrically consistent and risk-averse objective for the agent.
- **Advanced Feature Engineering**:
    - **Trend Analysis**: 20-day linear regression slope and VWAP distance.
    - **Volatility**: Realized volatility (rolling std of log returns).
    - **Regime Context**: Probabilistic signals, regime duration, and state-specific statistics.

---

## Technology Stack

- **Reinforcement Learning**: `Stable-Baselines3` (PPO), `Gymnasium`
- **Probabilistic Modeling**: `hmmlearn` (Gaussian HMM)
- **Technical Analysis**: `pandas-ta` (RSI, MACD, VWAP, ATR)
- **Data Pipeline**: `yfinance`, `scikit-learn` (Standard Scaling, Linear Regression)
- **Visualization**: `matplotlib` (Multi-panel equity and position charts)

---

## Project Structure

```text
rl-trading/
├── agent.py          # SB3 PPO Agent wrapper for continuous policies
├── data_loader.py    # Feature engineering & Leak-proof HMM pipeline
├── env.py            # Continuous Gymnasium environment with Log rewards
├── main.py           # CLI for training, evaluation, and visualization
├── utils.py          # Logging and directory management
├── models/           # Serialized RL model weights (.zip)
└── plots/            # High-fidelity performance visualizations
```

---

## Installation

1. **Clone and Navigate**:
   ```bash
   git clone <repository-url>
   cd rl-trading
   ```

2. **Environment Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install .
   ```

---

## Usage

### 1. Unified Pipeline (Train & Eval)
Train agents on default assets (BTC, ETH, AAPL, NVDA, GOOGL) and generate evaluation reports:
```bash
python main.py --mode both --timesteps 50000 --reward sortino
```

### 2. Targeted Training
Train a specific asset with a custom timeline:
```bash
python main.py --mode train --assets BTC-USD --timesteps 100000
```

### 3. Comprehensive Evaluation
Run evaluation on existing models to generate the 3-panel performance plots:
```bash
python main.py --mode eval --assets AAPL NVDA
```

---

## Performance Analysis

The system generates a comprehensive performance report for each evaluation run, including:
- **Trading Years**: The exact duration of the evaluation period.
- **Total Return**: Cumulative P&L of the RL strategy.
- **Annualized Return (%)**: The geometric average annual return.
- **Buy & Hold Comparison**: Benchmark against a simple holding strategy (Total and Annualized).
- **Sharpe/Sortino Ratio**: Risk-adjusted performance metrics.
- **Max Drawdown**: Maximum peak-to-trough decline.
3. **Target Position**: A filled area chart showing the agent's exact fractional exposure (Long vs. Short) over time.

---

## Disclaimer

*This project is for research and educational purposes only. Financial trading involves significant risk of loss. The authors do not guarantee any results and are not liable for financial outcomes.*
