import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from utils import Logger

class DataLoader:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.scalers = {}

    def download_data(self, symbol):
        """Download OHLCV data for a symbol."""
        Logger.info(f"Downloading data for {symbol}...")
        df = yf.download(symbol, start=self.start_date, end=self.end_date)
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Flatten columns if MultiIndex (sometimes yfinance does this)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df

    def add_indicators(self, df):
        """Add technical indicators."""
        df = df.copy()
        
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']
        
        # EMAs
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        
        # Volatility (ATR)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Drop rows with NaN from indicators
        df.dropna(inplace=True)
        return df

    def _add_hmm_regimes(self, df, n_regimes=3):
        """Add market regimes using Hidden Markov Model."""
        Logger.info(f"Detecting {n_regimes} market regimes using HMM...")
        
        # Features for HMM: Log Returns and Volatility (using rolling std of returns)
        returns = df['Log_Returns'].values.reshape(-1, 1)
        volatility = df['Log_Returns'].rolling(window=20).std().fillna(0).values.reshape(-1, 1)
        
        hmm_features = np.column_stack([returns, volatility])
        
        # Fit HMM
        model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(hmm_features)
        
        # Predict regimes
        regimes = model.predict(hmm_features)
        df['Regime'] = regimes
        
        return df

    def prepare_data(self, symbol):
        """Full pipeline: download -> indicators -> split."""
        df = self.download_data(symbol)
        df = self.add_indicators(df)
        df = self._add_hmm_regimes(df)
        
        # Time-based split
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size :]
        
        # Features for scaling
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                        'EMA_20', 'EMA_50', 'ATR', 'Regime']
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])
        test_scaled = scaler.transform(test_df[feature_cols])
        
        self.scalers[symbol] = scaler
        
        return {
            'train': (train_df, train_scaled),
            'val': (val_df, val_scaled),
            'test': (test_df, test_scaled),
            'feature_cols': feature_cols
        }

if __name__ == "__main__":
    loader = DataLoader(['AAPL'], '2020-01-01', '2023-12-31')
    data = loader.prepare_data('AAPL')
    print(f"Train shape: {data['train'][1].shape}")
    print(f"Features: {data['feature_cols']}")
