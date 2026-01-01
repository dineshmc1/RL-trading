import yfinance as yf
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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
        
        # Log Returns and Std Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility (Realized Vol)
        df['Realized_Vol'] = df['Log_Returns'].rolling(window=20).std()
        
        # VWAP Distance
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['VWAP']
        
        # Trend Slope (20-day linear regression slope)
        def get_slope(array):
            y = array
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            return model.coef_[0]
            
        df['Trend_Slope'] = df['Close'].rolling(window=20).apply(get_slope, raw=True)
        
        # --- Advanced Features ---
        
        # 1. Funding Proxy (Momentum Skew): (Close - MA_Short) / MA_Short
        # Using EMA_20 as proxy for "Short Term MA"
        df['Funding_Proxy'] = (df['Close'] - df['EMA_20']) / (df['EMA_20'] + 1e-6)
        
        # 2. Volatility of Volatility
        df['Vol_of_Vol'] = df['Realized_Vol'].rolling(window=20).std()
        
        # 3. Time Encoding (Cyclical)
        # Assuming Daily Data from Yahoo Finance
        # Day of Week (0=Monday, 6=Sunday)
        day_of_week = df.index.dayofweek
        df['Day_Cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
        df['Day_Sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
        
        # Day of Year (Seasonality)
        day_of_year = df.index.dayofyear
        df['Year_Cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        df['Year_Sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Drop rows with NaN from indicators
        df.dropna(inplace=True)
        return df

    def _add_hmm_regimes(self, df, train_df, n_regimes=3):
        """Add market regimes using Hidden Markov Model, trained on train_df."""
        Logger.info(f"Fitting HMM on training data ({len(train_df)} points)...")
        
        def extract_features(data):
            # Same features as before: Returns and Realized Vol
            rets = data['Log_Returns'].values.reshape(-1, 1)
            vol = data['Realized_Vol'].values.reshape(-1, 1)
            return np.column_stack([rets, vol])
            
        train_features = extract_features(train_df)
        all_features = extract_features(df)
        
        # Fit HMM on training data only
        model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(train_features)
        
        # Probabilistic output
        probs = model.predict_proba(all_features)
        for i in range(n_regimes):
            df[f'Regime_Prob_{i}'] = probs[:, i]
            
        # Regime ID for duration/vol/trend calculations
        regimes = model.predict(all_features)
        df['Regime'] = regimes
        
        # Regime Duration
        df['Regime_Duration'] = df.groupby((df['Regime'] != df['Regime'].shift()).cumsum()).cumcount() + 1
        
        # Regime Specific Stats (for expansion)
        # Note: These are slightly redundant with the probabilities but requested
        for i in range(n_regimes):
            # Example: Mean vol for this regime state
            state_vol = np.sqrt(model.covars_[i][1,1]) # Index 1 is Realized_Vol
            state_trend = model.means_[i][0] # Index 0 is Log_Returns
            df.loc[df['Regime'] == i, 'Regime_Vol_State'] = state_vol
            df.loc[df['Regime'] == i, 'Regime_Trend_State'] = state_trend

        return df, model

    def prepare_data(self, symbol):
        """Full pipeline: download -> indicators -> split."""
        df = self.download_data(symbol)
        df = self.add_indicators(df)
        
        # Initial drop for indicator windows
        df.dropna(inplace=True)
        
        # Time-based split for HMM training
        train_size = int(len(df) * 0.7)
        train_df_for_hmm = df.iloc[:train_size]
        
        # Fit HMM on train data only
        df, hmm_model = self._add_hmm_regimes(df, train_df_for_hmm)
        
        # Second drop for HMM or other rolling features
        df.dropna(inplace=True)
        
        # Re-split after indicator/hmm addition
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size :]
        
        # Final Feature List as requested
        feature_cols = [
            'Log_Returns', 'Realized_Vol', 'RSI', 'MACD', 
            'VWAP_Dist', 'Trend_Slope', 
            'Regime_Prob_0', 'Regime_Prob_1', 'Regime_Prob_2',
            'Funding_Proxy', 'Vol_of_Vol', 
            'Day_Cos', 'Day_Sin', 'Year_Cos', 'Year_Sin'
        ]
        
        # Check for any missing columns (e.g. if n_regimes != 3)
        feature_cols = [c for c in feature_cols if c in df.columns]
        
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
