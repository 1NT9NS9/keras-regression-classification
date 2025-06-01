"""
Stock Market Data Processing Module

This module handles data collection, preprocessing, and feature engineering
for stock market analysis using neural networks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class StockDataProcessor:
    """
    A comprehensive class for processing stock market data for ML models.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def get_stock_data(self, symbol, period='2y', interval='1d'):
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        Calculate various technical indicators.
        
        Args:
            df (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with technical indicators added
        """
        data = df.copy()
        
        # Simple Moving Averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
        
        # RSI
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        data['BB_position'] = (data['Close'] - data['BB_lower']) / data['BB_width']
        
        # Price change indicators
        data['Price_change'] = data['Close'].pct_change()
        data['Price_change_1d'] = data['Close'].shift(-1) / data['Close'] - 1
        data['High_Low_ratio'] = data['High'] / data['Low']
        data['Open_Close_ratio'] = data['Open'] / data['Close']
        
        # Volume indicators
        data['Volume_SMA_10'] = data['Volume'].rolling(window=10).mean()
        data['Volume_ratio'] = data['Volume'] / data['Volume_SMA_10']
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=10).std()
        
        return data
    
    def calculate_rsi(self, prices, window=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Price series
            window (int): RSI calculation window
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_features_for_regression(self, df, target_column='Close', lookback_window=10):
        """
        Create features for regression task (price prediction).
        
        Args:
            df (pd.DataFrame): Data with technical indicators
            target_column (str): Column to predict
            lookback_window (int): Number of previous days to use as features
            
        Returns:
            tuple: (X, y) features and targets
        """
        data = df.copy()
        
        # Select feature columns
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
            'Price_change', 'High_Low_ratio', 'Open_Close_ratio',
            'Volume_SMA_10', 'Volume_ratio', 'Volatility'
        ]
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Create sequences for time series prediction
        X, y = [], []
        
        for i in range(lookback_window, len(data)):
            # Features: lookback_window days of data
            X.append(data[feature_cols].iloc[i-lookback_window:i].values.flatten())
            # Target: next day's closing price
            y.append(data[target_column].iloc[i])
        
        self.feature_columns = [f"{col}_{j}" for j in range(lookback_window) for col in feature_cols]
        
        return np.array(X), np.array(y)
    
    def create_features_for_classification(self, df, lookback_window=10, threshold=0.02):
        """
        Create features for classification task (direction prediction).
        
        Args:
            df (pd.DataFrame): Data with technical indicators
            lookback_window (int): Number of previous days to use as features
            threshold (float): Threshold for price movement classification
            
        Returns:
            tuple: (X, y) features and targets
        """
        data = df.copy()
        
        # Calculate next day price change
        data['Next_day_change'] = data['Close'].shift(-1) / data['Close'] - 1
        
        # Create target classes: 0=Down, 1=Neutral, 2=Up
        def classify_movement(change):
            if pd.isna(change):
                return np.nan
            elif change > threshold:
                return 2  # Up
            elif change < -threshold:
                return 0  # Down
            else:
                return 1  # Neutral
        
        data['Direction'] = data['Next_day_change'].apply(classify_movement)
        
        # Select feature columns
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
            'Price_change', 'High_Low_ratio', 'Open_Close_ratio',
            'Volume_SMA_10', 'Volume_ratio', 'Volatility'
        ]
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Create sequences for time series classification
        X, y = [], []
        
        for i in range(lookback_window, len(data)):
            # Features: lookback_window days of data
            X.append(data[feature_cols].iloc[i-lookback_window:i].values.flatten())
            # Target: direction class
            y.append(data['Direction'].iloc[i])
        
        self.feature_columns = [f"{col}_{j}" for j in range(lookback_window) for col in feature_cols]
        
        return np.array(X), np.array(y)
    
    def prepare_data_for_training(self, X, y, test_size=0.2, random_state=42, scale_features=True):
        """
        Prepare data for training by splitting and scaling.
        
        Args:
            X (np.array): Features
            y (np.array): Targets
            test_size (float): Test set proportion
            random_state (int): Random seed
            scale_features (bool): Whether to scale features
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        if scale_features:
            # Fit scaler on training data only
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        """
        Get the names of the created features.
        
        Returns:
            list: Feature names
        """
        return self.feature_columns


# Example usage and testing
if __name__ == "__main__":
    processor = StockDataProcessor()
    
    # Fetch Apple stock data
    print("Fetching AAPL stock data...")
    data = processor.get_stock_data('AAPL', period='2y')
    
    if data is not None:
        print(f"Data shape: {data.shape}")
        print("Adding technical indicators...")
        
        # Add technical indicators
        data_with_indicators = processor.calculate_technical_indicators(data)
        print(f"Data with indicators shape: {data_with_indicators.shape}")
        
        # Create regression features
        print("Creating regression features...")
        X_reg, y_reg = processor.create_features_for_regression(data_with_indicators)
        print(f"Regression features shape: {X_reg.shape}, targets shape: {y_reg.shape}")
        
        # Create classification features
        print("Creating classification features...")
        X_clf, y_clf = processor.create_features_for_classification(data_with_indicators)
        print(f"Classification features shape: {X_clf.shape}, targets shape: {y_clf.shape}")
        
        print("Data processing completed successfully!") 