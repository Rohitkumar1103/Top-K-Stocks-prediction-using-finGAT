import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, days_per_week=5, weeks_for_training=3):
        self.days_per_week = days_per_week
        self.weeks_for_training = weeks_for_training
        self.feature_scaler = MinMaxScaler()
        
    def process_data(self, data_path):
        """Process stock data from CSV file."""
        # Load data
        df = pd.read_csv(data_path) if isinstance(data_path, str) else data_path
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Extract features
        features = ['open', 'high', 'low', 'close', 'volume', 'VWAP', 'MACD', 'Signal', 
           'MACD_Hist', 'Volatility_Skew', 'Vol_Price_Corr', 'Volume_Momentum',
           'Price_Acceleration', 'Volume_Trend', 'rsi', 'ROC', 'Volume_Time_Variations',
           'mean_reversion', 'Stochastic_K', 'Stochastic_D', 'Kalman_Filter']

        
        # Scale features
        df[features] = self.feature_scaler.fit_transform(df[features])
        
        # Calculate return ratio (target)
        df['return_ratio'] = df['close'].pct_change()
        df['movement'] = (df['return_ratio'] > 0).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def create_sequences(self, df, stock_id=None, sector_id=None):
        """Create sequences for training."""
        total_days = len(df)
        sequences = []
        targets = []
        stock_ids = []
        sector_ids = []
        
        # Calculate number of weeks
        total_weeks = total_days // self.days_per_week
        
        for i in range(total_weeks - self.weeks_for_training):
            # Get data for training weeks
            start_idx = i * self.days_per_week
            end_idx = start_idx + self.days_per_week * self.weeks_for_training
            target_idx = end_idx
            
            if target_idx >= total_days:
                break
                
            # Extract features for sequence
            sequence = []
            for week in range(self.weeks_for_training):
                week_start = start_idx + week * self.days_per_week
                week_end = week_start + self.days_per_week
                week_data = df.iloc[week_start:week_end]
                
                # Skip if we don't have enough days for this week
                if len(week_data) < self.days_per_week:
                    continue
                    
                features = week_data[['open', 'high', 'low', 'close', 'volume', 'VWAP', 'MACD', 'Signal', 
                                     'MACD_Hist', 'Volatility_Skew', 'Vol_Price_Corr', 'Volume_Momentum',
                                     'Price_Acceleration', 'Volume_Trend', 'rsi', 'ROC', 'Volume_Time_Variations',
                                     'mean_reversion', 'Stochastic_K', 'Stochastic_D', 'Kalman_Filter']].values
                sequence.append(features)
            
            # Skip if we don't have data for all weeks
            if len(sequence) < self.weeks_for_training:
                continue
                
            # Get target
            target_return = df.iloc[target_idx]['return_ratio']
            target_movement = df.iloc[target_idx]['movement']
            
            sequences.append(np.array(sequence))
            targets.append([target_return, target_movement])
            
            if stock_id is not None:
                stock_ids.append(stock_id)
            if sector_id is not None:
                sector_ids.append(sector_id)
        
        return {
            'sequences': np.array(sequences),
            'targets': np.array(targets),
            'stock_ids': np.array(stock_ids) if stock_id is not None else None,
            'sector_ids': np.array(sector_ids) if sector_id is not None else None
        }
    
    def prepare_batch(self, stock_data_dict, batch_indices):
        """Prepare batch for training or inference."""
        batch_sequences = []
        batch_targets = []
        batch_stock_ids = []
        batch_sector_ids = []
        
        for idx in batch_indices:
            batch_sequences.append(stock_data_dict['sequences'][idx])
            batch_targets.append(stock_data_dict['targets'][idx])
            
            if stock_data_dict['stock_ids'] is not None:
                batch_stock_ids.append(stock_data_dict['stock_ids'][idx])
            if stock_data_dict['sector_ids'] is not None:
                batch_sector_ids.append(stock_data_dict['sector_ids'][idx])
        
        return {
            'sequences': torch.tensor(np.array(batch_sequences), dtype=torch.float32),
            'targets': torch.tensor(np.array(batch_targets), dtype=torch.float32),
            'stock_ids': torch.tensor(np.array(batch_stock_ids), dtype=torch.long) if batch_stock_ids else None,
            'sector_ids': torch.tensor(np.array(batch_sector_ids), dtype=torch.long) if batch_sector_ids else None
        }
class StockDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, df):
        """
        Preprocess the stock data.
        
        Args:
            df: Pandas DataFrame with stock data
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert date to datetime if not already
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Extract hour and day features if they don't exist
        if 'Hour_of_Day' not in df.columns and 'date' in df.columns:
            df['Hour_of_Day'] = df['date'].dt.hour
        
        if 'Day_of_Month' not in df.columns and 'date' in df.columns:
            df['Day_of_Month'] = df['date'].dt.day
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
        
        # Normalize numerical features
        features_to_scale = ['open', 'high', 'low', 'close', 'volume', 'VWAP', 
                            'MACD', 'Signal', 'MACD_Hist', 'Volatility_Skew', 
                            'Vol_Price_Corr', 'Volume_Momentum', 'Price_Acceleration', 
                            'Volume_Trend', 'rsi', 'ROC', 'Volume_Time_Variations',
                            'mean_reversion', 'Stochastic_K', 'Stochastic_D', 'Kalman_Filter']
        
        # Only scale features that exist in the DataFrame
        features_to_scale = [f for f in features_to_scale if f in df.columns]
        
        if features_to_scale:
            df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
        
        return df
    
    def create_sequences(self, df, sequence_length=6):
        """
        Create sequences for time series prediction.
        
        Args:
            df: Preprocessed DataFrame
            sequence_length: Length of sequences to create
            
        Returns:
            X: Input sequences
            y: Target values (next day's close price)
        """
        data = df.copy()
        
        # Sort by date if available
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        # Select features for input
        features = ['open', 'high', 'low', 'close', 'volume', 'VWAP', 
               'MACD', 'Signal', 'MACD_Hist', 'Volatility_Skew', 
               'Vol_Price_Corr', 'Volume_Momentum', 'Price_Acceleration', 
               'Volume_Trend', 'rsi', 'ROC', 'Volume_Time_Variations',
               'mean_reversion', 'Stochastic_K', 'Stochastic_D', 'Kalman_Filter']
        
        # Only use features that exist in the DataFrame
        features = [f for f in features if f in data.columns]
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[features].iloc[i:i+sequence_length].values)
            y.append(data['close'].iloc[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    def calculate_return_ratio(self, df):
        """
        Calculate return ratio for each timestamp.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with added return_ratio column
        """
        if 'close' in df.columns:
            df['return_ratio'] = df['close'].pct_change()
            df['return_ratio'] = df['return_ratio'].fillna(0)
        
        return df
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        n = len(X)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
