import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime

def create_adjacency_matrices(stock_to_sector, num_sectors):
    """
    Create intra-sector and inter-sector adjacency matrices.
    
    Args:
        stock_to_sector: Dictionary mapping stock indices to sector indices
        num_sectors: Number of sectors
        
    Returns:
        intra_sector_adj: List of adjacency matrices for each sector
        inter_sector_adj: Adjacency matrix for sectors
    """
    # Group stocks by sector
    sector_to_stocks = {}
    for stock_idx, sector_idx in stock_to_sector.items():
        if sector_idx not in sector_to_stocks:
            sector_to_stocks[sector_idx] = []
        sector_to_stocks[sector_idx].append(stock_idx)
    
    # Create intra-sector adjacency matrices
    intra_sector_adj = []
    for sector_idx in range(num_sectors):
        if sector_idx in sector_to_stocks:
            stocks = sector_to_stocks[sector_idx]
            num_stocks = len(stocks)
            adj = np.ones((num_stocks, num_stocks))
            intra_sector_adj.append(adj)
        else:
            intra_sector_adj.append(np.array([]))
    
    # Create inter-sector adjacency matrix
    inter_sector_adj = np.ones((num_sectors, num_sectors))
    
    return intra_sector_adj, inter_sector_adj

def get_top_k_stocks(return_predictions, stock_names, k=5):
    """
    Get the top-K stocks based on predicted return ratios.
    
    Args:
        return_predictions: Tensor of predicted return ratios for each stock
        stock_names: List of stock names/symbols
        k: Number of top stocks to return
        
    Returns:
        List of tuples (stock_name, predicted_return)
    """
    # Convert tensor to numpy if needed
    if isinstance(return_predictions, torch.Tensor):
        return_predictions = return_predictions.detach().cpu().numpy()
    
    # Create list of (stock_name, predicted_return) tuples
    stock_predictions = list(zip(stock_names, return_predictions))
    
    # Sort by predicted return in descending order
    sorted_predictions = sorted(stock_predictions, key=lambda x: x[1], reverse=True)
    
    # Return top-K stocks
    return sorted_predictions[:k]

def calculate_sector_performance(stock_data, stock_to_sector, sector_names):
    """
    Calculate the average performance of each sector.
    
    Args:
        stock_data: Dictionary mapping stock symbols to their return ratios
        stock_to_sector: Dictionary mapping stock symbols to sector indices
        sector_names: List of sector names
        
    Returns:
        Dictionary mapping sector names to their average return ratios
    """
    sector_performance = {sector: [] for sector in sector_names}
    
    # Group stock returns by sector
    for stock, return_ratio in stock_data.items():
        if stock in stock_to_sector:
            sector_idx = stock_to_sector[stock]
            sector_name = sector_names[sector_idx]
            sector_performance[sector_name].append(return_ratio)
    
    # Calculate average return for each sector
    for sector in sector_performance:
        if sector_performance[sector]:
            sector_performance[sector] = sum(sector_performance[sector]) / len(sector_performance[sector])
        else:
            sector_performance[sector] = 0.0
    
    return sector_performance

def format_timestamp(timestamp):
    """Convert timestamp to readable format."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def load_stock_data(file_path):
    """
    Load stock data from CSV file.
    
    Args:
        file_path: Path to CSV file containing stock data
        
    Returns:
        Pandas DataFrame with stock data
    """
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

