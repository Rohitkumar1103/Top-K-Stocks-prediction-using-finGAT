from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import os
import json
from models.fingat import FinGAT
from models.data_processor import StockDataProcessor, DataProcessor
from utils.helpers import get_top_k_stocks, create_adjacency_matrices

app = Flask(__name__)
CORS(app)

# Load model and data processor
data_processor = StockDataProcessor()
stock_processor = DataProcessor()

# Define stock sectors (example)
sectors = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
    'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'UNH', 'ABBV'],
    'Consumer': ['AMZN', 'WMT', 'PG', 'KO', 'PEP'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
}

# Create stock_to_sector mapping
stock_to_sector = {}
sector_names = list(sectors.keys())
for sector_idx, sector_name in enumerate(sector_names):
    for stock in sectors[sector_name]:
        stock_to_sector[stock] = sector_idx

# Initialize model
input_dim = 21  # Match the input dimension of the saved model
hidden_dim = 16
embedding_dim = 16
num_stocks = sum(len(stocks) for stocks in sectors.values())
num_sectors = len(sectors)

model = FinGAT(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
    num_stocks=num_stocks,
    num_sectors=num_sectors
)

# Get the current directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load model weights if available - looking in the models folder in the same directory as app.py
model_path = os.path.join(current_dir, 'models', 'fingat_model.pth')
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print(f"Model not found at {model_path}")

@app.route('/api/top-stocks', methods=['POST'])
def get_top_stocks():
    try:
        data = request.json
        
        # Get data from request or use sample data
        if 'data' in data and len(data['data']) > 0:
            df = pd.DataFrame(data['data'])
        else:
            # Load sample data from the same directory as app.py
            sample_data_path = os.path.join(current_dir, 'sample_data.csv')
            if not os.path.exists(sample_data_path):
                # Try looking in a data subdirectory
                sample_data_path = os.path.join(current_dir, 'data', 'sample_data.csv')
            
            if not os.path.exists(sample_data_path):
                return jsonify({'error': f'Sample data not found at {sample_data_path}'}), 404
                
            df = pd.read_csv(sample_data_path)
        
        # Preprocess data
        df = data_processor.preprocess_data(df)
        
        # Create sequences
        X, _ = data_processor.create_sequences(df)
        
        # Check if X is empty
        if len(X) == 0:
            return jsonify({'error': 'No valid data sequences could be created'}), 400
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Ensure X_tensor has the correct shape
        if len(X_tensor.shape) == 2:  # [batch, features]
            X_tensor = X_tensor.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, features]
        elif len(X_tensor.shape) == 3:  # [batch, seq_len, features]
            X_tensor = X_tensor.unsqueeze(1)  # [batch, 1, seq_len, features]
        
        # Create adjacency matrices
        intra_sector_adj, inter_sector_adj = create_adjacency_matrices(stock_to_sector, num_sectors)
        
        # Convert adjacency matrices to tensors
        intra_sector_adj_tensors = [torch.tensor(adj, dtype=torch.float32) for adj in intra_sector_adj]
        inter_sector_adj_tensor = torch.tensor(inter_sector_adj, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            return_predictions, movement_predictions, _ = model(
                X_tensor, 
                stock_to_sector,
                intra_sector_adj_tensors,
                inter_sector_adj_tensor
            )
        
        # Get stock names (all stocks from all sectors)
        stock_names = []
        for sector in sectors:
            stock_names.extend(sectors[sector])
        
        # Get top-5 stocks
        top_stocks = get_top_k_stocks(return_predictions, stock_names, k=5)
        
        # Format response
        response = {
            'top_stocks': [
                {
                    'symbol': stock,
                    'predicted_return': float(return_val),
                    'sector': sector_names[stock_to_sector[stock]] if stock in stock_to_sector else 'Unknown'
                }
                for stock, return_val in top_stocks
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/api/test', methods=['GET', 'POST'])
def test():
    """Simple test endpoint to verify the API is working"""
    if request.method == 'POST':
        return jsonify({"message": "POST request received successfully"})
    return jsonify({"message": "GET request received successfully"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
