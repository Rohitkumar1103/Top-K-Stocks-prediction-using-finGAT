import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from models.fingat import FinGAT
from models.data_processor import DataProcessor
from utils.helpers import create_adjacency_matrices
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define parameters
input_dim = 21  # Number of features
hidden_dim = 16
embedding_dim = 16
num_epochs = 100
batch_size = 32
learning_rate = 0.001
delta = 0.01  # Balance parameter for multi-task learning
lambda_reg = 0.0001  # L2 regularization parameter

# Define sectors and stocks
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
num_stocks = sum(len(stocks) for stocks in sectors.values())
num_sectors = len(sectors)

model = FinGAT(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
    num_stocks=num_stocks,
    num_sectors=num_sectors
)

# Load and preprocess data
# Modified to look for the CSV file in the same directory as this script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'sample_data.csv')

# Check if file exists, otherwise try to find it in a data subdirectory
if not os.path.exists(data_path):
    data_path = os.path.join(current_dir, 'data', 'sample_data.csv')
    
    # If still not found, print available files in the current directory
    if not os.path.exists(data_path):
        print(f"Could not find sample_data.csv in {current_dir} or {os.path.join(current_dir, 'data')}")
        print("Files in current directory:")
        for file in os.listdir(current_dir):
            print(f"  - {file}")
        if os.path.exists(os.path.join(current_dir, 'data')):
            print("Files in data directory:")
            for file in os.listdir(os.path.join(current_dir, 'data')):
                print(f"  - {file}")
        raise FileNotFoundError(f"Could not find sample_data.csv")

print(f"Loading data from: {data_path}")
data_processor = DataProcessor()
stock_processor = DataProcessor(days_per_week=5, weeks_for_training=3)

df = pd.read_csv(data_path)
df = data_processor.process_data(df)

# Create sequences for training
stock_data_dict = stock_processor.create_sequences(df)

# Create adjacency matrices
intra_sector_adj, inter_sector_adj = create_adjacency_matrices(stock_to_sector, num_sectors)

# Convert adjacency matrices to tensors
intra_sector_adj_tensors = [torch.tensor(adj, dtype=torch.float32) for adj in intra_sector_adj]
inter_sector_adj_tensor = torch.tensor(inter_sector_adj, dtype=torch.float32)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)

# Training loop
losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Create random batches
    indices = np.random.permutation(len(stock_data_dict['sequences']))
    num_batches = len(indices) // batch_size
    
    # Handle case where there are fewer samples than batch_size
    if num_batches == 0 and len(indices) > 0:
        num_batches = 1
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        
        # Prepare batch
        batch = stock_processor.prepare_batch(stock_data_dict, batch_indices)
        
        # Forward pass
        return_predictions, movement_predictions, _ = model(
            batch['sequences'],
            stock_to_sector,
            intra_sector_adj_tensors,
            inter_sector_adj_tensor
        )
        
        # Calculate ranking loss
        ranking_loss = 0.0
        for i in range(len(return_predictions)):
            for j in range(len(return_predictions)):
                if i != j:
                    pred_diff = return_predictions[i] - return_predictions[j]
                    true_diff = batch['targets'][i, 0] - batch['targets'][j, 0]
                    ranking_loss += max(0, -pred_diff * true_diff)
        
        # Calculate movement prediction loss
        movement_targets = (batch['targets'][:, 0] > 0).float()
        movement_loss = torch.nn.functional.binary_cross_entropy_with_logits(
    movement_predictions,  # Raw logits, no sigmoid needed
    movement_targets.unsqueeze(1)
)


        
        # Total loss
        loss = (1 - delta) * ranking_loss + delta * movement_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Print epoch statistics
    avg_epoch_loss = epoch_loss / max(num_batches, 1)  # Avoid division by zero
    losses.append(avg_epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')
    
    # Validate on a small subset
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            # Handle case where there are fewer samples than 100
            val_size = min(100, len(stock_data_dict['sequences']))
            if val_size > 0:
                val_indices = np.random.choice(len(stock_data_dict['sequences']), val_size, replace=False)
                val_batch = stock_processor.prepare_batch(stock_data_dict, val_indices)
                
                return_predictions, movement_predictions, _ = model(
                    val_batch['sequences'],
                    stock_to_sector,
                    intra_sector_adj_tensors,
                    inter_sector_adj_tensor
                )
                
                # Calculate validation metrics
                val_ranking_loss = 0.0
                for i in range(len(return_predictions)):
                    for j in range(len(return_predictions)):
                        if i != j:
                            pred_diff = return_predictions[i] - return_predictions[j]
                            true_diff = val_batch['targets'][i, 0] - val_batch['targets'][j, 0]
                            val_ranking_loss += max(0, -pred_diff * true_diff)
                
                val_movement_targets = (val_batch['targets'][:, 0] > 0).float()
                val_movement_loss = torch.nn.functional.binary_cross_entropy_with_logits(
    movement_predictions,  # Raw logits, no sigmoid needed
    val_movement_targets.unsqueeze(1)
)
                
                val_loss = (1 - delta) * val_ranking_loss + delta * val_movement_loss
                print(f'Validation Loss: {float(val_loss):.4f}')

# Create models directory if it doesn't exist
models_dir = os.path.join(current_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save the trained model
model_path = os.path.join(models_dir, 'fingat_model.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(current_dir, 'training_loss.png'))
plt.close()

print('Training completed!')
