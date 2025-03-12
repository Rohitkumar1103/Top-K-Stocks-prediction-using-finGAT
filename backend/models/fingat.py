import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUAttention, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        outputs, _ = self.gru(x)
        # outputs shape: (batch_size, seq_len, hidden_dim)
        
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(outputs), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # Apply attention weights to outputs
        context = torch.sum(outputs * attention_weights, dim=1)
        # context shape: (batch_size, hidden_dim)
        
        return context, attention_weights

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Simple case for empty adjacency matrix
        if adj.numel() == 0:
            return torch.zeros_like(h)
        
        # Fix dimension mismatch - improved handling
        if h.shape[1] != self.in_features:
            print(f"Warning: Input features dimension mismatch. Got {h.shape[1]}, expected {self.in_features}")
            # Create a properly sized tensor
            new_h = torch.zeros(h.shape[0], self.in_features, device=h.device)
            # Copy as many values as possible
            min_features = min(h.shape[1], self.in_features)
            new_h[:, :min_features] = h[:, :min_features]
            h = new_h
            
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        # Check and adjust dimensions if needed
        if e.shape[1] != adj.shape[1]:
            # Create appropriate sized adjacency matrix
            adj = torch.ones((e.shape[0], e.shape[1]), device=e.device)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class FinGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_stocks, num_sectors, dropout=0.6, alpha=0.2):
        super(FinGAT, self).__init__()
        
        # Short-term sequential learning
        self.gru_attention = GRUAttention(input_dim, hidden_dim)
        
        # Intra-sector relation modeling
        self.intra_sector_gat = GraphAttentionLayer(hidden_dim, embedding_dim, dropout, alpha)
        
        # Long-term sequential learning
        self.long_term_gru_g = GRUAttention(embedding_dim, embedding_dim)
        self.long_term_gru_a = GRUAttention(hidden_dim, embedding_dim)
        
        # Inter-sector relation modeling
        self.inter_sector_gat = GraphAttentionLayer(embedding_dim, embedding_dim, dropout, alpha)
        
        # Fusion layer
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)
        
        # Task-specific layers
        self.return_predictor = nn.Linear(embedding_dim, 1)
        self.movement_predictor = nn.Linear(embedding_dim, 1)
        
        # Store model parameters
        self.num_stocks = num_stocks
        self.num_sectors = num_sectors
        
    def forward(self, stock_features, stock_to_sector, intra_sector_adj, inter_sector_adj):
        # Modified to handle different input shapes
        # Print shape for debugging
        print(f"Input stock_features shape: {stock_features.shape}")
        
        # Handle different input shapes
        if len(stock_features.shape) == 4:  # [batch_size, seq_len, days, features]
            batch_size, seq_len, days_per_week, feature_dim = stock_features.shape
            # Reshape to expected format or adapt processing
            stock_features = stock_features.unsqueeze(1)  # Add stocks dimension
            print(f"Reshaped to: {stock_features.shape}")
            num_stocks = 1
            num_weeks = seq_len
        elif len(stock_features.shape) == 5:  # Already in expected format
            batch_size, num_stocks, num_weeks, days_per_week, feature_dim = stock_features.shape
        else:
            raise ValueError(f"Unexpected input shape: {stock_features.shape}. Expected 4D or 5D tensor.")
        
        # Process each stock
        short_term_embeddings = []
        intra_sector_embeddings = []
        
        for stock_idx in range(num_stocks):
            # Short-term sequential learning for each week
            stock_weekly_embeddings = []
            
            for week_idx in range(num_weeks):
                # Get daily features for current stock and week
                daily_features = stock_features[:, stock_idx, week_idx, :, :]
                
                # Apply GRU with attention
                short_term_emb, _ = self.gru_attention(daily_features)
                stock_weekly_embeddings.append(short_term_emb)
            
            # Stack weekly embeddings
            stock_weekly_embeddings = torch.stack(stock_weekly_embeddings, dim=1)
            short_term_embeddings.append(stock_weekly_embeddings)
        
        short_term_embeddings = torch.stack(short_term_embeddings, dim=1)
        
        # Intra-sector relation modeling
        for sector_idx in range(len(intra_sector_adj)):
            # Find stocks in this sector
            sector_stocks = []
            if isinstance(stock_to_sector, dict):
                # Handle dictionary mapping
                stock_to_idx = {stock: idx for idx, stock in enumerate(stock_to_sector.keys())}
                sector_stocks = [stock_to_idx[i] for i, s_idx in stock_to_sector.items() 
                                if s_idx == sector_idx]
            else:
                # Handle other formats (list, tensor, etc.)
                sector_stocks = [i for i in range(num_stocks) 
                                if i < len(stock_to_sector) and stock_to_sector[i] == sector_idx]
            
            if len(sector_stocks) > 0:
                # Get embeddings for stocks in this sector
                # Check if sector_stocks contains valid indices
                # Ensure sector_stocks only contains valid indices
                valid_sector_stocks = [idx for idx in sector_stocks if idx < short_term_embeddings.shape[1]]
                if not valid_sector_stocks:
                    valid_sector_stocks = [0]  # Default to first index if none are valid
                
                # Use valid indices for indexing
                sector_stock_embeddings = short_term_embeddings[:, valid_sector_stocks, -1, :]
                # Reshape for GAT
                sector_stock_embeddings = sector_stock_embeddings.reshape(-1, sector_stock_embeddings.shape[-1])
                
                # Apply GAT
                sector_adj = intra_sector_adj[sector_idx]
                if isinstance(sector_adj, torch.Tensor) and sector_adj.numel() > 0:
                    if len(sector_stocks) <= sector_adj.shape[0]:
                        sector_adj = sector_adj[:len(sector_stocks), :len(sector_stocks)]
                    else:
                        # Create appropriate sized adjacency matrix if needed
                        sector_adj = torch.ones((len(sector_stocks), len(sector_stocks)), 
                                               device=sector_stock_embeddings.device)
                else:
                    # Create default adjacency matrix
                    sector_adj = torch.ones((len(sector_stocks), len(sector_stocks)), 
                                           device=sector_stock_embeddings.device)
                
                sector_gat_output = self.intra_sector_gat(sector_stock_embeddings, sector_adj)
                
                # Store embeddings
                for i, stock_idx in enumerate(sector_stocks):
                    if i < sector_gat_output.shape[0]:
                        intra_sector_embeddings.append(sector_gat_output[i:i+1])
        
        # Handle case where intra_sector_embeddings is empty
        if not intra_sector_embeddings:
            # Create default embeddings
            default_emb = torch.zeros((batch_size, self.intra_sector_gat.out_features), device=stock_features.device)
            for _ in range(num_stocks):
                intra_sector_embeddings.append(default_emb)
        
        # Fix for reshape error - Print size before reshape
        print(f"intra_sector_embeddings size before reshape: {len(intra_sector_embeddings)}, tensor size: {intra_sector_embeddings[0].shape}")
        
        # Concatenate all embeddings
        intra_sector_embeddings = torch.cat(intra_sector_embeddings, dim=0)
        print(f"intra_sector_embeddings after cat: {intra_sector_embeddings.shape}")
        
        # Calculate the embedding dimension
        embedding_dim = self.intra_sector_gat.out_features
        
        # Create properly sized tensor for reshaping
        intra_sector_embeddings_padded = torch.zeros(batch_size * num_stocks, embedding_dim, 
                                                   device=intra_sector_embeddings.device)
        
        # Copy as many values as possible
        actual_size = min(intra_sector_embeddings.size(0), batch_size * num_stocks)
        intra_sector_embeddings_padded[:actual_size] = intra_sector_embeddings[:actual_size]
        
        # Reshape with explicit dimensions
        intra_sector_embeddings = intra_sector_embeddings_padded.reshape(batch_size, num_stocks, embedding_dim)
        print(f"intra_sector_embeddings after reshape: {intra_sector_embeddings.shape}")
        
        # Long-term sequential learning
        long_term_g_embeddings = []
        long_term_a_embeddings = []
        
        for stock_idx in range(num_stocks):
            # Get sequences of embeddings for current stock
            g_sequence = intra_sector_embeddings[:, stock_idx, :]
            a_sequence = short_term_embeddings[:, stock_idx, :, :]
            
            # Apply GRU with attention for long-term learning
            long_term_g, _ = self.long_term_gru_g(g_sequence.unsqueeze(1))
            long_term_a, _ = self.long_term_gru_a(a_sequence)
            
            long_term_g_embeddings.append(long_term_g)
            long_term_a_embeddings.append(long_term_a)
        
        # Concatenate and ensure proper shape for both embeddings
        long_term_g_embeddings = torch.cat(long_term_g_embeddings, dim=1)
        long_term_a_embeddings = torch.cat(long_term_a_embeddings, dim=1)
        
        # Print shape for debugging
        print(f"long_term_g_embeddings shape: {long_term_g_embeddings.shape}")
        print(f"long_term_a_embeddings shape: {long_term_a_embeddings.shape}")
        
        # Ensure both embeddings have 3 dimensions [batch, stocks, features]
        if len(long_term_g_embeddings.shape) == 2:
            # Reshape to [batch_size, num_stocks, embedding_dim]
            long_term_g_embeddings = long_term_g_embeddings.unsqueeze(1)
            print(f"Reshaped long_term_g_embeddings: {long_term_g_embeddings.shape}")
        
        if len(long_term_a_embeddings.shape) == 2:
            # Reshape to [batch_size, num_stocks, embedding_dim]
            long_term_a_embeddings = long_term_a_embeddings.unsqueeze(1)
            print(f"Reshaped long_term_a_embeddings: {long_term_a_embeddings.shape}")
        
        # Sector-level modeling
        sector_embeddings = []
        
        for sector_idx in range(self.num_sectors):
            # Find stocks in this sector
            sector_stocks = []
            if isinstance(stock_to_sector, dict):
                # Create a mapping from stock symbols to indices if needed
                if isinstance(next(iter(stock_to_sector.keys()), None), str):
                    stock_to_idx = {stock: idx for idx, stock in enumerate(stock_to_sector.keys())}
                    sector_stocks = [stock_to_idx[i] for i, s_idx in stock_to_sector.items() 
                                    if s_idx == sector_idx]
                else:
                    sector_stocks = [i for i, s_idx in stock_to_sector.items() 
                                    if s_idx == sector_idx]
            else:
                sector_stocks = [i for i in range(num_stocks) 
                                if i < len(stock_to_sector) and stock_to_sector[i] == sector_idx]
            
            # Filter valid indices
            valid_sector_stocks = [idx for idx in sector_stocks if idx < long_term_g_embeddings.shape[1]]
            
            if valid_sector_stocks:
                # Modified indexing based on tensor dimensions
                if len(long_term_g_embeddings.shape) == 3:
                    # Original 3D tensor case
                    sector_stock_g_embeddings = long_term_g_embeddings[:, valid_sector_stocks, :]
                else:
                    # Handle 2D tensor case (should not happen after our fix above)
                    sector_stock_g_embeddings = long_term_g_embeddings.unsqueeze(1)
                
                # Max pooling to get sector embedding
                sector_embedding = torch.max(sector_stock_g_embeddings, dim=1)[0]
                sector_embeddings.append(sector_embedding.unsqueeze(1))
            else:
                # No valid stocks in this sector, create default embedding
                default_emb = torch.zeros((batch_size, 1, self.long_term_gru_g.gru.hidden_size), 
                                         device=long_term_g_embeddings.device)
                sector_embeddings.append(default_emb)
        
        # Handle case where sector_embeddings is empty
        if not sector_embeddings:
            # Create default sector embeddings
            default_sector_emb = torch.zeros((batch_size, 1, self.long_term_gru_g.gru.hidden_size), 
                                           device=stock_features.device)
            for _ in range(self.num_sectors):
                sector_embeddings.append(default_sector_emb)
        
        sector_embeddings = torch.cat(sector_embeddings, dim=1)
        
        # Inter-sector relation modeling
        # Reshape for GAT
        sector_emb_flat = sector_embeddings.reshape(-1, sector_embeddings.shape[-1])
        
        # Ensure sector_emb_flat has the correct feature dimension
        if sector_emb_flat.shape[1] != self.inter_sector_gat.in_features:
            print(f"Warning: Sector embedding dimension mismatch. Got {sector_emb_flat.shape[1]}, expected {self.inter_sector_gat.in_features}")
            # Create a properly sized tensor
            new_emb = torch.zeros(sector_emb_flat.shape[0], self.inter_sector_gat.in_features, device=sector_emb_flat.device)
            # Copy as many values as possible
            min_features = min(sector_emb_flat.shape[1], self.inter_sector_gat.in_features)
            new_emb[:, :min_features] = sector_emb_flat[:, :min_features]
            sector_emb_flat = new_emb
        
        # Ensure inter_sector_adj has correct shape
        if isinstance(inter_sector_adj, torch.Tensor):
            if inter_sector_adj.shape[0] != self.num_sectors:
                # Create appropriate sized adjacency matrix
                inter_sector_adj = torch.ones((self.num_sectors, self.num_sectors), 
                                             device=sector_emb_flat.device)
        else:
            # Create default adjacency matrix
            inter_sector_adj = torch.ones((self.num_sectors, self.num_sectors), 
                                         device=sector_emb_flat.device)
        
        # Fix for GAT shape mismatch
        if sector_emb_flat.shape[0] != inter_sector_adj.shape[0]:
            # Adjust shapes to be compatible
            min_size = min(sector_emb_flat.shape[0], inter_sector_adj.shape[0])
            sector_emb_flat = sector_emb_flat[:min_size]
            inter_sector_adj = inter_sector_adj[:min_size, :min_size]
        
        inter_sector_output = self.inter_sector_gat(sector_emb_flat, inter_sector_adj)
        
        # Reshape to match batch size
                # Reshape to match batch size
        if inter_sector_output.shape[0] != batch_size * self.num_sectors:
            # Repeat output for each batch
            inter_sector_output = inter_sector_output.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            inter_sector_output = inter_sector_output.reshape(batch_size, self.num_sectors, -1)
        
        # Fusion and prediction
        final_embeddings = []
        return_predictions = []
        movement_predictions = []
        
        for stock_idx in range(num_stocks):
            # Get sector for this stock
            sector_idx = 0  # Default to first sector
            
            if isinstance(stock_to_sector, dict):
                # Handle string keys if needed
                if isinstance(next(iter(stock_to_sector.keys()), None), str):
                    stock_to_idx = {stock: idx for idx, stock in enumerate(stock_to_sector.keys())}
                    idx_to_stock = {idx: stock for stock, idx in stock_to_idx.items()}
                    if stock_idx in idx_to_stock and idx_to_stock[stock_idx] in stock_to_sector:
                        sector_idx = stock_to_sector[idx_to_stock[stock_idx]]
                else:
                    # Handle numeric keys
                    if stock_idx in stock_to_sector:
                        sector_idx = stock_to_sector[stock_idx]
            elif isinstance(stock_to_sector, (list, tuple, torch.Tensor)) and stock_idx < len(stock_to_sector):
                sector_idx = stock_to_sector[stock_idx]
            
            # Ensure sector_idx is within bounds
            sector_idx = min(sector_idx, self.num_sectors - 1)
            
            # Concatenate embeddings
            # Handle different tensor dimensions
            if len(long_term_g_embeddings.shape) == 3:
                stock_g_emb = long_term_g_embeddings[:, stock_idx, :]
            else:
                stock_g_emb = long_term_g_embeddings
            
            if len(long_term_a_embeddings.shape) == 3:
                stock_a_emb = long_term_a_embeddings[:, stock_idx, :]
            else:
                stock_a_emb = long_term_a_embeddings
            
            stock_sector_emb = inter_sector_output[:, sector_idx, :]
            
            fused_emb = torch.cat([stock_g_emb, stock_a_emb, stock_sector_emb], dim=1)
            fused_emb = F.relu(self.fusion(fused_emb))
            
            final_embeddings.append(fused_emb.unsqueeze(1))
            
            # Predict return and movement
            return_pred = self.return_predictor(fused_emb)
            movement_pred = torch.sigmoid(self.movement_predictor(fused_emb))
            
            return_predictions.append(return_pred)
            movement_predictions.append(movement_pred)
        
        final_embeddings = torch.cat(final_embeddings, dim=1)
        return_predictions = torch.cat(return_predictions, dim=1)
        movement_predictions = torch.cat(movement_predictions, dim=1)
        
        return return_predictions, movement_predictions, final_embeddings

