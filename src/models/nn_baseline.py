
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, num_numerical_features, embedding_dims, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(SimpleMLP, self).__init__()
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cats, dim) for num_cats, dim in embedding_dims
        ])
        
        total_emb_dim = sum(dim for _, dim in embedding_dims)
        input_dim = num_numerical_features + total_emb_dim
        
        # MLP Layers
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
            
        self.mlp = nn.Sequential(*layers)
        
        # Output Layer
        self.output = nn.Linear(curr_dim, 1)
        
    def forward(self, x_num, x_cat):
        # x_num: (batch, num_numerical)
        # x_cat: (batch, num_categorical)
        
        # Process Embeddings
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] is 1D tensor of indices for i-th categorical feature
            embedded.append(emb_layer(x_cat[:, i]))
            
        if embedded:
            x_emb = torch.cat(embedded, dim=1)
            x = torch.cat([x_num, x_emb], dim=1)
        else:
            x = x_num
            
        x = self.mlp(x)
        return torch.sigmoid(self.output(x))
