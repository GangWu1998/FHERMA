import torch
import torch.nn as nn 
import torch.nn.functional as F

class EmbeddingCNN(nn.Module):
    
    def __init__(self, embedding_dim: int, num_filters: int = 100, filter_sizes: list[int] = [3,4,5],
                 num_classes: int = 3, dropout_rate: float = 0.5): 
        super(EmbeddingCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        self.fc_expand = nn.Linear(embedding_dim, embedding_dim * 4)
        self.seq_len = 4
        
        #1D conv
        self.convs = nn.ModuleList([nn.Conv1d(self.conv_input_dim, num_filters, kernel_size = k) for k in filter_sizes if k <= self.seq_len])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(self.convs) * num_filters, num_classes)
        
        def forward(self, x):
            #x: [batch_size, embedding_dim]
            batch_size = x.size(0)
            x = self.fc_expand(x) #[batch_size, embedding_dim * 4]
            # conv op
            conv_outputs = []
            for conv in self.convs:
                conv_out = F.relu(conv(x))
                pooled = F.max_pool1d(conv_out, conv_out.size(2))
                conv_outputs.append(pooled.squeeze(2))
            if conv_outputs:
                x = torch.cat(conv_outputs, dim = 1)
                x = x.view(batch_size, -1)
                
            x = self.dropout(x)
            x = self.fc(x)
            
            return x

class EmbeddingMLP(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: list[int] = [512, 256],
              num_classes: int = 3, dropout_rate: float = 0.5):
        super(EmbeddingMLP, self).__init__()
    
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])    
            prev_dim = hidden_dim 
            layers.append(nn.Linear(prev_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
        def forward(self, x):
            return self.classifier(x)
    
class EmbeddingTransformer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 8, num_layers: int = 2,
                 num_classes: int = 3, dropout_rate: float = 0.1):
        super(EmbeddingTransformer, self).__init__()
        self.embedding_dim = embediding_dim
        self.seq_len = 1
        emcoder_layer = nn.TransformerEncoderLayer(d_model = embedding_dim,
                                                nhead = num_heads,
                                                dropout = dropout_rate,
                                                batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer = encoder_layer,
                                                num_layers = num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(embedding_dim, num_classes)   
        ) 
        
        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)
            x = self.classifier(x)
            return x

        
        