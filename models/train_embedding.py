import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from data.data_loader import EmbeddingDataLoader
from data.embedding_dataset import create_embedding_dataloaders
from models.embedding_classifier import EmbeddingMLP, EmbeddingCNN, EmbeddingTransformer
from utils.metrics import calculate_metrics

def train_embedding_model():
    config = {
        'data_path': 'data/dataset.jsonl',
        'model_type': 'mlp',
        'num_classes': 3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 50,
        'test_size': 0.2,
        'normalize': True,
        'hidden_dims': [512, 256],
        'dropout_rate': 0.3
    }
    
    #1.load data
    print("Loading data...")
    data_loader = EmbeddingDataLoader(config['data_path'])
    
    if config['data_path'].endswith('.json') or config['data_path'].endswith('.jsonl'):
       embeddings, labels = data_loader.load_json_data()
    elif config['data_path'].endswith('.parquet'):
       embeddings, labels = data_loader.load_parquet_data()
       
    data_info = data_loader.get_data_info(embeddings, labels)
    print("data info:", data_info)
    
    #2.create data loader
    train_loader, test_loader = create_embedding_dataloaders(
        embedding_dim = embeddings_dim,
        test_size = config['test_size'],
        batch_size = config['batch_size'],
        normalize = config['normalize_embedding']
    )
    
    #3.create model
    embedding_dim = embeddings.shape[1]
    if config['model_type'] == 'mlp':
        model = EmbeddingMLP(
            embedding_dim = embedding_dim,
            hidden_dims = config['hidden_dims'],
            num_classes = config['num_classes'],
            dropout_rate = config['dropout_rate']
        )
    elif config['model_type'] == 'transformer':
        model = EmbeddingTransformer(
            embedding_dim = embedding_dim,
            num_classes = config['num_classes'],
            dropout_rate = config['dropout_rate']
        )
    elif config['model_type'] == 'cnn':
        model = EmbeddingCNN(
            embedding_dim = embedding_dim,
            num_filters = config['num_filters'],
            filter_sizes = config['filter_sizes'],
            num_classes =  config['num_classes'],
            dropout_rate = config['dropout_rate']
        )
    else:
        raise ValueError(f"Invalid model type: {config['model_type']}")
    
    #4.train set
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.5)
    
    #5 train
    best_accuracy = 0
    for epoch in range(config['num_epochs']):
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_embeddings, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())
            
    #6. test
    model.eval
    test_loss = 0
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(batch_labels.cpu().numpy())
            
    #7. calculate metrics
    test_accuracy = accuracy_score(test_targets, test_predictions)
    print(f'Epoch {epoch + 1}/{config["num_epochs"]}:')
    print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}') 
    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    #8. learn rate scheduler
    scheduler.step(test_loss)
    
    #9. save model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'embedding_dim': embedding_dim,
            'best_accuracy': best_accuracy},
            f'result/models/best_{config["model_type"]}.model_pth'
        )
    #final evaluation
    print(f"train finish! the best accuracy is {best_accuracy:.4f}")
    print("final report:") 
    print(classification_report(test_targets, test_predictions, target_names=['class 0', 'class 1', 'class 2']))
    
if __name__ == '__main__':
    train_embedding_model()