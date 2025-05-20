import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import (
    DakshinaTSVDataset,
    Encoder,
    Decoder,
    Seq2Seq,
    compute_exact_match_accuracy,
    compute_char_accuracy
)

# ---- Training & Evaluation Functions ----
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_char_acc = 0
    epoch_exact_match_acc = 0
    total_batches = 0
    
    for src, tgt in tqdm(dataloader, desc="Training"):
        batch_size = src.size(0)
        src, tgt = src.to(device), tgt.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt)
        
        # Flatten output and target tensors for loss calculation
        # Ignore the first token in target (<sos>)
        output_flat = output.reshape(-1, output.shape[-1])
        target_flat = tgt[:, 1:].reshape(-1)  # Shift right to predict next token
        
        # Calculate loss
        loss = criterion(output_flat, target_flat)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        
        # Calculate metrics
        char_acc = compute_char_accuracy(output, tgt[:, 1:])
        
        # Decode for exact match accuracy
        with torch.no_grad():
            predictions, _ = model.decode(src)
            exact_match_acc = compute_exact_match_accuracy(predictions, tgt, dataloader.dataset.tgt_vocab)
        
        # Accumulate metrics
        epoch_loss += loss.item() * batch_size
        epoch_char_acc += char_acc * batch_size
        epoch_exact_match_acc += exact_match_acc * batch_size
        total_batches += batch_size
    
    # Return average metrics
    return {
        'loss': epoch_loss / total_batches,
        'char_acc': epoch_char_acc / total_batches,
        'exact_match_acc': epoch_exact_match_acc / total_batches
    }

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_char_acc = 0
    epoch_exact_match_acc = 0
    total_batches = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            batch_size = src.size(0)
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass (use teacher forcing for loss calculation)
            output = model(src, tgt)
            
            # Flatten output and target tensors for loss calculation
            output_flat = output.reshape(-1, output.shape[-1])
            target_flat = tgt[:, 1:].reshape(-1)  # Shift right to predict next token
            
            # Calculate loss
            loss = criterion(output_flat, target_flat)
            
            # Calculate metrics
            char_acc = compute_char_accuracy(output, tgt[:, 1:])
            
            # Decode for exact match accuracy (no teacher forcing)
            predictions, _ = model.decode(src)
            exact_match_acc = compute_exact_match_accuracy(predictions, tgt, dataloader.dataset.tgt_vocab)
            
            # Count exact matches for reporting
            correct_batch = int(exact_match_acc * batch_size)
            correct_predictions += correct_batch
            total_predictions += batch_size
            
            # Accumulate metrics
            epoch_loss += loss.item() * batch_size
            epoch_char_acc += char_acc * batch_size
            epoch_exact_match_acc += exact_match_acc * batch_size
            total_batches += batch_size
    
    # Return average metrics
    return {
        'loss': epoch_loss / total_batches,
        'char_acc': epoch_char_acc / total_batches,
        'exact_match_acc': epoch_exact_match_acc / total_batches,
        'correct': correct_predictions,
        'total': total_predictions
    }

# ---- WandB Sweep Configuration ----
sweep_config = {
    "name": "Seq2Seq",
    "method": "bayes",
    'metric': {
        'name': 'validation_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'cell_type': {
            'values': ['lstm', 'gru', 'rnn']
        },
        'dropout': {
            'values': [0, 0.1, 0.2, 0.5]
        },
        'embedding_size': {
            'values': [64, 128, 256, 512]
        },
        'num_layers': {
            'values': [2, 3, 4]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'hidden_size': {
            'values': [128, 256, 512]
        },
        'bidirectional': {
            'values': [True, False]
        },
        'learning_rate': {
            "values": [0.001, 0.002, 0.0001, 0.0002]
        },
        'epochs': {
            'values': [15]
        },
        'optim': {
            "values": ['adam']
        },
        'teacher_forcing': {
            "values": [0.2, 0.5, 0.7]
        }
    }
}

# ---- WandB Sweep Function ----
def sweep_run():
    # Initialize WandB run
    run = wandb.init()
    
    # Get hyperparameters from sweep
    config = wandb.config
    
    # Create run name
    run_name = f"{config.cell_type}-e{config.embedding_size}-h{config.hidden_size}-n{config.num_layers}-d{config.dropout}-b{config.bidirectional}-tf{config.teacher_forcing}-lr{config.learning_rate}-bs{config.batch_size}-{config.optim}"
    wandb.run.name = run_name
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    train_tsv = '/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.train.tsv'
    dev_tsv = '/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.dev.tsv'
    vocab_dir = '/kaggle/working/vocab'
    model_dir = '/kaggle/working/models'
    
    # Create directories
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load or build vocabulary
    vocab_file = os.path.join(vocab_dir, 'src_vocab.json')
    if os.path.exists(vocab_file):
        with open(os.path.join(vocab_dir, 'src_vocab.json'), 'r') as f:
            src_vocab = json.load(f)
        with open(os.path.join(vocab_dir, 'tgt_vocab.json'), 'r') as f:
            tgt_vocab = json.load(f)
        print("Loaded existing vocabulary")
    else:
        print("Building new vocabulary")
        train_dataset = DakshinaTSVDataset(train_tsv, build_vocab=True)
        src_vocab, tgt_vocab = train_dataset.src_vocab, train_dataset.tgt_vocab
        
        # Save vocabulary
        with open(os.path.join(vocab_dir, 'src_vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(src_vocab, f, ensure_ascii=False)
        with open(os.path.join(vocab_dir, 'tgt_vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(tgt_vocab, f, ensure_ascii=False)
        print("Saved vocabulary")
    
    # Create datasets
    train_dataset = DakshinaTSVDataset(train_tsv, src_vocab, tgt_vocab)
    val_dataset = DakshinaTSVDataset(dev_tsv, src_vocab, tgt_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create model components
    encoder = Encoder(
        input_size=len(src_vocab),
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        cell_type=config.cell_type
    )
    
    # Calculate encoder output size (doubled if bidirectional)
    enc_hidden_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
    
    decoder = Decoder(
        output_size=len(tgt_vocab),
        embedding_size=config.embedding_size,
        enc_hidden_size=enc_hidden_size,
        dec_hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        cell_type=config.cell_type
    )
    
    # Create full model
    model = Seq2Seq(encoder, decoder, device, teacher_forcing_ratio=config.teacher_forcing)
    model = model.to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Loss function (ignore padding token)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Optimizer
    if config.optim == 'nadam':
        try:
            optimizer = optim.NAdam(model.parameters(), lr=config.learning_rate)
        except AttributeError:
            print("NAdam optimizer not available, falling back to Adam")
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Char Acc: {train_metrics['char_acc']:.4f}, "
              f"Exact Match: {train_metrics['exact_match_acc']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Char Acc: {val_metrics['char_acc']:.4f}, "
              f"Exact Match: {val_metrics['exact_match_acc']:.4f} ({val_metrics['correct']}/{val_metrics['total']})")
        
        # Convert exact match to percentage for wandb
        val_accuracy_percent = val_metrics['exact_match_acc'] * 100
        
        # Log to WandB
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_char_accuracy': train_metrics['char_acc'],
            'train_exact_match': train_metrics['exact_match_acc'],
            'val_loss': val_metrics['loss'],
            'val_char_accuracy': val_metrics['char_acc'],
            'val_exact_match': val_metrics['exact_match_acc'],
            'validation_accuracy': val_accuracy_percent  # This matches the metric name in sweep_config
        })
        
        # Save best model
        if val_metrics['exact_match_acc'] > best_val_acc:
            best_val_acc = val_metrics['exact_match_acc']
            
            # Save model
            model_path = os.path.join(model_dir, f"{run_name}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['exact_match_acc'],
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            }, model_path)
            
            # Create a new artifact for this model
            artifact_name = f"model-{run.id}-epoch{epoch+1}"
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(model_path)
            run.log_artifact(artifact)
            
            print(f"Saved new best model with validation accuracy: {best_val_acc:.4f}")

def run_best_params(args):
    # Best parameters as provided or use from args
    params = {
        'cell_type': args.cell_type,
        'dropout': args.dropout,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'embedding_size': args.embedding_size,
        'bidirectional': args.bidirectional,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'optim': args.optimizer,
        'teacher_forcing': args.teacher_forcing
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    train_tsv = args.train_file
    dev_tsv = args.dev_file
    test_tsv = args.test_file
    vocab_dir = args.vocab_dir
    model_dir = args.model_dir
    
    # Create directories
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize wandb if needed
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=params)
    
    # Load or build vocabulary
    vocab_file = os.path.join(vocab_dir, 'src_vocab.json')
    if os.path.exists(vocab_file):
        with open(os.path.join(vocab_dir, 'src_vocab.json'), 'r') as f:
            src_vocab = json.load(f)
        with open(os.path.join(vocab_dir, 'tgt_vocab.json'), 'r') as f:
            tgt_vocab = json.load(f)
        print("Loaded existing vocabulary")
    else:
        print("Building new vocabulary")
        train_dataset = DakshinaTSVDataset(train_tsv, build_vocab=True)
        src_vocab, tgt_vocab = train_dataset.src_vocab, train_dataset.tgt_vocab
        
        # Save vocabulary
        with open(os.path.join(vocab_dir, 'src_vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(src_vocab, f, ensure_ascii=False)
        with open(os.path.join(vocab_dir, 'tgt_vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(tgt_vocab, f, ensure_ascii=False)
        print("Saved vocabulary")
    
    # Create datasets
    train_dataset = DakshinaTSVDataset(train_tsv, src_vocab, tgt_vocab)
    val_dataset = DakshinaTSVDataset(dev_tsv, src_vocab, tgt_vocab)
    test_dataset = DakshinaTSVDataset(test_tsv, src_vocab, tgt_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
    
    print(f"Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model components
    encoder = Encoder(
        input_size=len(src_vocab),
        embedding_size=params['embedding_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        bidirectional=params['bidirectional'],
        cell_type=params['cell_type']
    )
    
    # Calculate encoder output size (doubled if bidirectional)
    enc_hidden_size = params['hidden_size'] * 2 if params['bidirectional'] else params['hidden_size']
    
    decoder = Decoder(
        output_size=len(tgt_vocab),
        embedding_size=params['embedding_size'],
        enc_hidden_size=enc_hidden_size,
        dec_hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        cell_type=params['cell_type']
    )
    
    # Create full model
    model = Seq2Seq(encoder, decoder, device, teacher_forcing_ratio=params['teacher_forcing'])
    model = model.to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Loss function (ignore padding token)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Optimizer
    if params['optim'] == 'nadam':
        try:
            optimizer = optim.NAdam(model.parameters(), lr=params['learning_rate'])
        except AttributeError:
            print("NAdam optimizer not available, falling back to Adam")
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    model_path = os.path.join(model_dir, "best_model.pt")
    
    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch+1}/{params['epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Char Acc: {train_metrics['char_acc']:.4f}, "
              f"Exact Match: {train_metrics['exact_match_acc']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Char Acc: {val_metrics['char_acc']:.4f}, "
              f"Exact Match: {val_metrics['exact_match_acc']:.4f} ({val_metrics['correct']}/{val_metrics['total']})")
        
        # Log to WandB
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_char_accuracy': train_metrics['char_acc'],
                'train_exact_match': train_metrics['exact_match_acc'],
                'val_loss': val_metrics['loss'],
                'val_char_accuracy': val_metrics['char_acc'],
                'val_exact_match': val_metrics['exact_match_acc']
            })
        
        # Save best model
        if val_metrics['exact_match_acc'] > best_val_acc:
            best_val_acc = val_metrics['exact_match_acc']
            best_epoch = epoch + 1
            
            # Save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['exact_match_acc'],
                'params': params
            }, model_path)
            
            print(f"Saved new best model with validation accuracy: {best_val_acc:.4f}")
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Character Accuracy: {test_metrics['char_acc']:.4f}")
    print(f"Exact Match Accuracy: {test_metrics['exact_match_acc']:.4f} "
          f"({test_metrics['correct']}/{test_metrics['total']})")
    
    # Log final results to WandB
    if args.use_wandb:
        wandb.log({
            'test_loss': test_metrics['loss'],
            'test_char_accuracy': test_metrics['char_acc'],
            'test_exact_match': test_metrics['exact_match_acc'],
            'best_val_accuracy': best_val_acc
        })
    
    # Display examples of correct and incorrect predictions
    print("\nAnalyzing predictions on test set...")
    model.eval()
    all_predictions = []
    all_targets = []
    all_sources = []
    
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            predictions, _ = model.decode(src)
            
            # Convert to readable strings
            id_to_char = {v: k for k, v in test_dataset.tgt_vocab.items() 
                         if k not in ['<pad>', '<sos>', '<eos>', '<unk>']}
            
            for i in range(src.size(0)):
                # Source (roman)
                src_str = ''.join([test_dataset.src_vocab.get(str(idx.item()), '') 
                                  for idx in src[i] if idx.item() not in [0, 1, 2, 3]])
                
                # Target (native)
                tgt_str = ''.join([id_to_char.get(idx.item(), '') 
                                  for idx in tgt[i, 1:] if idx.item() not in [0, 1, 2, 3]])
                
                # Prediction
                pred_str = ''.join([id_to_char.get(idx.item(), '') 
                                   for idx in predictions[i, 1:] if idx.item() not in [0, 1, 2, 3]])
                
                all_sources.append(src_str)
                all_targets.append(tgt_str)
                all_predictions.append(pred_str)
    
    # Get correct and incorrect examples
    correct_examples = [(s, t, p) for s, t, p in zip(all_sources, all_targets, all_predictions) if t == p]
    incorrect_examples = [(s, t, p) for s, t, p in zip(all_sources, all_targets, all_predictions) if t != p]
    
    # Display some correct examples
    print(f"\nCorrect Examples ({len(correct_examples)} total):")
    for i, (src, tgt, pred) in enumerate(correct_examples[:5]):
        print(f"{i+1}. Roman: '{src}'")
        print(f"   Native: '{tgt}'")
    
    # Display some incorrect examples
    print(f"\nIncorrect Examples ({len(incorrect_examples)} total):")
    for i, (src, tgt, pred) in enumerate(incorrect_examples[:5]):
        print(f"{i+1}. Roman: '{src}'")
        print(f"   Native (correct): '{tgt}'")
        print(f"   Prediction: '{pred}'")
    
    return {
        'val_accuracy': best_val_acc,
        'test_accuracy': test_metrics['exact_match_acc'],
        'correct': test_metrics['correct'],
        'total': test_metrics['total']
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Seq2Seq model for transliteration')
    
    # Dataset arguments
    parser.add_argument('--train_file', type=str, default='data/train.tsv', 
                        help='Path to training data')
    parser.add_argument('--dev_file', type=str, default='data/dev.tsv', 
                        help='Path to validation data')
    parser.add_argument('--test_file', type=str, default='data/test.tsv', 
                        help='Path to test data')
    parser.add_argument('--vocab_dir', type=str, default='vocab', 
                        help='Directory to save/load vocabulary')
    parser.add_argument('--model_dir', type=str, default='models', 
                        help='Directory to save models')
    
    # Model hyperparameters
    parser.add_argument('--cell_type', type=str, default='lstm', 
                        choices=['lstm', 'gru', 'rnn'], help='RNN cell type')
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=4, 
                        help='Number of RNN layers')
    parser.add_argument('--hidden_size', type=int, default=512, 
                        help='Hidden size of RNN')
    parser.add_argument('--embedding_size', type=int, default=512, 
                        help='Embedding size')
    parser.add_argument('--bidirectional', type=bool, default=True, 
                        help='Whether to use bidirectional encoder')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'nadam'], help='Optimizer')
    parser.add_argument('--teacher_forcing', type=float, default=0.7, 
                        help='Teacher forcing ratio')
    
    # WandB arguments
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Whether to use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='transliteration', 
                        help='WandB project name')
    parser.add_argument('--run_sweep', action='store_true', 
                        help='Whether to run a hyperparameter sweep')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.run_sweep:
        # Run WandB sweep
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=sweep_run)
    else:
        # Run with best params or provided args
        results = run_best_params(args)
        print(f"\nFinal Results - Validation Accuracy: {results['val_accuracy']:.4f}, "
              f"Test Accuracy: {results['test_accuracy']:.4f} ({results['correct']}/{results['total']})")