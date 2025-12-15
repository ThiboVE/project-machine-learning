from torch.utils.data import DataLoader,SubsetRandomSampler, Subset
from library.cVAE import GCN_Encoder, GRU_Decoder, cVAE
from sklearn.model_selection import StratifiedKFold
from library.GCN import GraphData, collate_graph_dataset
from typing import Iterable
from pathlib import Path
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna
import torch
import json

# Fixed params
FIXED_PARAMS = {
    'max_atoms': 30,
    'node_vec_len': 16,
    'use_GPU': False,  # Set to True if CUDA available
    'vocab_size': 24,
    'batch_size': 1000
}

PARAMS = {
    'params': {
        # High importance
        'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-1, 'log': True},
        'latent_dim': {'type': 'int', 'low': 8, 'high': 64},
        'n_hidden': {'type': 'int', 'low': 32, 'high': 96},
        'gru_dim': {'type': 'int', 'low': 8, 'high': 64},
        # Medium
        'n_conv_layers': {'type': 'int', 'low': 1, 'high': 4},
        'n_hidden_layers': {'type': 'int', 'low': 1, 'high': 3},
        'n_gru_layers': {'type': 'int', 'low': 1, 'high': 3},
        'n_fc_layers': {'type': 'int', 'low': 2, 'high': 3},
        'embedding_dim': {'type': 'int', 'low': 8, 'high': 24},
        # Low
        # 'batch_size': {'type': 'int', 'low': 100, 'high': 2000},
        'n_epochs': {'type': 'int', 'low': 3, 'high': 15},  # Low for inner, full for final
        'teacher_forcing_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9},
        'beta': {'type': 'float', 'low': 0.1, 'high': 10, 'log': True},
        'p_dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
    },
    'n_trials': 3,  # 45 Adjust: 20-50 for prototyping
    'direction': 'minimize',
}

# Auto-detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_GPU = (DEVICE == "cuda")
FIXED_PARAMS['use_GPU'] = use_GPU
print(f"Using device: {DEVICE} (GPU: {use_GPU})")


def loss_function(model, logits, targets, batch_size, beta=1):
    recon_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss_recon = recon_loss_fn(logits, targets)
    
    kl_loss = -0.5 * torch.sum(1 + model.z_logvar - model.z_mean.pow(2) - model.z_logvar.exp()) / batch_size
    loss = loss_recon + beta * kl_loss

    return loss

def smiles_to_idxs(smiles):
    modified_smiles = [] # [batch_size, seq_len]
    for smile in smiles:
        # Add the start and end tokens to the smiles
        char_list = ['<STR>'] + list(smile) + ['<END>']
        # Convert tokens to indices
        vocab_idx_list = torch.as_tensor([token2idx[ch] for ch in char_list], device=DEVICE)
        modified_smiles.append(vocab_idx_list)
    
    padded_smiles = padding(modified_smiles, device=DEVICE)
    return padded_smiles

def padding(smiles, device=DEVICE):
    # find max length
    max_seq_len = max(s.size(0) for s in smiles)

    # pad with a PAD token index
    pad_idx = token2idx['<PAD>']

    padded_smiles = torch.full((len(smiles), max_seq_len), pad_idx, dtype=torch.long, device=device)

    for i, seq in enumerate(smiles):
        padded_smiles[i, :seq.size(0)] = seq
    
    return padded_smiles

def smiles_level_accuracy(logits, targets, pad_idx=None):
    """
    logits:  (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    pad_idx: integer for PAD token (optional)
    """
    # Predicted token IDs
    preds = torch.argmax(logits, dim=-1)  # (batch, seq_len)

    # Boolean matrix: correct prediction per position
    correct = (preds == targets)  # (batch, seq_len)

    # Mask out padding positions
    if pad_idx is not None:
        mask = (targets != pad_idx).float()
        correct = correct.float() * mask
        lengths = mask.sum(dim=1)               # seq_len per SMILES (no pads)
    else:
        correct = correct.float()
        lengths = torch.full(
            (targets.size(0),), targets.size(1), device=targets.device
        )

    # Per-SMILES reconstruction accuracy
    per_smiles_acc = correct.sum(dim=1) / lengths   # (batch,)

    # Average over all SMILES in the batch
    avg_acc = per_smiles_acc.mean()

    return avg_acc.item(), per_smiles_acc


def train_model(
    epoch,
    model,
    training_dataloader,
    optimizer,
    loss_fn,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    """
    Custom function which defines how a model will be trained (per epoch), here the mean-squared loss between prediction and actual value is used as evaluation metric. This function will perform backpropagation which updates the weights of the networks based in this evaluation.
    """
    # Create variables to store losses and error
    avg_loss = 0
    avg_validity = 0
    count = 0
    epoch_acc_sum = 0.0
    epoch_smiles_count = 0
    # Switch model to train mode
    model.train()
    # Go over each batch in the dataloader
    for i, dataset in enumerate(training_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        gap = dataset[1]
        
        padded_smiles = smiles_to_idxs(dataset[2])
        batch_size, max_seq_len = padded_smiles.size()

        # Reshape inputs
        node_mat = node_mat.reshape(batch_size, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(batch_size, max_atoms, max_atoms)

        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            model_input = (node_mat.to(DEVICE), adj_mat.to(DEVICE), padded_smiles.to(DEVICE), gap.to(DEVICE))
            model_output = padded_smiles
        else:
            model_input = (node_mat, adj_mat, padded_smiles, gap)
            model_output = padded_smiles

        # Compute output from network
        model_prediction_distribution = model(*model_input) # [batch_size, max_smiles_seq_len, vocab_size]

        # Calculate loss
        loss = loss_fn(model, model_prediction_distribution.permute(0, 2, 1), model_output, batch_size)
        avg_loss += loss.item()

        # returns (batch_avg, per_smiles_vector)
        batch_avg, per_smiles_acc = smiles_level_accuracy(model_prediction_distribution, padded_smiles)
        epoch_acc_sum += per_smiles_acc.sum().item()   # sum of accuracies for this batch
        epoch_smiles_count += per_smiles_acc.size(0)   # number of molecules in batch

        # Set zero gradients for all tensors
        optimizer.zero_grad()

        # Do backward prop
        loss.backward()

        # Update optimizer parameters
        optimizer.step()

        # Increase count
        count += 1

    # Calculate avg loss and validity
    avg_loss = avg_loss / count

    # avg_validity = avg_validity / count
    epoch_smiles_accuracy = epoch_acc_sum / epoch_smiles_count

    # Print stats
    print(f"Epoch: [{epoch}]\tTraining Loss: [{avg_loss:.2f}]\tReconstruction accuracy: [{epoch_smiles_accuracy}]")

    # Return loss and validity
    return avg_loss, epoch_smiles_accuracy


def test_model(
    model,
    test_dataloader,
    loss_fn,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    """
    Test the ChemGCN model.
    Parameters
    ----------
    model : ChemGCN
        ChemGCN model object
    test_dataloader : data.DataLoader
        Test DataLoader
    loss_fn : like nn.MSELoss()
        Model loss function
    standardizer : Standardizer
        Standardizer object
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph
    Returns
    -------
    test_loss : float
        Test loss
    test_mae : float
        Test MAE
    """
    # Create variables to store losses and error
    test_loss = 0
    epoch_acc_sum = 0.0
    epoch_smiles_count = 0
    count = 0

    # Switch model to train mode
    model.eval()

    # Go over each batch in the dataloader
    for i, dataset in enumerate(test_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        gap = dataset[1]
        padded_smiles = smiles_to_idxs(dataset[2])
        batch_size, _ = padded_smiles.size()

        # Reshape inputs
        node_mat = node_mat.reshape(batch_size, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(batch_size, max_atoms, max_atoms)

        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            model_input = (node_mat.to(DEVICE), adj_mat.to(DEVICE), padded_smiles.to(DEVICE), gap.to(DEVICE))
            model_output = padded_smiles
        else:
            model_input = (node_mat, adj_mat, padded_smiles, gap)
            model_output = padded_smiles

        # Compute output from network
        model_prediction_distribution = model(*model_input) # [batch_size, max_smiles_seq_len, vocab_size]

        # Calculate loss
        loss = loss_fn(model, model_prediction_distribution.permute(0, 2, 1), model_output, batch_size)
        test_loss += loss.item()

        # returns (batch_avg, per_smiles_vector)
        batch_avg, per_smiles_acc = smiles_level_accuracy(model_prediction_distribution, padded_smiles)
        epoch_acc_sum += per_smiles_acc.sum().item()   # sum of accuracies for this batch
        epoch_smiles_count += per_smiles_acc.size(0)   # number of molecules in batch
        
        # Increase count
        count += 1

    # Calculate avg loss and MAE
    test_loss = test_loss / count
    epoch_smiles_accuracy = epoch_acc_sum / epoch_smiles_count

    # Print stats
    print(f"Training Loss: [{test_loss:.2f}]\tReconstruction accuracy: [{epoch_smiles_accuracy}]")

    # Return loss and validity
    return test_loss, epoch_smiles_accuracy

def create_model(trial=None, init_params=None, fixed_params=None):
    """
    Create cVAE model based on Optuna trial suggestions for the current phase.
    fixed_params: Dict of previously tuned params to fix (e.g., from prior phases).
    """

    if init_params is not None:
        params = init_params
    
    else:
        params = {}
        # Suggest params for current phase
        for param_name, spec in PARAMS['params'].items():
            if spec['type'] == 'float':
                if spec.get('log', False):
                    params[param_name] = trial.suggest_float(param_name, spec['low'], spec['high'], log=True)
                else:
                    params[param_name] = trial.suggest_float(param_name, spec['low'], spec['high'])
            elif spec['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, spec['low'], spec['high'])
    
    # Merge with fixed params from prior phases
    if fixed_params:
        params.update(fixed_params)
    
    # Extract for model construction
    lr = params['learning_rate']
    latent_dim = params['latent_dim']
    n_hidden = params['n_hidden']
    gru_dim = params['gru_dim']
    embedding_dim = params.get('embedding_dim', 8)  # Default if not tuned
    n_conv_layers = params.get('n_conv_layers', 2)
    n_hidden_layers = params.get('n_hidden_layers', 2)
    n_gru_layers = params.get('n_gru_layers', 2)
    n_fc_layers = params.get('n_fc_layers', 3)
    p_dropout = params.get('p_dropout', 0.1)
    teacher_forcing_ratio = params.get('teacher_forcing_ratio', 0.5)
    gcn_hidden_nodes = n_hidden + 1  # Derived
    
    # Build components
    encoder = GCN_Encoder(
        node_vec_len=FIXED_PARAMS['node_vec_len'],
        node_fea_len=n_hidden,
        hidden_fea_len=n_hidden,
        n_conv=n_conv_layers,
        n_hidden=n_hidden_layers,
        n_outputs=1,
        p_dropout=p_dropout
    )
    
    decoder = GRU_Decoder(
        vocab_size=FIXED_PARAMS['vocab_size'],
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        hidden_size=gru_dim,
        n_gru_layers=n_gru_layers,
        n_fc_layers=n_fc_layers
    ).to(DEVICE)
    
    model = cVAE(
        encoder=encoder,
        decoder=decoder,
        device=DEVICE,
        n_gcn_hidden_dim=gcn_hidden_nodes,
        n_gru_hidden_dim=gru_dim,
        latent_dim=latent_dim,
        vocab_size=FIXED_PARAMS['vocab_size'],
        embedding_dim=embedding_dim,
        teacher_forcing_ratio=teacher_forcing_ratio
    ).to(DEVICE)
    
    return model, params, lr


def get_dataloader(dataset_obj: GraphData, indices: Iterable, batch_size: int):
    subset = Subset(dataset_obj, indices)  # Creates the restricted view
    return DataLoader(
        subset,  # PyTorch shuffles subset's local indices
        batch_size=batch_size,
        shuffle=True,  # Random order within the subset
        collate_fn=collate_graph_dataset
    )

def get_vocab(smiles_list):
    # Collect all unique characters
    charset = set()
    for smi in smiles_list:
        for ch in smi:
            charset.add(ch)

    # Sort for consistency
    charset = sorted(list(charset))

    # Add special tokens
    special_tokens = ['<PAD>', '<END>', '<STR>']
    vocab_list = special_tokens + charset

    return vocab_list, len(vocab_list)


def make_stratified_bins(y, n_bins=10):
    """
    Creates quantile bins for stratified k-fold in regression.
    """
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

# ---------------------------------------------------------------

def inner_cv_objective(trial, outer_train_dataset):
    """10 inner folds on outer_train_indices for one trial."""
    n_inner_folds = 2
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

    y_inner_gaps = np.array([outer_train_dataset[i][1].squeeze(-1).item() for i in range(len(outer_train_dataset))])
    inner_binned_gaps = make_stratified_bins(y_inner_gaps, n_bins=10)

    inner_losses = []
    
    # Split relative indices with stratification
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
        inner_cv.split(range(len(outer_train_dataset)), y=inner_binned_gaps)
    ):
        print(f"Inner Fold {inner_fold+1}/{n_inner_folds} for trial {trial.number}...")

        batch_size = FIXED_PARAMS['batch_size']
        train_loader = get_dataloader(outer_train_dataset, inner_train_idx, batch_size)
        val_loader = get_dataloader(outer_train_dataset, inner_val_idx, batch_size)
        
        model, trial_params, lr = create_model(trial=trial)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        inner_epochs = trial_params['n_epochs']
        for epoch in range(inner_epochs):
            train_model(
                epoch, model, train_loader, optimizer,
                lambda m, l, t, b: loss_function(m, l, t, b, beta=trial_params['beta']),
                FIXED_PARAMS['use_GPU'], FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len']
            )
        
        val_loss, _ = test_model(
            model, val_loader,
            lambda m, l, t, b: loss_function(m, l, t, b, beta=trial_params['beta']),
            FIXED_PARAMS['use_GPU'], FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len']
        )
        inner_losses.append(val_loss)

        # Clear GPU mem post-trial
        if use_GPU:
            torch.cuda.empty_cache()
    
    avg_loss = np.mean(inner_losses)
    print(f"Avg Inner Loss: {avg_loss:.4f}")
    return avg_loss

def run_tuning_per_fold(outer_train_dataset):
    """Single-phase tuning on outer_train; final train/eval."""
    print("Starting Single-Phase Tuning...")
    def obj(trial):
        return inner_cv_objective(trial, outer_train_dataset)
    
    study = optuna.create_study(
        direction=PARAMS['direction'],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=10),  # Prune early
    )
    study.optimize(obj, n_trials=PARAMS['n_trials'], n_jobs=1)  # n_jobs>1 for multi-GPU if setup
    
    best_params = study.best_params
    print(f"Best Params: {best_params}")
    print(f"Best Inner CV Loss: {study.best_value:.4f}")

    return best_params

def load_fold_file(fold: int, fold_type: str):
    main_path = Path.cwd().parents[0]
    data_path = main_path / "data" / "cvae_folds"

    fold_file = data_path / f'fold_{fold}_{fold_type}_data.json'

    with open(fold_file, 'r') as f:
        fold_data = json.load(f)

    return fold_data


def main():
    # Load in the dataset
    main_path = Path.cwd().parents[0]
    data_path = main_path / "data" / "RDKit" / "rdkit_only_valid_smiles_qm9.pkl"
    dataset = GraphData(dataset_path=data_path, max_atoms=FIXED_PARAMS['max_atoms'], node_vec_len=FIXED_PARAMS['node_vec_len'])

    smiles_list: list[str] = dataset.smiles

    vocab_list, _ = get_vocab(smiles_list)

    # Create token2index mapping and its inverse
    global token2idx
    token2idx = {tok: idx for idx, tok in enumerate(vocab_list)}
    # idx2token = {idx: tok for tok, idx in token2idx.items()}

    dataset_indices = np.arange(0, len(dataset), 1)

    fold_data = load_fold_file(0, 'random')

    outer_train_idx = fold_data['train_indices']
    outer_test_idx = fold_data['test_indices']

    outer_train_indices = np.array(dataset_indices)[outer_train_idx].tolist()
    outer_test_indices = np.array(dataset_indices)[outer_test_idx].tolist()
    outer_train_dataset = Subset(dataset, outer_train_indices)

    best_params = run_tuning_per_fold(outer_train_dataset)

    # Final train on full outer_train
    print("  Final Training on Outer Train...")
    batch_size = best_params['batch_size']
    n_epochs = best_params['n_epochs']

    train_loader = get_dataloader(dataset, outer_train_indices, batch_size)
    test_loader = get_dataloader(dataset, outer_test_indices, batch_size)
    
    model, _, lr = create_model(params=best_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        train_model(
            epoch, model, train_loader, optimizer,
            lambda m, l, t, b: loss_function(m, l, t, b, beta=best_params['beta']),
            FIXED_PARAMS['use_GPU'], FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len']
        )
    
    print("  Evaluating on Outer Test...")
    test_loss, test_acc = test_model(
        model, test_loader,
        lambda m, l, t, b: loss_function(m, l, t, b, beta=best_params['beta']),
        FIXED_PARAMS['use_GPU'], FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len']
    )
    
    print(f"  Outer Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    print(test_loss, test_acc, best_params)

if __name__ == "__main__":
    main()