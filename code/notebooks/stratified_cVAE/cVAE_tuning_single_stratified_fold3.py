from torch.utils.data import DataLoader,SubsetRandomSampler, Subset
from library.cVAE import GCN_Encoder, GRU_Decoder, cVAE
from library.cVAE_helper import (
    loss_function,
    smiles_to_idxs,
    train_model,
    test_model,
    get_dataloader,
    make_stratified_bins,
    get_vocab
)


from sklearn.model_selection import StratifiedKFold
from library.GCN import GraphData
from pathlib import Path
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna
import torch
import json
import os

# Fixed params
FIXED_PARAMS = {
    'max_atoms': 30,
    'node_vec_len': 16,
    'use_GPU': False,  # Set to True if CUDA available
    'vocab_size': 24,
    # 'batch_size': 1000,
    'p_dropout': 0.1
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
        'batch_size': {'type': 'int', 'low': 100, 'high': 2000},
        'n_epochs': {'type': 'int', 'low': 10, 'high': 30},  # Low for inner, full for final
        'teacher_forcing_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9},
        'beta': {'type': 'float', 'low': 0.1, 'high': 10, 'log': True},
    },
    'n_trials': 3,  # 45 Adjust: 20-50 for prototyping
    'direction': 'minimize',
}

# Auto-detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_GPU = (DEVICE == "cuda")
FIXED_PARAMS['use_GPU'] = use_GPU
print(f"Using device: {DEVICE} (GPU: {use_GPU})")

OUTER_FOLD_IDX = 3
FOLD_TYPE = 'stratified'


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


# ---------------------------------------------------------------

def inner_cv_objective(trial, outer_train_dataset, token2idx, n_inner_folds=10):
    """10 inner folds on outer_train_indices for one trial."""
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

    y_inner_gaps = np.array([outer_train_dataset[i][1].squeeze(-1).item() for i in range(len(outer_train_dataset))])
    inner_binned_gaps = make_stratified_bins(y_inner_gaps, n_bins=10)

    inner_losses = []
    
    # Split relative indices with stratification
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
        inner_cv.split(range(len(outer_train_dataset)), y=inner_binned_gaps)
    ):
        print(f"Inner Fold {inner_fold+1}/{n_inner_folds} for trial {trial.number}...")

        model, trial_params, lr = create_model(trial=trial, fixed_params=FIXED_PARAMS)

        batch_size = trial_params['batch_size']
        train_loader = get_dataloader(outer_train_dataset, inner_train_idx, batch_size)
        val_loader = get_dataloader(outer_train_dataset, inner_val_idx, batch_size)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        inner_epochs = trial_params['n_epochs']
        for epoch in range(inner_epochs):
            train_model(
                epoch, model, train_loader, optimizer,
                lambda m, l, t, b: loss_function(m, l, t, b, beta=trial_params['beta']),
                FIXED_PARAMS['use_GPU'], DEVICE, FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len'],
                token2idx
            )
        
        val_loss, _ = test_model(
            model, val_loader,
            lambda m, l, t, b: loss_function(m, l, t, b, beta=trial_params['beta']),
            FIXED_PARAMS['use_GPU'], DEVICE, FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len'],
            token2idx
        )
        inner_losses.append(val_loss)

        # Clear GPU mem post-trial
        if use_GPU:
            torch.cuda.empty_cache()
    
    avg_loss = np.mean(inner_losses)
    print(f"Avg Inner Loss: {avg_loss:.4f}")
    return avg_loss

def run_tuning_per_fold(outer_train_dataset, token2idx):
    """Single-phase tuning on outer_train; final train/eval."""
    print("Starting Single-Phase Tuning...")
    def obj(trial):
        return inner_cv_objective(trial, outer_train_dataset, token2idx)
    
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


def save_results(output_data, fold_type):
    # Define output directory and ensure it exists
    output_dir = f'/data/gent/vo/000/gvo00004/vsc48887/machine_learning/results/GCN_model_outer'
    os.makedirs(output_dir, exist_ok=True)

    # Define output filename
    output_file = os.path.join(output_dir, f"outer_fold_{output_data['outer_fold_idx']}_{fold_type}_results.json")

    # Save to file as JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

def main():
    # Load in the dataset
    data_path = "rdkit_only_valid_smiles_qm9.pkl"
    dataset = GraphData(dataset_path=data_path, max_atoms=FIXED_PARAMS['max_atoms'], node_vec_len=FIXED_PARAMS['node_vec_len'])

    smiles_list: list[str] = dataset.smiles

    vocab_list, _ = get_vocab(smiles_list)

    # Create token2index mapping and its inverse
    token2idx = {tok: idx for idx, tok in enumerate(vocab_list)}

    dataset_indices = np.arange(0, len(dataset), 1)

    # Get outer fold nr ... from the respective json file
    fold_data = load_fold_file(OUTER_FOLD_IDX, FOLD_TYPE)

    outer_train_idx = fold_data['train_indices']
    outer_test_idx = fold_data['test_indices']

    outer_train_indices = np.array(dataset_indices)[outer_train_idx].tolist()
    outer_test_indices = np.array(dataset_indices)[outer_test_idx].tolist()
    outer_train_dataset = Subset(dataset, outer_train_indices)

    best_params = run_tuning_per_fold(outer_train_dataset, token2idx)

    # Final train on full outer_train
    print("  Final Training on Outer Train...")
    batch_size = best_params['batch_size']
    n_epochs = best_params['n_epochs']

    train_loader = get_dataloader(dataset, outer_train_indices, batch_size)
    test_loader = get_dataloader(dataset, outer_test_indices, batch_size)
    
    model, _, lr = create_model(init_params=best_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    for epoch in range(n_epochs):
        train_loss, train_acc = train_model(
                epoch, model, train_loader, optimizer,
                lambda m, l, t, b: loss_function(m, l, t, b, beta=best_params['beta']),
                FIXED_PARAMS['use_GPU'], DEVICE, FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len'],
                token2idx
            )
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
    
    print("  Evaluating on Outer Test...")
    test_loss, test_acc = test_model(
            model, test_loader,
            lambda m, l, t, b: loss_function(m, l, t, b, beta=best_params['beta']),
            FIXED_PARAMS['use_GPU'], DEVICE, FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len'],
            token2idx
        )
    
    # Collect relevant data to save
    output_data = {
        'outer_fold_idx': OUTER_FOLD_IDX,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        **best_params
    }

    save_results(output_data, FOLD_TYPE)

    # print(f"  Outer Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    # print(test_loss, test_acc, best_params)

if __name__ == "__main__":
    main()