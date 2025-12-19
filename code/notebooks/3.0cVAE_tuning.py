from library.cVAE import GCN_Encoder, GRU_Decoder, cVAE
from library.cVAE_loss_function import loss_function
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import  Subset
from library.GCN import GraphData
from library.cVAE_helper import (
    make_stratified_bins,
    get_dataloader,
    cVAE_train_model,
    cVAE_test_model,
    get_vocab,

)

from pathlib import Path
import numpy as np
import logging
import optuna
import torch
import json

# Fixed params
FIXED_PARAMS = {
    'max_atoms': 30,
    'node_vec_len': 16,
    'use_GPU': False,  # Set to True if CUDA available
    'vocab_size': 24,
    'batch_size': 1000,
    'p_dropout': 0.1,
    'n_epochs': 15
}

PARAMS = {
    'params': {
        # High importance
        'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-1, 'log': True},
        'latent_dim': {'type': 'int', 'low': 8, 'high': 64},
        'n_hidden': {'type': 'int', 'low': 16, 'high': 96},
        'gru_dim': {'type': 'int', 'low': 8, 'high': 64},
        # Medium
        'n_conv_layers': {'type': 'int', 'low': 1, 'high': 4},
        'n_hidden_layers': {'type': 'int', 'low': 1, 'high': 3},
        'n_gru_layers': {'type': 'int', 'low': 1, 'high': 3},
        'n_fc_layers': {'type': 'int', 'low': 2, 'high': 3},
        'embedding_dim': {'type': 'int', 'low': 8, 'high': 24},
        # Low
        'teacher_forcing_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9},
        'beta': {'type': 'float', 'low': 0.1, 'high': 10, 'log': True},
    },
    'n_trials': 15,
    'direction': 'minimize',
}

# Parameters of the first point I want to explore in the Bayesian optimization algorithm.
INIT_PARAMS = {
            "lr": 0.0013292918943162175,
            "latent_dim": 62,
            "n_hidden": 79,
            "gru_dim": 42,
            "n_conv_layers": 1,
            "n_hidden_layers": 1,
            "n_gru_layers": 1,
            "n_fc_layers": 3,
            "embedding_dim": 18,
            "teacher_forcing_ratio": 0.8759278817295955,
            "beta": 4.622589001020832
}


# Auto-detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_GPU = (DEVICE == "cuda")
FIXED_PARAMS['use_GPU'] = use_GPU

OUTER_SEED = 42
INNER_SEED = 156

DATA_PATH = Path.cwd().parents[0] / "data"

def create_model(trial=None, init_params=None, fixed_params=None):
    """
    Create cVAE model based on Optuna trial suggestions for the current phase.
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
        p_dropout=FIXED_PARAMS['p_dropout']
    ).to(DEVICE)
    
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

def inner_cv_objective(trial, outer_train_dataset, n_inner_folds=5):
    """Inner folds on outer_train_indices for one trial."""
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=INNER_SEED)

    y_inner_gaps = np.array([outer_train_dataset[i][1].squeeze(-1).item() for i in range(len(outer_train_dataset))])
    inner_binned_gaps = make_stratified_bins(y_inner_gaps, n_bins=10)

    inner_losses = []
    
    # Split relative indices with stratification
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
        inner_cv.split(range(len(outer_train_dataset)), y=inner_binned_gaps)
    ):
        logger.info(f"Inner Fold {inner_fold+1}/{n_inner_folds} for trial {trial.number}...")

        batch_size = FIXED_PARAMS['batch_size']
        train_loader = get_dataloader(outer_train_dataset, inner_train_idx, batch_size)
        val_loader = get_dataloader(outer_train_dataset, inner_val_idx, batch_size)
        
        model, trial_params, lr = create_model(trial=trial)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        inner_epochs = FIXED_PARAMS['n_epochs']
        for epoch in range(inner_epochs):
            inner_train_loss, inner_train_acc = cVAE_train_model(
                epoch, model, train_loader, optimizer,
                lambda m, l, t, b: loss_function(m, l, t, b, beta=trial_params['beta']),
                FIXED_PARAMS['use_GPU'], DEVICE, FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len'], token2idx
            )

            logger.info(f"Inner Train Loss: [{inner_train_loss:.2f}]\tReconstruction accuracy: [{inner_train_acc}]")
        
        val_loss, val_acc = cVAE_test_model(
            model, val_loader,
            lambda m, l, t, b: loss_function(m, l, t, b, beta=trial_params['beta']),
            FIXED_PARAMS['use_GPU'], DEVICE, FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len'], token2idx
        )
        inner_losses.append(val_loss)

        logger.info(f"Validation Loss: [{val_loss:.2f}]\tReconstruction accuracy: [{val_acc}]")

        # Clear GPU mem post-trial
        if use_GPU:
            torch.cuda.empty_cache()
    
    avg_loss = np.mean(inner_losses)
    logger.info(f"Avg Inner Loss: {avg_loss:.4f}")
    return avg_loss


def run_tuning_per_fold(outer_train_dataset):
    """Single-phase tuning on outer_train; final train/eval."""
    
    logger.info("Starting Single-Phase Tuning...")

    def obj(trial):
        return inner_cv_objective(trial, outer_train_dataset)
    
    study = optuna.create_study(
        direction=PARAMS['direction'],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=10),  # Prune early
    )

    study.enqueue_trial(INIT_PARAMS) # Define an inital trial I want the study to run
    study.optimize(obj, n_trials=PARAMS['n_trials'], n_jobs=1)
    
    best_params = study.best_params
    logger.info(f"Best Params: {best_params}")
    logger.info(f"Best Inner CV Loss: {study.best_value:.4f}")

    return best_params


def save_fold(fold, train_idx, test_idx, vocab_size, fold_type):
    """Save the data of an outer fold"""
    output_dir = DATA_PATH / "cvae_folds"
    output_dir.mkdir(exist_ok=True)

    fold_data = {
        'fold_id': fold,
        'train_indices': train_idx.tolist(),  # Global indices for Subset
        'test_indices': test_idx.tolist(),
        'n_samples_train': len(train_idx),
        'n_samples_test': len(test_idx),
        'vocab_size': vocab_size,  # For consistency
    }

    fold_file = output_dir / f'fold_{fold}_{fold_type}_data.json'

    with open(fold_file, 'w') as f:
        json.dump(fold_data, f)
    logger.info(f"Saved fold {fold} to {fold_file}")


def save_fold_result(fold, train_losses, train_accs, test_loss, test_acc, best_params, fold_type='stratified'):
    """Save the results of an outer fold"""

    output_dir = DATA_PATH / "results"
    output_dir.mkdir(exist_ok=True)

    fold_data = {
        'fold_id': fold,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_loss': test_loss,
        'test_acc': test_acc,
        **best_params
    }

    fold_file = output_dir / f'fold_{fold}_{fold_type}_results.json'

    with open(fold_file, 'w') as f:
        json.dump(fold_data, f)
    logger.info(f"Saved results of fold {fold} to {fold_file}")


def save_model(state_dict, fold, fold_type='stratified'):
    """Save the weights of a model after its final training in an outer fold"""
    output_dir = DATA_PATH / "models" 
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'fold_{fold}_{fold_type}_model.pt'

    torch.save(state_dict, output_file)
    logger.info(f"Saved model of fold {fold} to {output_dir}")


def setup_logging(DATA_PATH):
    # Create log directory
    log_dir = DATA_PATH / "logs"
    log_dir.mkdir(exist_ok=True)

    # Log filename
    log_filename = log_dir / "cvae_training.log"

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),
            logging.StreamHandler()
        ]
    )

    # Main logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_filename}")

    # Attach handlers to Optuna logger as well
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.INFO)
    # Avoid duplicate logs if handlers already exist
    if not optuna_logger.handlers:
        optuna_logger.addHandler(logging.FileHandler(log_filename, mode="a"))
        optuna_logger.addHandler(logging.StreamHandler())

    return logger


def main():
    # Load in the dataset
    data_path = DATA_PATH / "RDKit" / "rdkit_only_valid_smiles_qm9.pkl"
    dataset = GraphData(dataset_path=data_path, max_atoms=FIXED_PARAMS['max_atoms'], node_vec_len=FIXED_PARAMS['node_vec_len'])

    gaps: list[float] = dataset.outputs
    smiles_list: list[str] = dataset.smiles

    vocab_list, _ = get_vocab(smiles_list)

    # Create token2index mapping and its inverse
    global token2idx
    token2idx = {tok: idx for idx, tok in enumerate(vocab_list)}
    # idx2token = {idx: tok for tok, idx in token2idx.items()}

    dataset_indices = np.arange(0, len(dataset), 1)

    outer_binned_gaps = make_stratified_bins(gaps)

    n_outer_folds = 5
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=OUTER_SEED)

    for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(dataset_indices, outer_binned_gaps)):
        logger.info(f"\n=== OUTER FOLD {fold+1}/{n_outer_folds} ===")

        # Save fold information in json file
        save_fold(fold, outer_train_idx, outer_test_idx, FIXED_PARAMS['vocab_size'], 'stratified')

        outer_train_indices = np.array(dataset_indices)[outer_train_idx].tolist()
        outer_test_indices = np.array(dataset_indices)[outer_test_idx].tolist()
        outer_train_dataset = Subset(dataset, outer_train_indices)

        best_params = run_tuning_per_fold(outer_train_dataset)

        # Already save the best parameters of the outer fold
        save_fold_result(fold, 0, 0, 0, 0,  best_params)

        # Final train on full outer_train
        logger.info("  Final Training on Outer Train...")
        batch_size = FIXED_PARAMS['batch_size']
        n_epochs = FIXED_PARAMS['n_epochs']
        train_loader = get_dataloader(dataset, outer_train_indices, batch_size)
        test_loader = get_dataloader(dataset, outer_test_indices, batch_size)
        
        model, _, lr = create_model(init_params=best_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses, train_accs = [], []
        for epoch in range(n_epochs):
            train_loss, train_acc = cVAE_train_model(
                epoch, model, train_loader, optimizer,
                lambda m, l, t, b: loss_function(m, l, t, b, beta=best_params['beta']),
                FIXED_PARAMS['use_GPU'], DEVICE, FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len'], token2idx
            )

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            logger.info(f"Outer Epoch: [{epoch}]\tTraining Loss: [{train_loss:.2f}]\tReconstruction accuracy: [{train_acc}]")
        
        # Outer test
        logger.info("  Evaluating on Outer Test...")
        test_loss, test_acc = cVAE_test_model(
            model, test_loader,
            lambda m, l, t, b: loss_function(m, l, t, b, beta=best_params['beta']),
            FIXED_PARAMS['use_GPU'], DEVICE, FIXED_PARAMS['max_atoms'], FIXED_PARAMS['node_vec_len'], token2idx
        )

        # Save the results of the outer fold
        save_fold_result(fold, train_losses, train_accs, test_loss, test_acc, best_params)
        # Save the weights of the model
        save_model(model.state_dict(), fold)
        
        logger.info(f"  Outer Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        logger.info(test_loss, test_acc, best_params)

if __name__ == "__main__":
    logger = setup_logging(DATA_PATH)
    logger.info("Logging system initialized")

    main()