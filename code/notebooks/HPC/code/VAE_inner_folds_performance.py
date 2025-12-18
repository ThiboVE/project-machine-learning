# This code saves the performance (mean error) of all inner folds, for a specific set of hyperparameters
# Each job on the HPC will submit a different hyperparameter set

# Max number of jobs = len(params_combinations) * num_outer_folds - 1 = 2916 * 2

import numpy as np
import torch
from library.cVAE import *
from library.cVAE_helper import *
from library.general_functions import *
from pathlib import Path
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import itertools
import os
import json

job_id = int(os.environ.get("SLURM_JOB_ID"))
array_id = int(os.environ.get("PBS_ARRAYID"))

outer_fold_idx = 0
params_idx = array_id

def get_param_combinations(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


outer_results = []     # Test MAE per outer fold
best_hyperparams = []  # Best params per fold

param_grid = {
    "batch_size": [256, 512],
    "hidden_nodes": [16, 32, 64],
    "learning_rate": [0.001, 0.005, 0.01],
    'beta': [0.1,1,10],
    'latent_dim' : [8,32,64],
    'gru_dim': [16,32,64],
    "n_gru_layers": [1, 2],
    'embedding_dim': [8,16,24],
    'n_epochs': [20,30]
}

params_list = [x for x in get_param_combinations(param_grid)]
params = params_list[params_idx]



# Fix seeds
np.random.seed(10)
torch.manual_seed(10)
use_GPU = False

# Inputs
fixed_params = {'node_vec_len': 16, 'vocab_size': 24}
max_atoms = 30 # fixed value

data_path = "/data/gent/vo/000/gvo00004/vsc48887/machine_learning/results/rdkit_only_valid_smiles_qm9.pkl"
dataset = GraphData(dataset_path=data_path, max_atoms=max_atoms, 
                        node_vec_len=fixed_params['node_vec_len'])
dataset_indices = np.arange(0, len(dataset), 1)

y = np.array([float(dataset[i][1]) for i in range(len(dataset))])

num_bins = 10
gap_bins_outer = pd.qcut(y, q=num_bins, labels=False)

outer_seed = 154
inner_seed = 5165

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=inner_seed)

dataset_indices = np.arange(len(dataset))

temp = outer_cv.split(dataset_indices, gap_bins_outer)
outer_fold_splits = [x for x in temp]

train_val_idx = outer_fold_splits[outer_fold_idx][0]
# test_idx = outer_fold_splits[outer_fold_idx][1]

inner_folds_accuracy = []

y_train_val = gap_bins_outer[train_val_idx]

for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_val_idx, y_train_val)):

    train_idx = train_val_idx[inner_train_idx]
    val_idx = train_val_idx[inner_val_idx]

    # Build loaders
    train_loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SubsetRandomSampler(train_idx),
        collate_fn=collate_graph_dataset,
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SubsetRandomSampler(val_idx),
        collate_fn=collate_graph_dataset,
    )

    # Build new model
    model, _, _ = create_model(params,fixed_params, 'cpu')

    if use_GPU:
        model.cuda()

    # # Standardizer from training fold only
    # outputs = [float(dataset[i][1]) for i in train_idx]
    # standardizer = Standardizer(torch.Tensor(outputs))

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Training for inner CV
    for epoch in range(params['n_epochs']):
        VAE_train_model(
            epoch, model, train_loader, optimizer, loss_function,
            use_GPU, max_atoms, fixed_params['node_vec_len']
        )

    # Validation for inner CV
    _, val_accuracy = VAE_test_model(
        model, val_loader, loss_function,
        use_GPU, max_atoms, fixed_params['node_vec_len']
    )
    inner_folds_accuracy.append(val_accuracy)
    



# Define output directory and ensure it exists
output_dir = f'/data/gent/vo/000/gvo00004/vsc48887/machine_learning/results/VAE_model_inner'
os.makedirs(output_dir, exist_ok=True)

# Define output filename
output_file = os.path.join(output_dir, f"outer_fold_{outer_fold_idx}_params_{params_idx}.json")

# Collect relevant data to save
output_data = {
    'outer_fold_idx': outer_fold_idx,
    'params_idx': params_idx,
    "batch_size": params['batch_size'],
    "hidden_nodes": params["hidden_nodes"],
    "n_conv_layers": params["n_conv_layers"],
    "n_hidden_layers": params["n_hidden_layers"],
    "learning_rate": params['learning_rate'],
    'beta': params['beta'],
    'latent_dim' : params['latent_dim'],
    'gru_dim': params['gru_dim'],
    "n_gru_layers": params["n_gru_layers"],
    "n_fc_layers": params["n_fc_layers"],
    'embedding_dim': params['embedding_dim'],
    'n_epochs': params['n_epochs'],
    'inner_fold_accuracy_list': inner_folds_accuracy,
    'params_mean_accuracy': np.mean(inner_folds_accuracy)
}

# Save to file as JSON
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)
    


