# This code saves the performance (mean error) of all inner folds, for a specific set of hyperparameters
# Each job on the HPC will submit a different hyperparameter set

# Max number of jobs = len(params_combinations) * num_outer_folds - 1 = 243 * 5 - 1 = 1214

import numpy as np
import torch
from library.GCN import *
from pathlib import Path
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import itertools
import os
import json

job_id = int(os.environ.get("SLURM_JOB_ID"))
array_id = int(os.environ.get("PBS_ARRAYID"))

outer_fold_idx = array_id//243
params_idx = array_id%243

def get_param_combinations(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


outer_results = []     # Test MAE per outer fold
best_hyperparams = []  # Best params per fold

param_grid = {
    "batch_size": [256, 512, 1024],
    "hidden_nodes": [16, 32, 64],
    "n_conv_layers": [1, 2, 3],
    "n_hidden_layers": [1, 2, 3],
    "learning_rate": [0.001, 0.005, 0.01]
}

params_list = [x for x in get_param_combinations(param_grid)]
params = params_list[array_id]



# Fix seeds
np.random.seed(10)
torch.manual_seed(10)
use_GPU = False

# Inputs
max_atoms = 30 # fixed value
node_vec_len = 16 # fixed value
n_epochs = 30

data_path = "/data/gent/488/vsc48887/results/rdkit_only_valid_smiles_qm9.pkl"
dataset = GraphData(dataset_path=data_path, max_atoms=max_atoms, 
                        node_vec_len=node_vec_len)
dataset_indices = np.arange(0, len(dataset), 1)

y = np.array([dataset[i][1] for i in range(len(dataset))])

num_bins = 10
gap_bins_outer = pd.qcut(y, q=num_bins, labels=False)

outer_seed = 42
inner_seed = 123

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=inner_seed)

dataset_indices = np.arange(len(dataset))

temp = outer_cv.split(dataset_indices, gap_bins_outer)
outer_fold_splits = [x for x in temp]

train_val_idx = outer_fold_splits[outer_fold_idx][0]
# test_idx = outer_fold_splits[outer_fold_idx][1]

inner_fold_mae = []

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
    model = ChemGCN(
        node_vec_len=node_vec_len,
        node_fea_len=params["hidden_nodes"],
        hidden_fea_len=params["hidden_nodes"],
        n_conv=params["n_conv_layers"],
        n_hidden=params["n_hidden_layers"],
        n_outputs=1,
        p_dropout=0.1,
    )
    if use_GPU:
        model.cuda()

    # Standardizer from training fold only
    outputs = [dataset[i][1] for i in train_idx]
    standardizer = Standardizer(torch.Tensor(outputs))

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = torch.nn.L1Loss()

    # Training for inner CV
    for epoch in range(n_epochs):
        train_model(
            epoch, model, train_loader, optimizer, loss_fn,
            standardizer, use_GPU, max_atoms, node_vec_len
        )

    # Validation for inner CV
    _, val_mae = test_model(
        model, val_loader, loss_fn, standardizer,
        use_GPU, max_atoms, node_vec_len
    )
    inner_fold_mae.append(val_mae)
    



# Define output directory and ensure it exists
output_dir = f'/data/gent/488/vsc48887/results/GCN_model'
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
    'inner_fold_mea_list': inner_fold_mae,
    'params_mean_mae': np.mean(inner_fold_mae)
}

# Save to file as JSON
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)
    




# The following code uses Bayesian hyperparameter tuning, which is too expensive to run in a single job



# # This code saves the performance (mean error) of all inner folds, for a specific set of hyperparameters
# # Each job on the HPC will submit a different hyperparameter set

# import numpy as np
# import torch
# from library.GCN import *
# from pathlib import Path
# from torch.utils.data import DataLoader,SubsetRandomSampler
# from sklearn.model_selection import KFold
# import itertools
# import optuna

# # Fix seeds
# np.random.seed(10)
# torch.manual_seed(10)
# use_GPU = False

# # Inputs
# max_atoms = 30 # fixed value
# node_vec_len = 16 # fixed value
# n_epochs = 30

# data_path = "/data/gent/488/vsc48887/results/rdkit_only_valid_smiles_qm9.pkl"
# dataset = GraphData(dataset_path=data_path, max_atoms=max_atoms, 
#                         node_vec_len=node_vec_len)
# dataset_indices = np.arange(0, len(dataset), 1)

# def objective(trial, train_val_idx):

#     # Sample hyperparameters
#     params = {
#         "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
#         "hidden_nodes": trial.suggest_int("hidden_nodes", 16, 128),
#         "n_conv_layers": trial.suggest_int("n_conv_layers", 1, 5),
#         "n_hidden_layers": trial.suggest_int("n_hidden_layers", 1, 5),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
#     }

#     inner_cv = KFold(n_splits=10, shuffle=True, random_state=10)
#     maes = []

#     for inner_train_idx, inner_val_idx in inner_cv.split(train_val_idx):

#         train_idx = train_val_idx[inner_train_idx]
#         val_idx   = train_val_idx[inner_val_idx]

#         mae = run_inner_training_loop(params, train_idx, val_idx)
#         maes.append(mae)

#     return float(np.mean(maes))

# def run_inner_training_loop(params, train_idx, val_idx):

#     # Build loaders
#     train_loader = DataLoader(
#         dataset,
#         batch_size=params["batch_size"],
#         sampler=SubsetRandomSampler(train_idx),
#         collate_fn=collate_graph_dataset,
#     )
    
#     val_loader = DataLoader(
#         dataset,
#         batch_size=params["batch_size"],
#         sampler=SubsetRandomSampler(val_idx),
#         collate_fn=collate_graph_dataset,
#     )

#     # Build model
#     model = ChemGCN(
#         node_vec_len=node_vec_len,
#         node_fea_len=params["hidden_nodes"],
#         hidden_fea_len=params["hidden_nodes"],
#         n_conv=params["n_conv_layers"],
#         n_hidden=params["n_hidden_layers"],
#         n_outputs=1,
#         p_dropout=0.1,
#     )
#     if use_GPU:
#         model.cuda()

#     # Standardizer
#     outputs = [dataset[i][1] for i in train_idx]
#     standardizer = Standardizer(torch.Tensor(outputs))

#     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
#     loss_fn = torch.nn.L1Loss()

#     # Train
#     for epoch in range(n_epochs):
#         train_model(
#             epoch, model, train_loader, optimizer, loss_fn,
#             standardizer, use_GPU, max_atoms, node_vec_len
#         )

#     # Validate
#     _, val_mae = test_model(
#         model, val_loader, loss_fn, standardizer,
#         use_GPU, max_atoms, node_vec_len
#     )

#     return float(val_mae)

# outer_train_meas = []
# outer_train_loss = []
# outer_test_meas = []
# outer_test_loss = []
# best_hyperparams = []

# inner_cv = KFold(n_splits=10, shuffle=True, random_state=10)
# outer_cv = KFold(n_splits=5, shuffle=True, random_state=10)

# dataset_indices = np.arange(len(dataset))

# for outer_fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(dataset_indices)):
#     print(f"\n===== OUTER FOLD {outer_fold+1}/5 =====")

#     # Store validation mean MAE for each configuration
#     performance_dict = {}
    
#     study = optuna.create_study(direction="minimize")

#     study.optimize(
#         lambda trial: objective(trial, train_val_idx),
#         n_trials=40  # or however many you want
#     )

#     best_params = study.best_trial.params
#     best_hyperparams.append(best_params)

#     print(f">>> Best params for outer fold {outer_fold+1}: {best_params}")
    
#     test_loader = DataLoader(
#     dataset,
#     batch_size=best_params["batch_size"],
#     sampler=SubsetRandomSampler(test_idx),
#     collate_fn=collate_graph_dataset
#     )
#     # ================================
#     # RETRAIN ON FULL TRAIN+VALIDATION
#     # ================================
#     full_train_loader = DataLoader(
#         dataset,
#         batch_size=best_params["batch_size"],
#         sampler=SubsetRandomSampler(train_val_idx),
#         collate_fn=collate_graph_dataset,
#     )

#     model = ChemGCN(
#         node_vec_len=node_vec_len,
#         node_fea_len=best_params["hidden_nodes"],
#         hidden_fea_len=best_params["hidden_nodes"],
#         n_conv=best_params["n_conv_layers"],
#         n_hidden=best_params["n_hidden_layers"],
#         n_outputs=1,
#         p_dropout=0.1,
#     )
#     if use_GPU:
#         model.cuda()

#     outputs = [dataset[i][1] for i in train_val_idx]
#     standardizer = Standardizer(torch.Tensor(outputs))

#     optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
#     loss_fn = torch.nn.L1Loss()

#     for epoch in range(n_epochs):
#         train_loss, train_mae = train_model(
#             epoch, model, full_train_loader, optimizer, loss_fn,
#             standardizer, use_GPU, max_atoms, node_vec_len
#         )

#     # ================================
#     # FINAL TEST ON OUTER FOLD
#     # ================================
#     test_loss, test_mae = test_model(
#         model, test_loader, loss_fn, standardizer,
#         use_GPU, max_atoms, node_vec_len
#     )

#     outer_train_meas.append(train_mae)
#     outer_train_loss.append(train_loss)
#     outer_test_meas.append(test_mae)
#     outer_test_loss.append(test_loss)
#     print(f"===== Outer Fold {outer_fold+1} Test MAE: {test_mae:.4f} =====")
    
    
# # To do:
# # Make sure HPC contains the packages
# # Save results to json instead of printing
# # Make the file work per outer fold
# # Use stratified K-fold