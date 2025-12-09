# This code saves the performance (mean error) of all outer folds
# Each job on the HPC will train a different outer fold

# Max number of jobs = num_outer_folds = 5

import numpy as np
import torch
from library.GCN import *
from library.general_functions import *
from pathlib import Path
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import itertools
import os
import json

job_id = int(os.environ.get("SLURM_JOB_ID"))
array_id = int(os.environ.get("PBS_ARRAYID"))

outer_fold_idx = array_id

input_dir = f'/data/gent/vo/000/gvo00004/vsc48887/machine_learning/results/GCN_model'
df = json_folder_to_dataframe(input_dir,single_dict=True)

df_outer_fold = df[df['outer_fold_idx']==outer_fold_idx]

params_mean_mae = list(df_outer_fold['params_mean_mae'])
params_lowest_mean_idx = params_mean_mae.index(min(params_mean_mae))

best_batch_size = list(df_outer_fold['batch_size'])[params_lowest_mean_idx]
best_hidden_nodes = list(df_outer_fold['hidden_nodes'])[params_lowest_mean_idx]
best_n_conv_layers = list(df_outer_fold['n_conv_layers'])[params_lowest_mean_idx]
best_n_hidden_layers = list(df_outer_fold['n_hidden_layers'])[params_lowest_mean_idx]
best_learning_rate = list(df_outer_fold['learning_rate'])[params_lowest_mean_idx]

# Inputs
max_atoms = 30 # fixed value
node_vec_len = 16 # fixed value
n_epochs = 50
use_GPU = False

# Get dataset and outer fold indices (make sure seeds align with the inner folds code)
data_path = "/data/gent/vo/000/gvo00004/vsc48887/machine_learning/results/rdkit_only_valid_smiles_qm9.pkl"
dataset = GraphData(dataset_path=data_path, max_atoms=max_atoms, 
                        node_vec_len=node_vec_len)
dataset_indices = np.arange(0, len(dataset), 1)

y = np.array([float(dataset[i][1]) for i in range(len(dataset))])

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
test_idx = outer_fold_splits[outer_fold_idx][1]

# Training on train and validation data for outer fold
full_train_loader = DataLoader(
    dataset,
    batch_size=best_batch_size,
    sampler=SubsetRandomSampler(train_val_idx),
    collate_fn=collate_graph_dataset,
)

test_loader = DataLoader(
    dataset,
    batch_size=best_batch_size,
    sampler=SubsetRandomSampler(test_idx),
    collate_fn=collate_graph_dataset
)

model = ChemGCN(
    node_vec_len=node_vec_len,
    node_fea_len=best_hidden_nodes,
    hidden_fea_len=best_hidden_nodes,
    n_conv=best_n_conv_layers,
    n_hidden=best_n_hidden_layers,
    n_outputs=1,
    p_dropout=0.1,
)
if use_GPU:
    model.cuda()

outputs = [float(dataset[i][1]) for i in train_val_idx]
standardizer = Standardizer(torch.Tensor(outputs))

optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate)
loss_fn = torch.nn.L1Loss()

train_losses = []
train_maes = []

for epoch in range(n_epochs):
    train_loss, train_mae = train_model(
        epoch, model, full_train_loader, optimizer, loss_fn,
        standardizer, use_GPU, max_atoms, node_vec_len
    )
    train_losses.append(train_loss)
    train_maes.append(train_mae)

# Test the outer fold
test_loss, test_mae = test_model(
    model, test_loader, loss_fn, standardizer,
    use_GPU, max_atoms, node_vec_len
)

# Define output directory and ensure it exists
output_dir = f'/data/gent/vo/000/gvo00004/vsc48887/machine_learning/results/GCN_model_outer'
os.makedirs(output_dir, exist_ok=True)

# Define output filename
output_file = os.path.join(output_dir, f"outer_fold_{outer_fold_idx}_mae.json")

# Collect relevant data to save
output_data = {
    'outer_fold_idx': outer_fold_idx,
    'n_epochs': n_epochs,
    "batch_size": best_batch_size,
    "hidden_nodes": best_hidden_nodes,
    "n_conv_layers": best_n_conv_layers,
    "n_hidden_layers": best_n_hidden_layers,
    "learning_rate": best_learning_rate,
    'test_loss': test_loss,
    'test_mae': test_mae,
    'train_losses': train_losses,
    'train_maes': train_maes
}

# Save to file as JSON
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)


