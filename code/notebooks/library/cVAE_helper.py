from torch.utils.data import DataLoader,SubsetRandomSampler, Subset
from library.cVAE import *
from sklearn.model_selection import StratifiedKFold
from library.GCN import GraphData, collate_graph_dataset
from typing import Iterable
from pathlib import Path
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import json


def loss_function(model, logits, targets, batch_size, beta=1):
    recon_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss_recon = recon_loss_fn(logits, targets)
    
    kl_loss = -0.5 * torch.sum(1 + model.z_logvar - model.z_mean.pow(2) - model.z_logvar.exp()) / batch_size
    loss = loss_recon + beta * kl_loss

    return loss


def padding(smiles, token2idx, device):
    # find max length
    max_seq_len = max(s.size(0) for s in smiles)

    # pad with a PAD token index
    pad_idx = token2idx['<PAD>']

    padded_smiles = torch.full((len(smiles), max_seq_len), pad_idx, dtype=torch.long, device=device)

    for i, seq in enumerate(smiles):
        padded_smiles[i, :seq.size(0)] = seq
    
    return padded_smiles


def smiles_to_idxs(smiles, token2idx, device):
    modified_smiles = [] # [batch_size, seq_len]
    for smile in smiles:
        # Add the start and end tokens to the smiles
        char_list = ['<STR>'] + list(smile) + ['<END>']
        # Convert tokens to indices
        vocab_idx_list = torch.as_tensor([token2idx[ch] for ch in char_list], device=device)
        modified_smiles.append(vocab_idx_list)
    
    padded_smiles = padding(modified_smiles, token2idx, device=device)
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


def VAE_train_model(
    epoch,
    model,
    training_dataloader,
    optimizer,
    loss_fn,
    use_GPU,
    device,
    max_atoms,
    node_vec_len,
    token2idx
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
        
        padded_smiles = smiles_to_idxs(dataset[2], token2idx, device)
        batch_size, max_seq_len = padded_smiles.size()

        # Reshape inputs
        node_mat = node_mat.reshape(batch_size, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(batch_size, max_atoms, max_atoms)

        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            model_input = (node_mat.to(device), adj_mat.to(device), padded_smiles.to(device), gap.to(device))
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


def VAE_test_model(
    model,
    test_dataloader,
    loss_fn,
    use_GPU,
    device,
    max_atoms,
    node_vec_len,
    token2idx
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
        padded_smiles = smiles_to_idxs(dataset[2], token2idx, device)
        batch_size, _ = padded_smiles.size()

        # Reshape inputs
        node_mat = node_mat.reshape(batch_size, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(batch_size, max_atoms, max_atoms)

        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            model_input = (node_mat.to(device), adj_mat.to(device), padded_smiles.to(device), gap.to(device))
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


def create_model(params, fixed_params, device):
    # Extract for model construction
    lr = params['learning_rate']
    latent_dim = params['latent_dim']
    n_hidden = params['n_hidden']
    gru_dim = params['gru_dim']
    embedding_dim = params.get('embedding_dim', 8)  # Default if not tuned
    n_conv_layers = params.get('n_conv_layers', 3)
    n_hidden_layers = params.get('n_hidden_layers', 2)
    n_gru_layers = params.get('n_gru_layers', 2)
    n_fc_layers = params.get('n_fc_layers', 2)
    p_dropout = params.get('p_dropout', 0.1)
    teacher_forcing_ratio = params.get('teacher_forcing_ratio', 0.5)
    gcn_hidden_nodes = n_hidden + 1  # Derived
    
    # Build components
    encoder = GCN_Encoder(
        node_vec_len=fixed_params['node_vec_len'],
        node_fea_len=n_hidden,
        hidden_fea_len=n_hidden,
        n_conv=n_conv_layers,
        n_hidden=n_hidden_layers,
        n_outputs=1,
        p_dropout=p_dropout
    )
    
    decoder = GRU_Decoder(
        vocab_size=fixed_params['vocab_size'],
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        hidden_size=gru_dim,
        n_gru_layers=n_gru_layers,
        n_fc_layers=n_fc_layers
    ).to(device)
    
    model = cVAE(
        encoder=encoder,
        decoder=decoder,
        device=device,
        n_gcn_hidden_dim=gcn_hidden_nodes,
        n_gru_hidden_dim=gru_dim,
        latent_dim=latent_dim,
        vocab_size=fixed_params['vocab_size'],
        embedding_dim=embedding_dim,
        teacher_forcing_ratio=teacher_forcing_ratio
    ).to(device)
    
    return model, params, lr


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