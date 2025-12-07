from sklearn.model_selection import StratifiedKFold
from library.GCN import GraphData
from pathlib import Path
import numpy as np
import json
import pickle  # For saving subsets if needed; JSON for indices
import pandas as pd

SEED = 42

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
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

def main():
    # Load dataset
    main_path = Path.cwd().parents[0]
    data_path = main_path / "data" / "RDKit" / "rdkit_only_valid_smiles_qm9.pkl"
    dataset = GraphData(dataset_path=data_path, max_atoms=30, node_vec_len=16)

    gaps = np.array(dataset.outputs)  # For binning
    smiles_list = dataset.smiles
    vocab_list, vocab_size = get_vocab(smiles_list)  # Assume your get_vocab here
    token2idx = {tok: idx for idx, tok in enumerate(vocab_list)}

    dataset_indices = np.arange(len(dataset))
    outer_binned_gaps = make_stratified_bins(gaps)

    n_outer_folds = 5
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=SEED)

    output_dir = Path('../data/cvae_folds')
    output_dir.mkdir(exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(dataset_indices, outer_binned_gaps)):
        fold_data = {
            'fold_id': fold,
            'train_indices': train_idx.tolist(),  # Global indices for Subset
            'test_indices': test_idx.tolist(),
            'n_samples_train': len(train_idx),
            'n_samples_test': len(test_idx),
            'vocab_size': vocab_size  # For consistency
        }
        fold_file = output_dir / f'fold_{fold}_data.json'
        with open(fold_file, 'w') as f:
            json.dump(fold_data, f)
        print(f"Saved fold {fold} to {fold_file}")

if __name__ == "__main__":
    main()