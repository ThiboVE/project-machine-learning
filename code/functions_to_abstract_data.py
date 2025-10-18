import pandas as pd
def extract_qm9_data(dataset, n_samples=None):
    """
    Extract relevant info from QM9 dataset.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        The loaded QM9 dataset.
    n_samples : int, optional
        Number of molecules to extract (for testing). If None, use all.

    Returns
    -------
    pd.DataFrame
        DataFrame with SMILES, HOMO, LUMO, GAP, etc.
    """
    rows = []
    total = len(dataset) if n_samples is None else min(n_samples, len(dataset))
    
    for i in range(total):
        mol = dataset[i]
        
        # Some QM9 versions store SMILES differently
        smiles = getattr(mol, "smiles", None)
        
        # Each molecule has a y tensor with 12 quantum properties
        # indices vary slightly by dataset version, but typically:
        # 7 = HOMO, 8 = LUMO, 9 = gap
        y = mol.y.squeeze().tolist()
        
        row = {
            "index": i,
            "smiles": smiles,
            "num_atoms": mol.x.shape[0],
            "HOMO": y[7],
            "LUMO": y[8],
            "gap": y[9],
        }
        rows.append(row)
        
        # Optional: print progress every 1000 molecules
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total} molecules...")
    
    return pd.DataFrame(rows)
