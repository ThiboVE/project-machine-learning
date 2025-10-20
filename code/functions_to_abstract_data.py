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
      
        y = mol.y.squeeze().tolist()
        
        row = {
            "index": i,
            "smiles": smiles,
            "num_atoms": mol.x.shape[0],
            "mu": y[0],
            "alpha": y[1],
            "HOMO": y[2],
            "LUMO": y[3],
            "gap": y[4],
            "r2": y[5],
            "zpve": y[6],
            "U0": y[7],
            "U298": y[8],
            "H298": y[9],
            "G298": y[10],
            "Cv": y[11],
            "U0_atom": y[12],
            "U298_atom": y[13],
            "H298_atom": y[14],
            "G298_atom": y[15],
            "A": y[16],
            "B": y[17],
            "C": y[18],
            "atoms":mol.z,
            "pos": mol.pos,
            "edge_idx": mol.edge_index,
            "edge_attr": mol.edge_attr
        }
        rows.append(row)
        
        # Optional: print progress every 1000 molecules
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total} molecules...")
    
    return pd.DataFrame(rows)
