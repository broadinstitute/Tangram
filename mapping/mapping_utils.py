"""
    Mapping helpers
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import logging

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

from . import mapping_optimizer as mo



def prepare_adatas_cell_space(adata_cells, adata_space, marker_genes=None):
    """
        Return `adata_cells` and `adata_space` ready to be mapped.
        Returned adatas have same genes (chosen from `marker_genes`).

        Spatial data needs to be already in ROI
        scRNA-seq data needs to be in library-size corrected raw counts.
    """

    if marker_genes is None:
        # Use all genes
        marker_genes = adata_cells.var.index.values
    else:
        marker_genes = list(marker_genes)

    # Refine `marker_genes` so that they are shared by both adatas
    mask = adata_cells.var.index.isin(marker_genes)
    marker_genes = adata_cells.var[mask].index.values
    mask = adata_space.var.index.isin(marker_genes)
    marker_genes = adata_space.var[mask].index.values
    logging.info(f'{len(marker_genes)} marker genes shared by AnnDatas.')

    # Subset adatas on marker genes
    mask = adata_cells.var.index.isin(marker_genes)
    adata_cells = adata_cells[:, mask]
    mask = adata_space.var.index.isin(marker_genes)
    adata_space = adata_space[:, mask]
    assert adata_space.n_vars == adata_cells.n_vars

    # re-order spatial adata to match gene order in single cell adata
    adata_space = adata_space[:, adata_cells.var.index.values]
    assert adata_space.var.index.equals(adata_cells.var.index)

    return adata_cells, adata_space


def map_cells_to_space(adata_cells, adata_space, mode='simple',
                      device='cuda:0', learning_rate=0.1, num_epochs=1000):
    """
        Map single cell data (`adata_cells`) on spatial data (`adata_space`).
        Returns a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes.
    """
    
    logging.info('Allocate tensors for mapping.')

    # AnnData matrix can be sparse or not
    if isinstance(adata_cells.X, csc_matrix) or isinstance(adata_cells.X, csr_matrix):
        S = np.array(adata_cells.X.toarray(), dtype='float32')
    elif isinstance(adata_cells.X, np.ndarray):
        S = np.array(adata_cells.X, dtype='float32')
    else:
        X_type = type(adata_cells.X)
        logging.error('AnnData X has unrecognized type: {}'.format(X_type))
        raise NotImplementedError
    
    if isinstance(adata_space.X, csc_matrix) or isinstance(adata_space.X, csr_matrix):
        G = np.array(adata_space.X.toarray(), dtype='float32')
    elif isinstance(adata_space.X, np.ndarray):
        G = np.array(adata_space.X, dtype='float32')
    else:
        X_type = type(adata_space.X)
        logging.error('AnnData X has unrecognized type: {}'.format(X_type))
        raise NotImplementedError

    d = np.zeros(adata_space.n_obs)
    device = torch.device(device)  # for gpu

    if mode == 'simple':
        hyperparameters = {
            'lambda_d': 0,  # KL (ie density) term
            'lambda_g1': 1,  # gene-voxel cos sim
            'lambda_g2': 0,  # voxel-gene cos sim
            'lambda_r': 0,  # regularizer: penalize entropy
        }
    else:
        raise NotImplementedError

    mapper = mo.Mapper(
        S=S, G=G, d=d, device=device,
        **hyperparameters,
    )

    logging.info('Begin training...')
    # TODO `train` should return the loss function
    mapping_matrix = mapper.train(
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    logging.info('Saving results..')
    adata_map = sc.AnnData(X=mapping_matrix,
                           obs=adata_cells.obs.copy(),
                           var=adata_space.obs.copy())

    return adata_map



