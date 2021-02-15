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
from . import utils as ut


def pp_adatas(adata_1, adata_2, genes=None):
    """
    Pre-process AnnDatas so that they can be mapped. Specifically:
    - Subset the AnnDatas to `genes` (non-shared genes are removed).
    - Re-order genes in `adata_2` so that they are consistent with those in `adata_1`.
    :param adata_1:
    :param adata_2:
    :param genes:
    List of genes to use. If `None`, all genes are used.
    :return:
    """
    adata_1 = adata_1.copy()
    adata_2 = adata_2.copy()
    adata_1.var_names_make_unique()
    adata_2.var_names_make_unique()
    
    if genes is None:
        # Use all genes
        genes = adata_1.var.index.values
    else:
        genes = list(genes)
    
    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(genes) & set(adata_1.var.index) & set(adata_2.var.index))
    logging.info(f'{len(genes)} marker genes shared by AnnDatas.')

    # Subset adatas on marker genes
    adata_1 = adata_1[:, genes]
    adata_2 = adata_2[:, genes]

    assert adata_2.var.index.equals(adata_1.var.index)
    return adata_1, adata_2


def adata_to_cluster_expression(adata, label, scale=True, add_density=True):
    """
    Convert an AnnData to a new AnnData with cluster expressions. Clusters are based on `label` in `adata.obs`.  The returned AnnData has an observation for each cluster, with the cluster-level expression equals to the average expression for that cluster.
    All annotations in `adata.obs` except `label` are discarded in the returned AnnData.
    If `add_density`, the normalized number of cells in each cluster is added to the returned AnnData as obs.cluster_density.
    :param adata:
    :param label: label for aggregating
    """
    try:
        value_counts = adata.obs[label].value_counts(normalize=True)
    except KeyError as e:
        raise ValueError('Provided label must belong to adata.obs.')
    unique_labels = value_counts.index
    new_obs = pd.DataFrame({label: unique_labels})
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var)

    X_new = np.empty((len(unique_labels), adata.shape[1]))
    for index, l in enumerate(unique_labels):
        if not scale:
            X_new[index] = adata[adata.obs[label] == l].X.mean(axis=0)
        else:
            X_new[index] = adata[adata.obs[label] == l].X.sum(axis=0)
    adata_ret.X = X_new

    if add_density:
        adata_ret.obs['cluster_density'] = adata_ret.obs[label].map(lambda i: value_counts[i])

    return adata_ret

def map_cells_to_space(adata_cells, adata_space, mode='cells', adata_map=None,
                       device='cuda:0', learning_rate=0.1, num_epochs=1000, d=None, cluster_label=None, scale=True, lambda_d=0, lambda_g1=1, lambda_g2=0, lambda_r=0):
    """
        Map single cell data (`adata_1`) on spatial data (`adata_2`). If `adata_map`
        is provided, resume from previous mapping.
        Returns a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes.
        :param mode: Tangram mode. Currently supported: `cell`, `clusters`
        :param lambda_d (float): Optional. Hiperparameter for the density term of the optimizer. Default is 1.
        :param lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
        :param lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 1.
        :param lambda_r (float): Optional. Entropy regularizer for the learned mapping matrix. An higher entropy promotes probabilities of each cell peaked over a narrow portion of space. lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
    """

    if adata_cells.var.index.equals(adata_space.var.index) is False:
        logging.error('Incompatible AnnDatas. Run `pp_adatas().')
        raise ValueError
    if mode == 'clusters' and cluster_label is None:
        raise ValueError('An cluster_label must be specified if mode = clusters.')

    if mode == 'clusters':
        adata_cells = adata_to_cluster_expression(adata_cells, cluster_label, scale, add_density=True)

    logging.info('Allocate tensors for mapping.')
    # Allocate tensors (AnnData matrix can be sparse or not)
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

    if mode == 'cells':
        d = None

    if mode == 'clusters':
        if d is None:
            d = np.ones(G.shape[0])/G.shape[0]

    # Choose device
    device = torch.device(device)  # for gpu

    hyperparameters = {
            'lambda_d': lambda_d,  # KL (ie density) term
            'lambda_g1': lambda_g1,  # gene-voxel cos sim
            'lambda_g2': lambda_g2,  # voxel-gene cos sim
            'lambda_r': lambda_r,  # regularizer: penalize entropy
        }

    # # Init hyperparameters
    # if mode == 'cells':
    #     hyperparameters = {
    #         'lambda_d': 0,  # KL (ie density) term
    #         'lambda_g1': 1,  # gene-voxel cos sim
    #         'lambda_g2': 0,  # voxel-gene cos sim
    #         'lambda_r': 0,  # regularizer: penalize entropy
    #     }
    # elif mode == 'clusters':
    #     hyperparameters = {
    #         'lambda_d': 1,  # KL (ie density) term
    #         'lambda_g1': 1,  # gene-voxel cos sim
    #         'lambda_g2': 0,  # voxel-gene cos sim
    #         'lambda_r': 0,  # regularizer: penalize entropy
    #         'd_source': np.array(adata_cells.obs['cluster_density']) # match sourge/target densities
    #     }
    # else:
    #     raise NotImplementedError

    # Train Tangram
    logging.info('Begin training...')
    mapper = mo.Mapper(
        S=S, G=G, d=d, device=device, adata_map=adata_map,
        **hyperparameters,
    )
    # TODO `train` should return the loss function
    mapping_matrix = mapper.train(
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    logging.info('Saving results..')
    adata_map = sc.AnnData(X=mapping_matrix,
                           obs=adata_cells.obs.copy(),
                           var=adata_space.obs.copy())

    # Annotate cosine similarity of each training gene
    G_predicted = (adata_map.X.T @ S)
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)
    training_genes = list(np.reshape(adata_cells.var.index.values, (-1,)))
    df_cs = pd.DataFrame(cos_sims, training_genes, columns=['train_score'])
    df_cs = df_cs.sort_values(by='train_score', ascending=False)
    adata_map.uns['train_genes_df'] = df_cs

    # Annotate sparsity of each training genes
    ut.annotate_gene_sparsity(adata_cells)
    ut.annotate_gene_sparsity(adata_space)
    adata_map.uns['train_genes_df']['sparsity_sc'] = adata_cells.var.sparsity
    adata_map.uns['train_genes_df']['sparsity_sp'] = adata_space.var.sparsity
    adata_map.uns['train_genes_df']['sparsity_diff'] = adata_space.var.sparsity - adata_cells.var.sparsity

    return adata_map

