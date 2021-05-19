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

# from torch.nn.functional import cosine_similarity

logging.getLogger().setLevel(logging.INFO)


def pp_adatas(adata_sc, adata_sp, genes=None):
    """
    Pre-process AnnDatas so that they can be mapped. Specifically:
    - Remove genes that all entries are zero
    - Find the intersection between adata_sc, adata_sp and given marker gene list, save the intersected markers in two adatas
    - Calculate density priors and save it with adata_sp

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): spatial expression data
        genes (List): Optional. List of genes to use. If `None`, all genes are used.
    
    Returns:
        update adata_sc by creating `uns` `training_genes` field 
        update adata_sp by creating `uns` `training_genes` field and creating `obs` `rna_count_based_density` & `uniform_density` field

    """

    # put all var index to lower case to align
    adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
    adata_sp.var.index = [g.lower() for g in adata_sp.var.index]

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if genes is None:
        # Use all genes
        genes = [g.lower() for g in adata_sc.var.index]
    else:
        genes = list(g.lower() for g in genes)

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))
    logging.info(f"{len(genes)} shared marker genes.")

    adata_sc.uns["training_genes"] = genes
    adata_sp.uns["training_genes"] = genes
    logging.info(
        f"training genes list is saved in `uns``training_genes` of both single cell and spatial Anndatas."
    )

    # Calculate uniform density prior as 1/number_of_spots
    rna_count_per_spot = adata_sp.X.sum(axis=1)
    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info(
        f"uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata."
    )

    # Calculate rna_count_based density prior as % of rna molecule count
    rna_count_per_spot = adata_sp.X.sum(axis=1)
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(
        rna_count_per_spot
    )
    logging.info(
        f"rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata."
    )


def adata_to_cluster_expression(adata, cluster_label, scale=True, add_density=True):
    """
    Convert an AnnData to a new AnnData with cluster expressions. Clusters are based on `label` in `adata.obs`.  The returned AnnData has an observation for each cluster, with the cluster-level expression equals to the average expression for that cluster.
    All annotations in `adata.obs` except `label` are discarded in the returned AnnData.
    
    Args:
        adata (AnnData): single cell data
        cluster_label (String): level for aggregating
        scale (bool): Optional. Whether weight input single cell by # of cells in cluster. Default is True.
        add_density (bool): Optional. If True, the normalized number of cells in each cluster is added to the returned AnnData as obs.cluster_density.

    Returns:
        AnnData: aggregated single cell data

    """
    try:
        value_counts = adata.obs[cluster_label].value_counts(normalize=True)
    except KeyError as e:
        raise ValueError("Provided label must belong to adata.obs.")
    unique_labels = value_counts.index
    new_obs = pd.DataFrame({cluster_label: unique_labels})
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)

    X_new = np.empty((len(unique_labels), adata.shape[1]))
    for index, l in enumerate(unique_labels):
        if not scale:
            X_new[index] = adata[adata.obs[cluster_label] == l].X.mean(axis=0)
        else:
            X_new[index] = adata[adata.obs[cluster_label] == l].X.sum(axis=0)
    adata_ret.X = X_new

    if add_density:
        adata_ret.obs["cluster_density"] = adata_ret.obs[cluster_label].map(
            lambda i: value_counts[i]
        )

    return adata_ret


def map_cells_to_space(
    adata_sc,
    adata_sp,
    mode="cells",
    device="cuda:0",
    learning_rate=0.1,
    num_epochs=1000,
    cluster_label=None,
    scale=True,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    lambda_count=1,
    lambda_f_reg=1,
    target_count=None,
    random_state=None,
    verbose=True,
    density_prior=None,
    experiment=None,
):
    """
    Map single cell data (`adata_sc`) on spatial data (`adata_sp`). If `adata_map`
    is provided, resume from previous mapping.

    Args:

        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cluster_label (string): the level that the single cell data will be aggregate at, this is only valid for clusters mode mapping
        mode (string): Optional. Tangram mapping mode. Currently supported: 'cell', 'clusters', 'constrained'. Default is 'clusters'
        scale (bool): Optional. Whether weight input single cell by # of cells in cluster, only valid when cluster_label is not None. Default is True.
        lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 0.
        lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
        lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. An higher entropy promotes probabilities of each cell peaked over a narrow portion of space. lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        device (string or torch.device): Optional. Default is 'cpu'.
        experiment (string): Optional. experiment object in comet-ml for logging training in comet-ml. Defulat is None.
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is True.
        density_prior (ndarray or string): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If not provided, the density term is ignored. 

    Returns:
        a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes.

    """

    # check invalid values for arguments
    if lambda_g1 == 0:
        raise ValueError("lambda_g1 cannot be 0.")

    if density_prior is not None and lambda_d == 0:
        raise ValueError("When density_prior is not None, lambda_d cannot be 0.")

    if mode not in ["clusters", "cells", "constrained"]:
        raise ValueError('Argument "mode" must be "cells" or "clusters"')

    if mode == "clusters" and cluster_label is None:
        raise ValueError("A cluster_label must be specified if mode is 'clusters'.")

    if mode == "constrained" and target_count is None:
        raise ValueError("target_count must be specified if mode is 'constrained'.")

    if mode == "clusters":
        adata_sc = adata_to_cluster_expression(
            adata_sc, cluster_label, scale, add_density=True
        )

    # Check if training_genes key exist/is valid in adatas.uns
    if "training_genes" not in adata_sc.uns.keys():
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if "training_genes" not in adata_sp.uns.keys():
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # get traiing_genes
    training_genes = adata_sc.uns["training_genes"]

    logging.info("Allocate tensors for mapping.")
    # Allocate tensors (AnnData matrix can be sparse or not)

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    # define density_prior if 'rna_count_based' is passed to the density_prior argument:
    if density_prior == "rna_count_based":
        density_prior = adata_sp.obs["rna_count_based_density"]

    # define density_prior if 'uniform' is passed to the density_prior argument:
    elif density_prior == "uniform":
        density_prior = adata_sp.obs["uniform_density"]

    if mode in ["cells", "constrained"]:
        d = density_prior

    if mode == "clusters":
        d = density_prior
        if d is None:
            d = adata_sp.obs["uniform_density"]

    # Choose device
    device = torch.device(device)  # for gpu

    if verbose:
        print_each = 100
    else:
        print_each = None

    if mode in ["cells", "clusters"]:
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
        }

        logging.info(
            "Begin training with {} genes in {} mode...".format(
                len(training_genes), mode
            )
        )
        mapper = mo.Mapper(
            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters,
        )

        # TODO `train` should return the loss function

        mapping_matrix, training_history = mapper.train(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            print_each=print_each,
            experiment=experiment,
        )

    # constrained mode
    elif mode == "constrained":
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
            "lambda_count": lambda_count,
            "lambda_f_reg": lambda_f_reg,
            "target_count": target_count,
        }

        logging.info(
            "Begin training with {} genes in {} mode...".format(
                len(training_genes), mode
            )
        )

        mapper = mo.MapperConstrained(
            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters,
        )

        mapping_matrix, F_out, training_history = mapper.train(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            print_each=print_each,
            experiment=experiment,
        )

    logging.info("Saving results..")
    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    if mode == "constrained":
        adata_map.obs["F_out"] = F_out

    # Annotate cosine similarity of each training gene
    G_predicted = adata_map.X.T @ S
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    # Annotate sparsity of each training genes
    ut.annotate_gene_sparsity(adata_sc)
    ut.annotate_gene_sparsity(adata_sp)
    adata_map.uns["train_genes_df"]["sparsity_sc"] = adata_sc[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_sp"] = adata_sp[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_diff"] = (
        adata_sp[:, training_genes].var.sparsity
        - adata_sc[:, training_genes].var.sparsity
    )

    adata_map.uns["training_history"] = training_history

    return adata_map

