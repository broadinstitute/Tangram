"""
Hyperparameter tuning for cell mapping using ray.

Installation:
    pip install ray
    pip install optuna

Example:
    import ray
    from ray import tune
    ray.init()
    metric = ["cell_map_consistency","cell_map_agreement","cell_map_certainty",
              "gene_expr_consistency","gene_expr_correctness"]
    config = {
        "learning_rate" : tune.loguniform(0.001, 1),
        "lambda_g1": tune.uniform(0, 1.0),
        "lambda_r": tune.loguniform(1e-20, 1e-3),
        "lambda_l2": tune.loguniform(1e-20, 1e-3),
        "lambda_neighborhood_g1": tune.uniform(0, 1.0),
        "lambda_ct_islands": tune.uniform(0, 1.0),
        "lambda_getis_ord": tune.uniform(0, 1.0),
    }
    tuner = tg.mapping_hyperparameter_tuning(adata_sc, adata_sp, metric, config)
    results = tuner.get_results().get_dataframe()
"""

import numpy as np
import torch
import scipy
import logging

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from . import mapping_optimizer as mo
from . import utils as ut
from . import spatial_weights as sw

logging.getLogger().setLevel(logging.INFO)

# Benchmarking metrics for repeated run comparisons

def pearson_corr(cube):
    """
    Compute the pearson correlation for the first axis
    Args:
        cube (Array): Values (r,n,j)
    Returns:
        Array: All pairwise Pearson correlations (r x r)
    Example:
        Cell (type) mapping or gene expression prediction consistency: n_runs x n_genes/cell(type)s x n_spots => n_runpairs
    """
    idx = np.tril_indices(cube.shape[0], -1)
    return np.corrcoef(np.reshape(cube,(cube.shape[0],-1)))[idx]

def vote_entropy(pred_probs_cube):
    """
    Compute the normalized vote entropy across the last axis
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Vote entropy values (r,i)
    Example:
        Cell mapping agreement: n_runs x n_cells x n_spots => n_runs x n_cells
    """
    votes_encoded = np.zeros(pred_probs_cube.shape)
    votes = pred_probs_cube.argmax(axis=2)
    for run in range(pred_probs_cube.shape[0]):
        votes_encoded[run,np.arange(pred_probs_cube.shape[1]),votes[run]] = 1
    return scipy.stats.entropy(votes_encoded.mean(axis=0), axis=1) / np.log(pred_probs_cube.shape[2])

def consensus_entropy(pred_probs_cube):
    """
    Compute the normalized consensus entropy across the last axis
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Consensus entropy values (r,i)
    Example:
        Cell mapping certainty: n_runs x n_cells x n_spots => n_runs x n_cells
    """
    consensus_mapping = pred_probs_cube.mean(axis=0)
    return scipy.stats.entropy(consensus_mapping, axis=1) / np.log(pred_probs_cube.shape[2])

# Tuning functions

def train_multiple_Mapper(config,data):
    """
    Wrapper function for hyperparameter tuning, enables to evaluate consistency meassurements by training multiple mappers for each configuration.
    Args:
        config (dict): Hyperparameter setup.
        data (list): Needed data for training.
    """
    from ray import train

    S,G,d_source,d,device,print_each,voxel_weights,ct_encode,neighborhood_filter,spatial_weights,train_genes_idx,val_genes_idx = data
    hyperparameters = {"d_source": d_source}
    for param in list(set(["lambda_d","lambda_g1","lambda_g2","lambda_neighborhood_g1","lambda_r","lambda_l1","lambda_l2","lambda_ct_islands","lambda_getis_ord"]).intersection(set(config.keys()))):
        hyperparameters[param] = config[param]

    learning_rate = 0.1
    if "learning_rate" in config.keys():
        learning_rate = config["learning_rate"]
    num_epochs = 1000
    if "num_epochs" in config.keys():
        num_epochs = config["num_epochs"]

    mapping_matrices = list()
    val_gene_scores = list()
    for run in range(3):
        mapper = mo.Mapper(
            S=S,
            G=G,
            d=d,
            train_genes_idx=train_genes_idx,
            val_genes_idx=val_genes_idx,
            voxel_weights=voxel_weights,
            neighborhood_filter=neighborhood_filter,
            ct_encode=ct_encode,
            spatial_weights=spatial_weights,
            device=device,
            random_state=run,
            **hyperparameters,
        )
        mapping_matrix, training_history = mapper.train(
            print_each=print_each,
            val_each=1,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
        mapping_matrices.append(mapping_matrix)
        val_gene_scores.append(training_history["val_gene_score"][-1])

    cell_mapping_cube = np.array(mapping_matrices)
    gene_expr_cube = np.array([(S[:,val_genes_idx].T @ mapping_matrix) for mapping_matrix in mapping_matrices])
    train.report({"cell_map_consistency" : pearson_corr(cell_mapping_cube).mean(),
                  "cell_map_agreement" : 1-vote_entropy(cell_mapping_cube).mean(),
                  "cell_map_certainty" : 1-consensus_entropy(cell_mapping_cube).mean(),
                  "gene_expr_consistency" : pearson_corr(gene_expr_cube).mean(),
                  "gene_expr_correctness" : np.array(val_gene_scores).mean()})

def mapping_hyperparameter_tuning(
    adata_sc,
    adata_sp,
    metric,
    config,
    tuner_num_samples=2000,
    cv_train_genes=None,
    cv_val_genes=None,
    cluster_label=None,
    device="cpu",
    density_prior='rna_count_based',
):
    """
    Tune Hyperparameters for mapping of cells (`adata_sc`) to spatial spots (`adata_sp`) using a Optuna’s search algorithm.

    Args:
        adata_sc (AnnData): single cell data.
        adata_sp (AnnData): gene spatial data.
        metric (list): metrics used for tuning.
        config (dict): Hyperparameter search space.
        tuner_num_samples (int): Optional. Number of search space samples during optimization. Default is 2000.
        cv_train_genes (list): Optional. Training gene list. Default is None.
        cv_val_genes (list): Optional. Validation gene list. Default is None.
        train_genes_idx (ndarray): Optional. Gene indices used for training from the training gene list.
        val_genes_idx (ndarray): Optional. Gene indices used for validation from the training gene list.
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for `mode=clusters`.
        device (string or torch.device): Optional. Default is 'cpu'.
        density_prior (str, ndarray or None): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If None, the density term is ignored. Default value is 'rna_count_based'.

    Returns:
        ray.tune.Tuner after hyperparameter tuning.
    """
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    if (type(density_prior) is str) and (
        density_prior not in ["rna_count_based", "uniform", None]
    ):
        raise ValueError("Invalid input for density_prior.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    overlap_genes = adata_sc.uns["overlap_genes"]

    if cv_train_genes is None:
        train_genes_idx = list(range(len(overlap_genes)))
    else:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            train_genes_idx = adata_sc[:, overlap_genes].var_names.get_indexer(cv_train_genes)
        else:
            raise ValueError("Given training genes should be subset of two AnnDatas.")

    if cv_val_genes is None:
        val_genes_idx = list(range(len(overlap_genes)))
    else:
        if set(cv_val_genes).issubset(set(adata_sc.uns["training_genes"])):
            val_genes_idx = adata_sc[:, overlap_genes].var_names.get_indexer(cv_val_genes)
        else:
            raise ValueError("Given validation genes should be subset of two AnnDatas.")

    logging.info("Allocate tensors for mapping.")

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, overlap_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, overlap_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, overlap_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, overlap_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    d_source = None
    if density_prior == "rna_count_based":
        density_prior = adata_sp.obs["rna_count_based_density"]
    elif density_prior == "uniform":
        density_prior = adata_sp.obs["uniform_density"]
    d = density_prior

    tuner_device = device
    device = torch.device(device)

    if not set(metric).issubset(set(['cell_map_consistency', 'cell_map_agreement', 'cell_map_certainty', 'gene_expr_consistency', 'gene_expr_correctness'])):
        raise ValueError('Argument "metric" must be a subset of ["cell_map_consistency", "cell_map_agreement", "cell_map_certainty", "gene_expr_consistency", "gene_expr_correctness"]')

    if not set(config.keys()).issubset(set(["learning_rate","num_epochs","lambda_d","lambda_g1","lambda_g2","lambda_neighborhood_g1","lambda_r","lambda_l1","lambda_l2","lambda_ct_islands","lambda_getis_ord"])):
        raise ValueError('Keys of the argument "config" must be a subset of ["learning_rate","num_epochs","lambda_d","lambda_g1","lambda_g2","lambda_neighborhood_g1","lambda_r","lambda_l1","lambda_l2","lambda_ct_islands","lambda_getis_ord"]')

    print_each = None

    if not ray.is_initialized():
        logging.error("ray is not initialized. Run `ray.init()`.")

    voxel_weights = sw.spatial_weights(adata_sp, standardized=True, self_inclusion=True)
    if not cluster_label in adata_sc.obs.keys():
        raise ValueError("cluster_label must be specified for the cell type island extension.")
    neighborhood_filter = sw.spatial_weights(adata_sp, standardized=False, self_inclusion=False)
    ct_encode = ut.one_hot_encoding(adata_sc.obs[cluster_label]).values
    spatial_weights = sw.spatial_weights(adata_sp, standardized=False, self_inclusion=True)

    data = [S,G,d_source,d,device,print_each,voxel_weights,ct_encode,neighborhood_filter,spatial_weights,train_genes_idx,val_genes_idx]

    optuna_search = OptunaSearch(
        metric=metric,
        mode=["max"] * len(metric))

    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train_multiple_Mapper,data=data), {tuner_device: 1}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=tuner_num_samples,
        ),
        param_space=config,
    )
    tuner.fit()
    return tuner
