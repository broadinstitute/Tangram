"""
    Utility functions to pre- and post-process data for Tangram.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import gzip
import pickle
import scanpy as sc

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from comet_ml import Experiment

from . import mapping_utils as mu

import logging
import warnings

from sklearn.metrics import auc

# import torch
# from torch.nn.functional import cosine_similarity

warnings.filterwarnings("ignore")
logger_ann = logging.getLogger("anndata")
logger_ann.disabled = True


def read_pickle(filename):
    """
        Helper to read pickle file which may be zipped or not.
        Args:
            filename (str): A valid string path.
        Returns:
            The file object.
    """
    try:
        with gzip.open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object
    except OSError:
        with open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object


def annotate_gene_sparsity(adata):
    """
    
    """
    mask = adata.X != 0
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
    gene_sparsity = np.asarray(gene_sparsity)
    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
    adata.var["sparsity"] = gene_sparsity


def get_matched_genes(prior_genes_names, sn_genes_names, excluded_genes=None):
    """
    Given the list of genes in the spatial data and the list of genes in the single nuclei, identifies the subset of
    genes included in both lists and returns the corresponding matching indices.
    Args:
        prior_genes_names (sequence): List of gene names in the spatial data.
        sn_genes_names (sequence): List of gene names in the single nuclei data.
        excluded_genes (sequence): Optional. List of genes to be excluded. These genes are excluded even if present in both datasets.
            If None, no genes are excluded. Default is None.
    Returns:
        A tuple (mask_prior_indices, mask_sn_indices, selected_genes), with:
            mask_prior_indices (list): List of indices for the selected genes in 'prior_genes_names'.
            mask_sn_indices (list): List of indices for the selected genes in 'sn_genes_names'.
            selected_genes (list): List of names of the selected genes.
        For each i, selected_genes[i] = prior_genes_names[mask_prior_indices[i]] = sn_genes_names[mask_sn_indices[i].
    """
    prior_genes_names = np.array(prior_genes_names)
    sn_genes_names = np.array(sn_genes_names)

    mask_prior_indices = []
    mask_sn_indices = []
    selected_genes = []
    if excluded_genes is None:
        excluded_genes = []
    for index, i in enumerate(sn_genes_names):
        if i in excluded_genes:
            continue
        try:
            mask_prior_indices.append(np.argwhere(prior_genes_names == i)[0][0])
            # if no exceptions above:
            mask_sn_indices.append(index)
            selected_genes.append(i)
        except IndexError:
            pass

    assert len(mask_prior_indices) == len(mask_sn_indices)
    return mask_prior_indices, mask_sn_indices, selected_genes


def one_hot_encoding(l, keep_aggregate=False):
    """
    Given a sequence, returns a DataFrame with a column for each unique value in the sequence and a one-hot-encoding.
    Args:
        l (sequence): List to be transformed.
        keep_aggregate: Optional. If True, the output includes an additional column for the original list. Default is False.
    Returns:
        A DataFrame with a column for each unique value in the sequence and a one-hot-encoding, and an additional
            column with the input list if 'keep_aggregate' is True.
            The number of rows are equal to len(l).
    """
    df_enriched = pd.DataFrame({"cl": l})
    for i in l.unique():
        df_enriched[i] = list(map(int, df_enriched["cl"] == i))
    if not keep_aggregate:
        del df_enriched["cl"]
    return df_enriched


def project_cell_annotations(adata_map, annotation="cell_type"):
    """
    Transfer `annotation` from single cell data onto space.
    Args:
        adata_map (AnnData): cell-by-spot AnnData returned by `train` function.
        annotation (str): Cell annotations matrix with shape (number_cells, number_annotations).
    Returns:
        A dataframe with spatial probabilities for each annotation (number_spots, number_annotations)
    """
    df = one_hot_encoding(adata_map.obs[annotation])
    df_ct_prob = adata_map.X.T @ df
    return df_ct_prob


def project_genes(adata_map, adata_sc, cluster_label=None, scale=True):
    """
        Transfer gene expression from the single cell onto space.
        Returns a spot-by-gene AnnData containing spatial gene 
        expression from the single cell data.
    """

    adata_sc = adata_sc.copy()

    # put all var index to lower case to align
    adata_sc.var.index = [g.lower() for g in adata_sc.var.index]

    adata_sc.var_names_make_unique()

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)

    if cluster_label:
        adata_sc = mu.adata_to_cluster_expression(adata_sc, cluster_label, scale=scale)

    if not adata_map.obs.index.equals(adata_sc.obs.index):
        raise ValueError("The two AnnDatas need to have same `obs` index.")
    if hasattr(adata_sc.X, "toarray"):
        adata_sc.X = adata_sc.X.toarray()
    X_space = adata_map.X.T @ adata_sc.X
    adata_ge = sc.AnnData(X=X_space, obs=adata_map.var, var=adata_sc.var)
    training_genes = adata_map.uns["train_genes_df"].index.values
    adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
    return adata_ge


def compare_spatial_geneexp(ad_ge, ad_sp, ad_sc=None):
    """
         Compare gene expression in the generated ans original spatial AnnDatas. 
         Used to compared mapped single cell data to original spatial data.
         Returns a DataFrame with similarity scores between genes.
    """

    ad_ge, ad_sp = mu.pp_adatas(ad_ge, ad_sp)
    annotate_gene_sparsity(ad_sp)

    assert ad_ge.var.index.equals(ad_sp.var.index)

    # Annotate cosine similarity of each training gene
    cos_sims = []

    if hasattr(ad_ge.X, "toarray"):
        X_1 = ad_ge.X.toarray()
    else:
        X_1 = ad_ge.X
    if hasattr(ad_sp.X, "toarray"):
        X_2 = ad_sp.X.toarray()
    else:
        X_2 = ad_sp.X

    for v1, v2 in zip(X_1.T, X_2.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    genes = list(np.reshape(ad_ge.var.index.values, (-1,)))
    df_g = pd.DataFrame(cos_sims, genes, columns=["score"])
    for adata in [ad_ge, ad_sp]:
        if "is_training" in adata.var.keys():
            df_g["is_training"] = adata.var.is_training

    df_g["sparsity_sp"] = ad_sp.var.sparsity

    if ad_sc is not None:
        ad_sc, ad_sp = mu.pp_adatas(ad_sc, ad_sp)
        annotate_gene_sparsity(ad_sc)

        df_g = df_g.merge(
            pd.DataFrame(ad_sc.var["sparsity"]), left_index=True, right_index=True,
        )
        df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
        df_g["sparsity_diff"] = df_g["sparsity_sp"] - df_g["sparsity_sc"]

    df_g = df_g.sort_values(by="score", ascending=False)
    return df_g


def cv_data_gen(ad_sc, ad_sp, mode="loo"):
    """ This function generates cross validation datasets

    Args:
        ad_sc: AnnData, single cell data
        ad_sp: AnnData, gene spatial data
        mode: string, support 'loo' and 'kfold'

    """
    genes_array = np.array(list(set(ad_sc.var.index.values)))

    if mode == "loo":
        cv = LeaveOneOut()
    elif mode == "kfold":
        cv = KFold(n_splits=10)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = genes_array[train_idx]
        test_genes = list(genes_array[test_idx])
        ad_sc_train, ad_sp_train = ad_sc[:, train_genes], ad_sp[:, train_genes]
        yield ad_sc_train, ad_sp_train, test_genes


def cross_val(
    ad_sc,
    ad_sp,
    cluster_label=None,
    mapping_mode="clusters",
    scale=True,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    num_epochs=1000,
    device="cpu",
    learning_rate=0.1,
    mode="loo",
    return_gene_pred=False,
    experiment=None,
    random_state=None,
    verbose=False,
):
    """ This function executes cross validation

    Args:
        ad_sc: AnnData, single cell data
        ad_sp: AnnData, gene spatial data
        lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
        lambda_d (float): Optional. Strength of density regularizer. Default is 0.
        lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer.
        cluster_label: string, the level that the single cell data will be aggregate at, this is only valid for clusters mode mapping
        scale: bool, whether weight input single cell by cluster data by # of cells in cluster, only valid when cluster_label is not None
        mode: string, cross validation mode, 'loo' and 'kfold' supported
        return_gene_pred: bool, if return prediction and true spatial expression data for test gene, only applicable when 'loo' mode is on, default is False
        experiment: experiment object in comet-ml for logging training in comet-ml
    Returns:
        cv_dict: dict, a dictionary contains information of cross validation (hyperparameters, average test score and train score, etc.)
        (df_test_gene_pred, df_test_gene_true): tuple, only return this tuple when return_gene_pred is True and mode is 'loo'
    """

    logger_root = logging.getLogger()
    logger_root.disabled = True
    logger_ann = logging.getLogger("anndata")
    logger_ann.disabled = True

    test_genes_list = []
    test_pred_list = []
    test_score_list = []
    train_score_list = []
    curr_cv_set = 1
    for ad_sc_train, ad_sp_train, test_genes in cv_data_gen(ad_sc, ad_sp, mode):
        # train
        adata_map = mu.map_cells_to_space(
            adata_cells=ad_sc_train,
            adata_space=ad_sp_train,
            mode=mapping_mode,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            cluster_label=cluster_label,
            scale=scale,
            lambda_d=lambda_d,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_r=lambda_r,
            random_state=random_state,
            verbose=False,
        )

        # project on space
        ad_ge = project_genes(
            adata_map, ad_sc, cluster_label=cluster_label, scale=scale
        )

        # retrieve result for test gene (genes X cluster/cell)
        if mode == "loo" and return_gene_pred:
            ad_ge_test = ad_ge[:, test_genes].X.T
            test_pred_list.append(ad_ge_test)

        # output scores
        df_g = compare_spatial_geneexp(ad_ge, ad_sp)
        test_score = df_g[df_g["is_training"] == False]["score"].mean()
        train_score = list(adata_map.uns["training_history"]["main_loss"])[-1]

        # output avg score
        test_genes_list.append(test_genes)
        test_score_list.append(test_score)
        train_score_list.append(train_score)

        if verbose == True:
            msg = "cv set: {}----train score: {:.3f}----test score: {:.3f}".format(
                curr_cv_set, train_score, test_score
            )
            print(msg)

        if experiment:
            experiment.log_metric("test_score_{}".format(curr_cv_set), test_score)
            experiment.log_metric("train_score_{}".format(curr_cv_set), train_score)

        curr_cv_set += 1

    # use nanmean to ignore nan in score list
    avg_test_score = np.nanmean(test_score_list)
    avg_train_score = np.nanmean(train_score_list)

    cv_dict = {
        "mode": mode,
        "weighting": scale,
        "lambda_d": lambda_d,
        "lambda_g1": lambda_g1,
        "lambda_g2": lambda_g2,
        "avg_test_score": avg_test_score,
        "avg_train_score": avg_train_score,
    }

    print("cv avg test score {:.3f}".format(avg_test_score))
    print("cv avg train score {:.3f}".format(avg_train_score))

    if experiment:
        experiment.log_metric("avg test score", avg_test_score)
        experiment.log_metric("avg train score", avg_train_score)

    if mode == "loo" and return_gene_pred:

        # output AnnData for generated spatial data by LOOCV
        ad_ge_cv = sc.AnnData(
            X=np.squeeze(test_pred_list).T,
            obs=ad_sp.obs.copy(),
            var=pd.DataFrame(
                test_score_list,
                columns=["test_score"],
                index=np.squeeze(test_genes_list),
            ),
        )

        return cv_dict, ad_ge_cv

    return cv_dict


def eval_metric(df_all_genes, test_genes=None):
    """
    calculate metrics on given test_genes set for evaluation
    
    Args:
    df_all_genes: pandas dataframe returned by compare_spatial_geneexp(ad_ge, ad_sp, ad_sc); 
                   with "gene names" as the index and "score", "is_training", "sparsity_sc", "sparsity_sp", "sparsity_diff" as the columns
    test_genes: list of test genes, if not given, test_genes will be set to genes where 'is_training' field is False

    Returns: dict with values of each evaluation metric, tuple of auc fitted coordinates and raw coordinates(test_score vs. sparsity_sp coordinates)
    """

    # validate test_genes:
    if test_genes is not None:
        if not set(test_genes).issubset(set(df_all_genes.index.values)):
            raise ValueError(
                "the input of test_genes should be subset of genes of input dataframe"
            )
        test_genes = np.unique(test_genes)

    else:
        test_genes = [df_all_genes[df_all_genes["is_training"] == False].index.values]

    # calculate:
    test_gene_scores = df_all_genes.loc[test_genes]["score"]
    test_gene_sparsity_sp = df_all_genes.loc[test_genes]["sparsity_sp"]
    test_gene_sparsity_sc = df_all_genes.loc[test_genes]["sparsity_sc"]
    test_gene_sparsity_diff = df_all_genes.loc[test_genes]["sparsity_diff"]
    test_score_avg = test_gene_scores.mean()
    train_score_avg = df_all_genes[df_all_genes["is_training"] == True]["score"].mean()

    # g1 metric
    count_test_genes = test_gene_scores.shape[0]
    test_score_sps_sp_g1 = (
        np.sum((test_gene_scores * (1 - test_gene_sparsity_sp))) / count_test_genes
    )
    test_score_sps_sc_g1 = (
        np.sum((test_gene_scores * (1 - test_gene_sparsity_sc))) / count_test_genes
    )
    test_score_sps_diff_g1 = (
        np.sum((test_gene_scores * (1 - np.abs(test_gene_sparsity_diff))))
        / count_test_genes
    )

    # g2 metric
    test_score_sps_sp_g2 = np.sum(
        (test_gene_scores * (1 - test_gene_sparsity_sp))
        / (1 - test_gene_sparsity_sp).sum()
    )
    test_score_sps_sc_g2 = np.sum(
        (test_gene_scores * (1 - test_gene_sparsity_sc))
        / (1 - test_gene_sparsity_sc).sum()
    )
    test_score_sps_diff_g2 = np.sum(
        (test_gene_scores * (1 - np.abs(test_gene_sparsity_diff)))
        / (1 - np.abs(test_gene_sparsity_diff)).sum()
    )

    # tm metric
    # Fit polynomial'
    xs = df_all_genes.score.values
    ys = df_all_genes.sparsity_sp.values
    pol_deg = 3
    pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
    pol_xs = np.linspace(0, 1, 10)  # x linearly spaced
    pol = np.poly1d(pol_cs)  # build polynomial as function
    pol_ys = [pol(x) for x in pol_xs]  # compute polys

    # Compute are under the curve of polynomial
    auc_test_score = auc(pol_xs, pol_ys)

    metric_dict = {
        "avg_test_score": test_score_avg,
        "avg_train_score": train_score_avg,
        "sp_sparsity_weighted_test_score_v1": test_score_sps_sp_g1,
        "sc_sparsity_weighted_test_score_v1": test_score_sps_sc_g1,
        "diff_sparsity_weighted_test_score_v1": test_score_sps_diff_g1,
        "sp_sparsity_weighted_test_score_v2": test_score_sps_sp_g2,
        "sc_sparsity_weighted_test_score_v2": test_score_sps_sc_g2,
        "diff_sparsity_weighted_test_score_v2": test_score_sps_diff_g2,
        "auc_test_score": auc_test_score,
    }

    auc_coordinates = ((pol_xs, pol_ys), (xs, ys))

    return metric_dict, auc_coordinates


# # DEPRECATED
# def transfer_annotations_prob(mapping_matrix, to_transfer):
#     """
#     Transfer cell annotations onto space through a mapping matrix.
#     Args:
#         mapping_matrix (ndarray): Mapping matrix with shape (number_cells, number_spots).
#         to_transfer (ndarray): Cell annotations matrix with shape (number_cells, number_annotations).
#     Returns:
#         A matrix of annotations onto space, with shape (number_spots, number_annotations)
#     """
#     return mapping_matrix.transpose() @ to_transfer


# def transfer_annotations_prob_filter(mapping_matrix, filter, to_transfer):
#     """
#     Transfer cell annotations onto space through a mapping matrix and a filter.
#     Args:
#         mapping_matrix (ndarray): Mapping matrix with shape (number_cells, number_spots).
#         filter (ndarray): Filter with shape (number_cells,).
#         to_transfer (ndarray): Cell annotations matrix with shape (number_cells, number_annotations).
#     Returns:
#         A matrix of annotations onto space, with shape (number_spots, number_annotations).
#     """
#     tt = to_transfer * filter[:, np.newaxis]
#     return mapping_matrix.transpose() @ tt


# def df_to_cell_types(df, cell_types):
#     """
#     Utility function that "randomly" assigns cell coordinates in a voxel to known numbers of cell types in that voxel.
#     Used for deconvolution.
#     Args:
#         df (DataFrame): Columns correspond to cell types.  Each row in the DataFrame corresponds to a voxel and
#             specifies the known number of cells in that voxel for each cell type (int).
#             The additional column 'centroids' specifies the coordinates of the cells in the voxel (sequence of (x,y) pairs).
#         cell_types (sequence): Sequence of cell type names to be considered for deconvolution.
#             Columns in 'df' not included in 'cell_types' are ignored for assignment.
#     Returns:
#         A dictionary <cell type name> -> <list of (x,y) coordinates for the cell type>
#     """
#     df_cum_sums = df[cell_types].cumsum(axis=1)

#     df_c = df.copy()

#     for i in df_cum_sums.columns:
#         df_c[i] = df_cum_sums[i]

#     cell_types_mapped = defaultdict(list)
#     for i_index, i in enumerate(cell_types):
#         for j_index, j in df_c.iterrows():
#             start_ind = 0 if i_index == 0 else j[cell_types[i_index - 1]]
#             end_ind = j[i]
#             cell_types_mapped[i].extend(j['centroids'][start_ind:end_ind].tolist())
#     return cell_types_mapped
