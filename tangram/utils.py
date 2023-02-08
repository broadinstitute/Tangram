"""
    Utility functions to pre- and post-process data for Tangram.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import gzip
import pickle
import scanpy as sc
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

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
    Annotates gene sparsity in given Anndatas. 
    Update given Anndata by creating `var` "sparsity" field with gene_sparsity (1 - % non-zero observations).

    Args:
        adata (Anndata): single cell or spatial data.

    Returns:
        None
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
        keep_aggregate (bool): Optional. If True, the output includes an additional column for the original list. Default is False.

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


def project_cell_annotations(
    adata_map, adata_sp, annotation="cell_type", threshold=0.5
):
    """
    Transfer `annotation` from single cell data onto space. 

    Args:
        adata_map (AnnData): cell-by-spot AnnData returned by `train` function.
        adata_sp (AnnData): spatial data used to save the mapping result.
        annotation (str): Optional. Cell annotations matrix with shape (number_cells, number_annotations). Default is 'cell_type'.
        threshold (float): Optional. Valid for using with adata_map.obs['F_out'] from 'constrained' mode mapping. 
        Cell's probability below this threshold will be dropped. Default is 0.5.
    Returns:
        None.
        Update spatial Anndata by creating `obsm` `tangram_ct_pred` field with a dataframe with spatial prediction for each annotation (number_spots, number_annotations) 
    """

    df = one_hot_encoding(adata_map.obs[annotation])
    if "F_out" in adata_map.obs.keys():
        df_ct_prob = adata_map[adata_map.obs["F_out"] > threshold]

    df_ct_prob = adata_map.X.T @ df
    df_ct_prob.index = adata_map.var.index

    adata_sp.obsm["tangram_ct_pred"] = df_ct_prob
    logging.info(
        f"spatial prediction dataframe is saved in `obsm` `tangram_ct_pred` of the spatial AnnData."
    )


def create_segment_cell_df(adata_sp):
    """
    Produces a Pandas dataframe where each row is a segmentation object, columns reveals its position information.

    Args:
        adata_sp (AnnData): spot-by-gene AnnData structure. Must contain obsm.['image_features']

    Returns:
        None.
        Update spatial AnnData.uns['tangram_cell_segmentation'] with a dataframe: each row represents a segmentation object (single cell/nuclei). Columns are 'spot_idx' (voxel id), and 'y', 'x', 'centroids' to specify the position of the segmentation object.
        Update spatial AnnData.obsm['trangram_spot_centroids'] with a sequence
    """

    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing parameter for tangram deconvolution. Run `sqidpy.im.calculate_image_features`."
        )

    centroids = adata_sp.obsm["image_features"][["segmentation_centroid"]].copy()
    centroids["centroids_idx"] = [
        np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
        for k, i in zip(
            adata_sp.obs.index.values,
            adata_sp.obsm["image_features"]["segmentation_label"],
        )
    ]
    centroids_idx = centroids.explode("centroids_idx")
    centroids_coords = centroids.explode("segmentation_centroid")
    segmentation_df = pd.DataFrame(
        centroids_coords["segmentation_centroid"].to_list(),
        columns=["y", "x"],
        index=centroids_coords.index,
    )
    segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
    segmentation_df.index.set_names("spot_idx", inplace=True)
    segmentation_df.reset_index(
        drop=False, inplace=True,
    )

    adata_sp.uns["tangram_cell_segmentation"] = segmentation_df
    adata_sp.obsm["tangram_spot_centroids"] = centroids["centroids_idx"]
    logging.info(
        f"cell segmentation dataframe is saved in `uns` `tangram_cell_segmentation` of the spatial AnnData."
    )
    logging.info(
        f"spot centroids is saved in `obsm` `tangram_spot_centroids` of the spatial AnnData."
    )


def count_cell_annotations(
    adata_map, adata_sc, adata_sp, annotation="cell_type", threshold=0.5,
):
    """
    Count cells in a voxel for each annotation.
    
    Args:
        adata_map (AnnData): cell-by-spot AnnData returned by `train` function.
        adata_sc (AnnData): cell-by-gene AnnData.
        adata_sp (AnnData): spatial AnnData data used to save the mapping result.
        annotation (str): Optional. Cell annotations matrix with shape (number_cells, number_annotations). Default is 'cell_type'.
        threshold (float): Optional. Valid for using with adata_map.obs['F_out'] from 'constrained' mode mapping. 
                           Cell's probability below this threshold will be dropped. Default is 0.5.
    
    Returns:
        None.
        Update spatial AnnData by creating `obsm` `tangram_ct_count` field which contains a dataframe that each row is a spot and each column has the cell count for each cell annotation (number_spots, number_annotations).
    
    """

    if "spatial" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing spatial information in AnnDatas. Please make sure coordinates are saved with AnnData.obsm['spatial']"
        )

    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing parameter for tangram deconvolution. Run `sqidpy.im.calculate_image_features`."
        )

    if (
        "tangram_cell_segmentation" not in adata_sp.uns.keys()
        or "tangram_spot_centroids" not in adata_sp.obsm.keys()
    ):
        raise ValueError(
            "Missing parameter for tangram deconvolution. Run `create_segment_cell_df`."
        )

    xs = adata_sp.obsm["spatial"][:, 1]
    ys = adata_sp.obsm["spatial"][:, 0]
    cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

    df_segmentation = adata_sp.uns["tangram_cell_segmentation"]
    centroids = adata_sp.obsm["tangram_spot_centroids"]

    # create a dataframe
    df_vox_cells = df_vox_cells = pd.DataFrame(
        data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
        index=list(adata_sp.obs.index),
    )

    # get the most probable voxel for each cell
    resulting_voxels = np.argmax(adata_map.X, axis=1)

    # create a list with filtered cells and the voxels where they have been placed with the
    # highest probability a cell i is filtered if F_i > threshold'
    if "F_out" in adata_map.obs.keys():
        filtered_voxels_to_types = [
            (j, adata_sc.obs[annotation][k])
            for i, j, k in zip(
                adata_map.obs["F_out"], resulting_voxels, range(len(adata_sc))
            )
            if i > threshold
        ]

        vox_ct = filtered_voxels_to_types

    else:
        vox_ct = list(zip(resulting_voxels, adata_sc.obs[annotation]))

    df_classes = one_hot_encoding(adata_sc.obs[annotation])
    for index, i in enumerate(df_classes.columns):
        df_vox_cells[i] = 0

    for k, v in vox_ct:
        df_vox_cells.iloc[k, df_vox_cells.columns.get_loc(v)] += 1

    adata_sp.obsm["tangram_ct_count"] = df_vox_cells
    logging.info(
        f"spatial cell count dataframe is saved in `obsm` `tangram_ct_count` of the spatial AnnData."
    )


def deconvolve_cell_annotations(adata_sp, filter_cell_annotation=None):
    """
    Assigns cell annotation to each segmented cell. Produces an AnnData structure that saves the assignment in its obs dataframe.

    Args:
        adata_sp (AnnData): Spatial AnnData structure.
        filter_cell_annotation (sequence): Optional. Sequence of cell annotation names to be considered for deconvolution. Default is None. When no values passed, all cell annotation names in adata_sp.obsm["tangram_ct_pred"] will be used.

    Returns:
        AnnData: Saves the cell annotation assignment result in its obs dataframe where each row representing a segmentation object, column 'x', 'y', 'centroids' contain its position and column 'cluster' is the assigned cell annotation.
    """

    if (
        "tangram_ct_count" not in adata_sp.obsm.keys()
        or "tangram_cell_segmentation" not in adata_sp.uns.keys()
    ):
        raise ValueError("Missing tangram parameters. Run `count_cell_annotations`.")

    segmentation_df = adata_sp.uns["tangram_cell_segmentation"]

    if filter_cell_annotation is None:
        filter_cell_annotation = pd.unique(
            list(adata_sp.obsm["tangram_ct_pred"].columns)
        )
    else:
        filter_cell_annotation = pd.unique(filter_cell_annotation)

    df_vox_cells = adata_sp.obsm["tangram_ct_count"]
    cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)
    df_list = []
    for k in cell_types_mapped.keys():
        df = pd.DataFrame({"centroids": np.array(cell_types_mapped[k], dtype="object")})
        df["cluster"] = k
        df_list.append(df)
    cluster_df = pd.concat(df_list, axis=0)
    cluster_df.reset_index(inplace=True, drop=True)

    merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
    merged_df.drop(columns="spot_idx", inplace=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(inplace=True, drop=True)

    adata_segment = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    adata_segment.obsm["spatial"] = merged_df[["y", "x"]].to_numpy()
    adata_segment.uns = adata_sp.uns

    return adata_segment


def project_genes(adata_map, adata_sc, cluster_label=None, scale=True):
    """
    Transfer gene expression from the single cell onto space.

    Args:
        adata_map (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cluster_label (AnnData): Optional. Should be consistent with the 'cluster_label' argument passed to `map_cells_to_space` function.
        scale (bool): Optional. Should be consistent with the 'scale' argument passed to `map_cells_to_space` function.

    Returns:
        AnnData: spot-by-gene AnnData containing spatial gene expression from the single cell data.
    """

    # put all var index to lower case to align
    adata_sc.var.index = [g.lower() for g in adata_sc.var.index]

    # make varnames unique for adata_sc
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
    adata_ge = sc.AnnData(
        X=X_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
    )
    training_genes = adata_map.uns["train_genes_df"].index.values
    adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
    return adata_ge


def compare_spatial_geneexp(adata_ge, adata_sp, adata_sc=None, genes=None):
    """ Compares generated spatial data with the true spatial data

    Args:
        adata_ge (AnnData): generated spatial data returned by `project_genes`
        adata_sp (AnnData): gene spatial data
        adata_sc (AnnData): Optional. When passed, sparsity difference between adata_sc and adata_sp will be calculated. Default is None.
        genes (list): Optional. When passed, returned output will be subset on the list of genes. Default is None.

    Returns:
        Pandas Dataframe: a dataframe with columns: 'score', 'is_training', 'sparsity_sp'(spatial data sparsity). 
                          Columns - 'sparsity_sc'(single cell data sparsity), 'sparsity_diff'(spatial sparsity - single cell sparsity) returned only when adata_sc is passed.
    """

    logger_root = logging.getLogger()
    logger_root.disabled = True

    # Check if training_genes/overlap_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_ge.uns.keys())):
        raise ValueError(
            "Missing tangram parameters. Use `project_genes()` to get adata_ge."
        )

    assert list(adata_sp.uns["overlap_genes"]) == list(adata_ge.uns["overlap_genes"])

    if genes is None:
        overlap_genes = adata_ge.uns["overlap_genes"]
    else:
        overlap_genes = genes

    annotate_gene_sparsity(adata_sp)

    # Annotate cosine similarity of each training gene
    cos_sims = []

    if hasattr(adata_ge.X, "toarray"):
        X_1 = adata_ge[:, overlap_genes].X.toarray()
    else:
        X_1 = adata_ge[:, overlap_genes].X
    if hasattr(adata_sp.X, "toarray"):
        X_2 = adata_sp[:, overlap_genes].X.toarray()
    else:
        X_2 = adata_sp[:, overlap_genes].X

    for v1, v2 in zip(X_1.T, X_2.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_g = pd.DataFrame(cos_sims, overlap_genes, columns=["score"])
    for adata in [adata_ge, adata_sp]:
        if "is_training" in adata.var.keys():
            df_g["is_training"] = adata.var.is_training

    df_g["sparsity_sp"] = adata_sp[:, overlap_genes].var.sparsity

    if adata_sc is not None:
        if not set(["training_genes", "overlap_genes"]).issubset(
            set(adata_sc.uns.keys())
        ):
            raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

        assert list(adata_sc.uns["overlap_genes"]) == list(
            adata_sp.uns["overlap_genes"]
        )
        annotate_gene_sparsity(adata_sc)

        df_g = df_g.merge(
            pd.DataFrame(adata_sc[:, overlap_genes].var["sparsity"]),
            left_index=True,
            right_index=True,
        )
        df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
        df_g["sparsity_diff"] = df_g["sparsity_sp"] - df_g["sparsity_sc"]

    else:
        logging.info(
            "To create dataframe with column 'sparsity_sc' or 'aprsity_diff', please also pass adata_sc to the function."
        )

    if genes is not None:
        df_g = df_g.loc[genes]

    df_g = df_g.sort_values(by="score", ascending=False)
    return df_g


def cv_data_gen(adata_sc, adata_sp, cv_mode="loo"):
    """ Generates pair of training/test gene indexes cross validation datasets

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        mode (str): Optional. support 'loo' and '10fold'. Default is 'loo'.

    Yields:
        tuple: list of train_genes, list of test_genes
    """

    # Check if training_genes key exist/is valid in adatas.uns
    if "training_genes" not in adata_sc.uns.keys():
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if "training_genes" not in adata_sp.uns.keys():
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"]):
        raise ValueError(
            "Unmatched training_genes field in two Anndatas. Run `pp_adatas()`."
        )

    genes_array = np.array(adata_sp.uns["training_genes"])

    if cv_mode == "loo":
        cv = LeaveOneOut()
    elif cv_mode == "10fold":
        cv = KFold(n_splits=10)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = list(genes_array[train_idx])
        test_genes = list(genes_array[test_idx])
        yield train_genes, test_genes


def cross_val(
    adata_sc,
    adata_sp,
    cluster_label=None,
    mode="clusters",
    scale=True,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    lambda_count=1,
    lambda_f_reg=1,
    target_count=None,
    num_epochs=1000,
    device="cuda:0",
    learning_rate=0.1,
    cv_mode="loo",
    return_gene_pred=False,
    density_prior=None,
    random_state=None,
    verbose=False,
):
    """
    Executes cross validation

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cluster_label (str): the level that the single cell data will be aggregate at, this is only valid for clusters mode mapping
        mode (str): Optional. Tangram mapping mode. Currently supported: 'cell', 'clusters', 'constrained'. Default is 'clusters'.
        scale (bool): Optional. Whether weight input single cell by # of cells in cluster, only valid when cluster_label is not None. Default is True.
        lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
        lambda_d (float): Optional. Strength of density regularizer. Default is 0.
        lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        device (str or torch.device): Optional. Default is 'cuda:0'.
        cv_mode (str): Optional. cross validation mode, 'loo' ('leave-one-out') and '10fold' supported. Default is 'loo'.
        return_gene_pred (bool): Optional. if return prediction and true spatial expression data for test gene, only applicable when 'loo' mode is on, default is False.
        density_prior (ndarray or str): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If not provided, the density term is ignored. 
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is False.
    
    Returns:
        cv_dict (dict): a dictionary contains information of cross validation (hyperparameters, average test score and train score, etc.)
        adata_ge_cv (AnnData): predicted spatial data by LOOCV. Only returns when `return_gene_pred` is True and in 'loo' mode.
        test_gene_df (Pandas dataframe): dataframe with columns: 'score', 'is_training', 'sparsity_sp'(spatial data sparsity)
    """

    logger_root = logging.getLogger()
    logger_root.disabled = True
    logger_ann = logging.getLogger("anndata")
    logger_ann.disabled = True

    test_genes_list = []
    test_pred_list = []
    test_score_list = []
    train_score_list = []
    test_df_list = []
    curr_cv_set = 1

    if cv_mode == "loo":
        length = len(list(adata_sc.uns["training_genes"]))
    elif cv_mode == "10fold":
        length = 10

    if mode == "clusters":
        adata_sc_agg = mu.adata_to_cluster_expression(adata_sc, cluster_label, scale)

    for train_genes, test_genes in tqdm(
        cv_data_gen(adata_sc, adata_sp, cv_mode), total=length
    ):
        # train
        adata_map = mu.map_cells_to_space(
            adata_sc=adata_sc,
            adata_sp=adata_sp,
            cv_train_genes=train_genes,
            mode=mode,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            cluster_label=cluster_label,
            scale=scale,
            lambda_d=lambda_d,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_r=lambda_r,
            lambda_count=lambda_count,
            lambda_f_reg=lambda_f_reg,
            target_count=target_count,
            random_state=random_state,
            verbose=False,
            density_prior=density_prior,
        )

        cv_genes = train_genes + test_genes

        # project on space
        adata_ge = project_genes(
            adata_map, adata_sc[:, cv_genes], cluster_label=cluster_label, scale=scale,
        )

        # retrieve result for test gene (genes X cluster/cell)
        if cv_mode == "loo" and return_gene_pred:
            adata_ge_test = adata_ge[:, test_genes].X.T
            test_pred_list.append(adata_ge_test)

        # output test genes dataframe
        if mode == "clusters":
            df_g = compare_spatial_geneexp(adata_ge, adata_sp, adata_sc_agg, cv_genes)
        else:
            df_g = compare_spatial_geneexp(adata_ge, adata_sp, adata_sc, cv_genes)

        test_df = df_g[df_g.index.isin(test_genes)]
        test_score = df_g.loc[test_genes]["score"].mean()
        train_score = np.float(list(adata_map.uns["training_history"]["main_loss"])[-1])

        # output avg score
        test_genes_list.append(test_genes)
        test_score_list.append(test_score)
        train_score_list.append(train_score)
        test_df_list.append(test_df)

        if verbose == True:
            msg = "cv set: {}----train score: {:.3f}----test score: {:.3f}".format(
                curr_cv_set, train_score, test_score
            )
            print(msg)

        curr_cv_set += 1

    # use nanmean to ignore nan in score list
    avg_test_score = np.nanmean(test_score_list)
    avg_train_score = np.nanmean(train_score_list)

    cv_dict = {
        "avg_test_score": avg_test_score,
        "avg_train_score": avg_train_score,
    }

    print("cv avg test score {:.3f}".format(avg_test_score))
    print("cv avg train score {:.3f}".format(avg_train_score))

    if cv_mode == "loo" and return_gene_pred:

        # output df_test_genes dataframe
        test_gene_df = pd.concat(test_df_list, axis=0)

        # output AnnData for generated spatial data by LOOCV
        adata_ge_cv = sc.AnnData(
            X=np.squeeze(test_pred_list).T,
            obs=adata_sp.obs.copy(),
            var=pd.DataFrame(
                test_score_list,
                columns=["test_score"],
                index=np.squeeze(test_genes_list),
            ),
        )

        return cv_dict, adata_ge_cv, test_gene_df

    return cv_dict


def eval_metric(df_all_genes, test_genes=None):
    """
    Compute metrics on given test_genes set for evaluation
    
    Args:
        df_all_genes (Pandas dataframe): returned by compare_spatial_geneexp(adata_ge, adata_sp); 
        test_genes (list): list of test genes, if not given, test_genes will be set to genes where 'is_training' field is False

    Returns:      
        dict with values of each evaluation metric ("avg_test_score", "avg_train_score", "auc_score"), 
        tuple of auc fitted coordinates and raw coordinates(test_score vs. sparsity_sp coordinates)
    """

    # validate test_genes:
    if test_genes is not None:
        if not set(test_genes).issubset(set(df_all_genes.index.values)):
            raise ValueError(
                "the input of test_genes should be subset of genes of input dataframe"
            )
        test_genes = np.unique(test_genes)

    else:
        test_genes = list(
            set(df_all_genes[df_all_genes["is_training"] == False].index.values)
        )

    # calculate:
    test_gene_scores = df_all_genes.loc[test_genes]["score"]
    test_gene_sparsity_sp = df_all_genes.loc[test_genes]["sparsity_sp"]
    test_score_avg = test_gene_scores.mean()
    train_score_avg = df_all_genes[df_all_genes["is_training"] == True]["score"].mean()

    # sp sparsity weighted score
    test_score_sps_sp_g2 = np.sum(
        (test_gene_scores * (1 - test_gene_sparsity_sp))
        / (1 - test_gene_sparsity_sp).sum()
    )

    # tm metric
    # Fit polynomial'
    xs = list(test_gene_scores)
    ys = list(test_gene_sparsity_sp)
    pol_deg = 2
    pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
    pol_xs = np.linspace(0, 1, 10)  # x linearly spaced
    pol = np.poly1d(pol_cs)  # build polynomial as function
    pol_ys = [pol(x) for x in pol_xs]  # compute polys
    
    if pol_ys[0] > 1:
        pol_ys[0] = 1

    # if real root when y = 0, add point (x, 0):
    roots = pol.r
    root = None
    for i in range(len(roots)):
        if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
            root = roots[i]
            break

    if root is not None:
        pol_xs = np.append(pol_xs, root)
        pol_ys = np.append(pol_ys, 0)       
        
    np.append(pol_xs, 1)
    np.append(pol_ys, pol(1))

    # remove point that are out of [0,1]
    del_idx = []
    for i in range(len(pol_xs)):
        if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
            del_idx.append(i)

    pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
    pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]

    # Compute are under the curve of polynomial
    auc_test_score = np.real(auc(pol_xs, pol_ys))

    metric_dict = {
        "avg_test_score": test_score_avg,
        "avg_train_score": train_score_avg,
        "sp_sparsity_score": test_score_sps_sp_g2,
        "auc_score": auc_test_score,
    }

    auc_coordinates = ((pol_xs, pol_ys), (xs, ys))

    return metric_dict, auc_coordinates


# DEPRECATED
def transfer_annotations_prob(mapping_matrix, to_transfer):
    """
    Transfer cell annotations onto space through a mapping matrix.

    Args:
        mapping_matrix (ndarray): Mapping matrix with shape (number_cells, number_spots).
        to_transfer (ndarray): Cell annotations matrix with shape (number_cells, number_annotations).
        
    Returns:
        A matrix of annotations onto space, with shape (number_spots, number_annotations)
    """
    return mapping_matrix.transpose() @ to_transfer


def transfer_annotations_prob_filter(mapping_matrix, filter, to_transfer):
    """
    Transfer cell annotations onto space through a mapping matrix and a filter.
    Args:
        mapping_matrix (ndarray): Mapping matrix with shape (number_cells, number_spots).
        filter (ndarray): Filter with shape (number_cells,).
        to_transfer (ndarray): Cell annotations matrix with shape (number_cells, number_annotations).
    Returns:
        A matrix of annotations onto space, with shape (number_spots, number_annotations).
    """
    tt = to_transfer * filter[:, np.newaxis]
    return mapping_matrix.transpose() @ tt


def df_to_cell_types(df, cell_types):
    """
    Utility function that "randomly" assigns cell coordinates in a voxel to known numbers of cell types in that voxel.
    Used for deconvolution.

    Args:
        df (DataFrame): Columns correspond to cell types.  Each row in the DataFrame corresponds to a voxel and
        specifies the known number of cells in that voxel for each cell type (int).
        The additional column 'centroids' specifies the coordinates of the cells in the voxel (sequence of (x,y) pairs).
        cell_types (sequence): Sequence of cell type names to be considered for deconvolution.
        Columns in 'df' not included in 'cell_types' are ignored for assignment.

    Returns:
        A dictionary <cell type name> -> <list of (x,y) coordinates for the cell type>
    """
    df_cum_sums = df[cell_types].cumsum(axis=1)

    df_c = df.copy()

    for i in df_cum_sums.columns:
        df_c[i] = df_cum_sums[i]

    cell_types_mapped = defaultdict(list)
    for i_index, i in enumerate(cell_types):
        for j_index, j in df_c.iterrows():
            start_ind = 0 if i_index == 0 else j[cell_types[i_index - 1]]
            end_ind = j[i]
            cell_types_mapped[i].extend(j["centroids"][start_ind:end_ind].tolist())
    return cell_types_mapped
