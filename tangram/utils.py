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


def read_pickle(filename):
    """
        Helper to read pickle file which may be zipped or not.
        Args:
            filename (str): A valid string path.
        Returns:
            The file object.
    """
    try:
        with gzip.open(filename, 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object
    except OSError:
        with open(filename, 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object


def annotate_gene_sparsity(adata):
    """
    
    """
    mask = adata.X != 0
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
    gene_sparsity = np.asarray(gene_sparsity)
    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1, ))
    adata.var['sparsity'] = gene_sparsity
    
    
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
    df_enriched = pd.DataFrame({'cl': l})
    for i in l.unique():
        df_enriched[i] = list(map(int, df_enriched['cl'] == i))
    if not keep_aggregate:
        del df_enriched['cl']
    return df_enriched


def project_cell_annotations(adata_map, annotation='cell_type'):
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
    if cluster_label:
        adata_sc = mu.adata_to_cluster_expression(adata_sc, cluster_label, scale=scale)

    if not adata_map.obs.index.equals(adata_sc.obs.index):
        raise ValueError('The two AnnDatas need to have same `obs` index.')
    if hasattr(adata_sc.X, 'toarray'):
        adata_sc.X = adata_sc.X.toarray()
    X_space = adata_map.X.T @ adata_sc.X
    adata_ge = sc.AnnData(X=X_space, obs=adata_map.var, var=adata_sc.var)
    training_genes = adata_map.uns['train_genes_df'].index.values
    adata_ge.var['is_training'] = adata_ge.var.index.isin(training_genes)
    return adata_ge


def compare_spatial_geneexp(adata_space_1, adata_space_2):
    """
         Compare gene expression in the two spatial AnnDatas. 
         Used to compared mapped single cell data to original spatial data.
         Returns a DataFrame with similarity scores between genes.
    """
    
    adata_space_1, adata_space_2 = mu.pp_adatas(adata_space_1, adata_space_2)
    annotate_gene_sparsity(adata_space_1)
    annotate_gene_sparsity(adata_space_2)

    # Annotate cosine similarity of each training gene
    cos_sims = []

    if hasattr(adata_space_1.X, 'toarray'):
        X_1 = adata_space_1.X.toarray()
    else:
        X_1 = adata_space_1.X
    if hasattr(adata_space_2.X, 'toarray'):
        X_2 = adata_space_2.X.toarray()
    else:
        X_2 = adata_space_2.X

    for v1, v2 in zip(X_1.T, X_2.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)
    genes = list(np.reshape(adata_space_1.var.index.values, (-1,)))
    df_g = pd.DataFrame(cos_sims, genes, columns=['score'])
    for adata in [adata_space_1, adata_space_2]:
        if 'is_training' in adata.var.keys():
            df_g['is_training'] = adata.var.is_training
    df_g['sparsity_1'] = adata_space_1.var.sparsity
    df_g['sparsity_2'] = adata_space_2.var.sparsity
    df_g = df_g.sort_values(by='score', ascending=False)
    return df_g


def cv_data_gen(ad_sc, ad_sp, mode='loo'):
    """ This function generates cross validation datasets

    Args:
        ad_sc:
        ad_sp:
        mode: string, support 'loo' and 'kfold'
    """
    genes_array = np.array(list(set(ad_sc.var.index.values)))

    if mode == 'loo':
        cv = LeaveOneOut()
    elif mode == 'kfold':
        cv = KFold(n_splits=10)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = genes_array[train_idx]
        test_genes = genes_array[test_idx]
        ad_sc_train, ad_sp_train = ad_sc[:, train_genes], ad_sp[:, train_genes]
        yield ad_sc_train, ad_sp_train, test_genes

def cross_val(ad_sc,
              ad_sp,
              cluster_label=None,
              scale=True,
              lambda_d=0,
              lambda_g1=1,
              lambda_g2=0,
              lambda_r=0,
              num_epochs=1000,
              device='cpu',
              learning_rate=0.1,
              mode='loo',
              experiment=None
              ):
    """ This function executes cross validation

    Args:
        experiment: experiment object in comet-ml for logging training in comet-ml
    """
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
            mode='clusters',
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            cluster_label=cluster_label,
            scale=scale,
            lambda_d=lambda_d,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_r=lambda_r
        )

        # project on space
        ad_ge = project_genes(adata_map, ad_sc, cluster_label=cluster_label)

        # retrieve result for test gene (genes X cluster/cell)
        ad_ge_test = ad_ge[:,test_genes].X.T

        # output scores
        df_g = compare_spatial_geneexp(ad_ge, ad_sp)
        test_score = df_g[df_g['is_training'] == False]['score'].mean()
        train_score = df_g[df_g['is_training'] == True]['score'].mean()

        # output avg score
        test_genes_list.append(test_genes)
        test_pred_list.append(ad_ge_test)
        test_score_list.append(test_score)
        train_score_list.append(train_score)
        print(
            "cv set: {}----train score: {:.3f}----test score: {:.3f}".format(curr_cv_set, train_score, test_score))
        if experiment:
            experiment.log_metric('test_score_{}'.format(curr_cv_set), test_score)
            experiment.log_metric('train_score_{}'.format(curr_cv_set), train_score)

        curr_cv_set += 1

    avg_test_score = np.mean(test_score_list)
    avg_train_score = np.mean(train_score_list)

    cv_dict = {'mode': mode,
               'weighting': scale,
               'lambda_d': lambda_d, 'lambda_g1': lambda_g1, 'lambda_g2': lambda_g2,
               'avg_test_score': avg_test_score,
               'avg_train_score': avg_train_score}

    test_gene_dict = {'test_gene': test_genes_list,
                      'pred_sp': test_pred_list,
                      'test_score': test_score_list}

    print('cv test score {:.3f}'.format(avg_test_score))
    print('cv train score {:.3f}'.format(avg_train_score))

    if experiment:
        experiment.log_metric("avg test score", np.average(avg_test_score))
        experiment.log_metric("avg train score", np.average(avg_train_score))

    return cv_dict, test_gene_dict


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
            cell_types_mapped[i].extend(j['centroids'][start_ind:end_ind].tolist())
    return cell_types_mapped
