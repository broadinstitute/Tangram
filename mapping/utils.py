"""
This module includes a set of utility functions to prepare and post-process data for Tangram.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import gzip
import pickle


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
