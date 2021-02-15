"""
This module includes plotting utility functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from scipy.stats import entropy
import scanpy as sc
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

from . import utils as ut
from . import mapping_utils as mu


def plot_training_scores(adata_map, bins='auto', alpha=.7):
    """
    
    """
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    df = adata_map.uns['train_genes_df']
    axs_f = axs.flatten()
    
#     axs_f[0].set_title('Training scores for single genes')
    sns.histplot(data=df, y='train_score', bins=10, ax=axs_f[0]);

    axs_f[1].set_title('score vs sparsity (single cells)')
    sns.scatterplot(data=df, y='train_score', x='sparsity_sc', ax=axs_f[1], alpha=alpha)
#     sns.distplot(data=df, x='train_score', bins=bins, ax=axs_f[0]);
    
    axs_f[2].set_title('score vs sparsity (spatial)')
    sns.scatterplot(data=df, y='train_score', x='sparsity_sp', ax=axs_f[2], alpha=alpha)
    
    axs_f[3].set_title('score vs sparsity (sp - sc)')
    sns.scatterplot(data=df, y='train_score', x='sparsity_diff', ax=axs_f[3], alpha=alpha)
    
    plt.tight_layout()

    
def plot_gene_sparsity(adata_1, adata_2, xlabel='adata_1', ylabel='adata_2', s=1, genes=None):
    """
        Compare sparsity of all genes between `adata_1` and `adata_2`.
    """
    logging.info('Pre-processing AnnDatas...')
    adata_1, adata_2 = mu.pp_adatas(adata_1, adata_2, genes=genes)
    logging.info('Annotating sparsity...')
    ut.annotate_gene_sparsity(adata_1)
    ut.annotate_gene_sparsity(adata_2)
    xs = adata_1.var['sparsity'].values
    ys = adata_2.var['sparsity'].values
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1)
    ax.set_xlabel('sparsity (' + xlabel + ')')
    ax.set_ylabel('sparsity (' + ylabel + ')')
    ax.set_title(f'Gene sparsity ({len(xs)} genes)')
    ax.scatter(xs, ys, s=s, marker='x')

    
def ordered_predictions(xs, ys, preds, reverse=False):
    """
    Utility function that orders 2d points based on values associated to each point.
    Args:
        xs: Sequence of x coordinates (floats).
        ys: Sequence of y coordinates (floats).
        ys: Sequence of y coordinates (floats).
        reverse: Optional. A Boolean. False will sort ascending, True will sort descending. Default is False.
    Returns:
        Returns the ordered xs, ys, preds.
    """
    assert len(xs) == len(ys) == len(preds)
    return list(zip(*[(x, y, z) for x, y, z in sorted(zip(xs, ys, preds), key=lambda pair: pair[2], reverse=reverse)]))


def plot_cell_annotation(adata_map, annotation='cell_type', 
                         x='x', y='y', nrows=None, ncols=None,
                         marker_size=5, cmap='viridis', suptitle_add=False):
    """
        Transfer an annotation for a single cell dataset onto space, and visualize
        corresponding spatial probability maps.
        `adata_map`: cell-by-spot-AnnData containing mapping result
        `annotation`: Must be a column in `adata_map.obs`.
        `x`: column name for spots x-coordinates (must be in `adata_map.var`)
        `y`: column name for spots y-coordinates (must be in `adata_map.var`)
    """

    # TODO ADD CHECKS for x and y
    
    df_annotation = ut.project_cell_annotations(adata_map, annotation=annotation)

    if nrows is None or ncols is None:
        ncols = 1
        nrows = len(df_annotation.columns)

    fig, axs = plt.subplots(nrows, ncols, 
                              figsize=(ncols*3, nrows*3), 
                              sharex=True, sharey=True) 

    axs_f = axs.flatten()
    [ax.axis('off') for ax in axs_f]

    if len(df_annotation.columns) > nrows*ncols:
        logging.warning('Number of panels smaller than annotations. Increase `nrows`/`ncols`.')
    
    iterator = zip(df_annotation.columns, range(nrows*ncols))
    for ann, index in iterator:
        xs, ys, preds = ordered_predictions(adata_map.var[x], 
                                            adata_map.var[y], 
                                            df_annotation[ann])
        axs_f[index].scatter(x=xs, y=ys, c=preds, s=marker_size, cmap=cmap)
#         axs_f[index].axis('off')
        axs_f[index].set_aspect(1)
        axs_f[index].set_title(ann)
        
    if suptitle_add is True:
        fig.suptitle(annotation)


def plot_genes(genes, adata_measured, adata_predicted, x='x', y='y', s=5, log=False):
    """

    """
    # TODO: not very elegant and slow as hell
    if isinstance(adata_measured.X, csc_matrix) or isinstance(adata_measured.X, csr_matrix):
        adata_measured.X = adata_measured.X.toarray()
    
    fig, axs = plt.subplots(nrows=len(genes), ncols=2, figsize=(6, len(genes)*3))
    for ix, gene in enumerate(genes):
        xs, ys, vs = ordered_predictions(adata_measured.obs[x], 
                                         adata_measured.obs[y], 
                                         np.array(adata_measured[:, gene].X).flatten())
        if log:
            vs = np.log(1+np.asarray(vs))
        axs[ix, 0].scatter(xs, ys, c=vs, cmap='inferno', s=s)
        axs[ix, 0].set_title(gene + ' (measured)')
        axs[ix, 0].axis('off')
        
        xs, ys, vs = ordered_predictions(adata_predicted.obs[x], 
                                         adata_predicted.obs[y], 
                                         np.array(adata_predicted[:, gene].X).flatten())
        if log:
            vs = np.log(1+np.asarray(vs))
        axs[ix, 1].scatter(xs, ys, c=vs, cmap='inferno', s=s)
        axs[ix, 1].set_title(gene + ' (predicted)')
        axs[ix, 1].axis('off')
    
    
def quick_plot_gene(gene, adata, x='x', y='y', s=50, log=False):
    """
    Utility function to quickly plot a gene in a AnnData structure ordered by intensity of the gene signal.
    Args:
        gene (str): Gene name.
        adata: AnnData structure.
        x: Optional. Name for the first coordinate in AnnData.obs. Default is 'x'.
        y: Optional. Name for the second coordinate in AnnData.obs. Default is 'y'.
        log: Optional. Whether to apply the log before plotting. Default is False.
    """
    xs, ys, vs = ordered_predictions(adata.obs[x], adata.obs[y], np.array(adata[:, gene].X).flatten())
    if log:
        vs = np.log(1+np.asarray(vs))
    plt.scatter(xs, ys, c=vs, cmap='viridis', s=s)


def plot_annotation_entropy(adata_map, annotation='cell_type'):
    """
        
    """
    qk = np.ones(shape=(adata_map.n_obs, adata_map.n_vars))
    adata_map.obs['entropy'] = entropy(adata_map.X, base=adata_map.X.shape[1], axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.set_ylim(0, 1)
    sns.boxenplot(x=annotation, y="entropy", data=adata_map.obs, ax=ax);
    plt.xticks(rotation=30);


# Colors used in the manuscript for deterministic assignment.
mapping_colors = {'L6 CT': (0.19215686274509805, 0.5098039215686274, 0.7411764705882353),
                  'L6 IT': (0.4196078431372549, 0.6823529411764706, 0.8392156862745098),
                  'L5/6 NP': (0.6196078431372549, 0.792156862745098, 0.8823529411764706),
                  'L6b': '#0000c2ff',
                  'L2/3 IT': (0.9019607843137255, 0.3333333333333333, 0.050980392156862744),
                  'L5 IT': (0.19215686274509805, 0.6392156862745098, 0.32941176470588235),
                  'L5 ET': (0.4549019607843137, 0.7686274509803922, 0.4627450980392157),
                  'Oligo': (0.4588235294117647, 0.4196078431372549, 0.6941176470588235),
                  'Vip': (0.6196078431372549, 0.6039215686274509, 0.7843137254901961),
                  'Astro': '#ffdd55ff',
                  'Micro-PVM': '#000000ff',
                  'Pvalb': (0.38823529411764707, 0.38823529411764707, 0.38823529411764707),
                  'Lamp5': (0.5882352941176471, 0.5882352941176471, 0.5882352941176471),
                  'Sst': (0.7411764705882353, 0.7411764705882353, 0.7411764705882353),
                  'Sst Chodl': (0.8509803921568627, 0.8509803921568627, 0.8509803921568627),
                  'Sncg': (0.5176470588235295, 0.23529411764705882, 0.2235294117647059),
                  'Peri': (0.6784313725490196, 0.28627450980392155, 0.2901960784313726),
                  'VLMC': (0.8392156862745098, 0.3803921568627451, 0.4196078431372549),
                  'Endo': (0.9058823529411765, 0.5882352941176471, 0.611764705882353),
                  'Meis2': '#FFA500ff',
                  'SMC': '#000000ff',
                  'L6 PT': '#4682B4ff',
                  'L5 PT': '#a1ed7bff',
                  'L5 NP': '#6B8E23ff',
                  'L4': '#d61f1dff',
                  'Macrophage': '#2b2d2fff',
                  'CR': '#000000ff'}