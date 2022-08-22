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

import pandas as pd
import logging
import matplotlib as mpl
from matplotlib.gridspec import GridSpec


def q_value(data, perc):
    """
    Computes min and max values according to percentile for colormap in plot functions

    Args:
        data (numpy array): input
        perc (float): percentile that between 0 and 100 inclusive

    Returns:
        tuple of floats: will be later used to define the data range covers by the colormap
    """
    vmin = np.nanpercentile(data, perc)
    vmax = np.nanpercentile(data, 100 - perc)

    return vmin, vmax


def plot_training_scores(adata_map, bins=10, alpha=0.7):
    """
    Plots the 4-panel training diagnosis plot

    Args:
        adata_map (AnnData):
        bins (int or string): Optional. Default is 10.
        alpha (float): Optional. Ranges from 0-1, and controls the opacity. Default is 0.7.

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    df = adata_map.uns["train_genes_df"]
    axs_f = axs.flatten()

    # set limits for axis
    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    #     axs_f[0].set_title('Training scores for single genes')
    sns.histplot(data=df, y="train_score", bins=bins, ax=axs_f[0], color="coral")

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_sc",
        ax=axs_f[1],
        alpha=alpha,
        color="coral",
    )

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_sp",
        ax=axs_f[2],
        alpha=alpha,
        color="coral",
    )

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_diff",
        ax=axs_f[3],
        alpha=alpha,
        color="coral",
    )

    plt.tight_layout()


def plot_gene_sparsity(
    adata_1, adata_2, xlabel="adata_1", ylabel="adata_2", genes=None, s=1
):
    """
    Compare sparsity of all genes between `adata_1` and `adata_2`.

    Args:
        adata_1 (AnnData): Input data
        adata_2 (AnnData): Input data
        xlabel (str): Optional. For setting the xlabel in the plot. Default is 'adata_1'.
        ylabel (str): Optional. For setting the ylabel in the plot. Default is 'adata_2'.  
        genes (list): Optional. List of genes to use. If `None`, all genes are used.
        s (float): Optional. Controls the size of marker. Default is 1.

    Returns:
        None
    """
    logging.info("Pre-processing AnnDatas...")
    adata_1, adata_2 = mu.pp_adatas(adata_1, adata_2, genes=genes)
    assert adata_1.uns["training_genes"] == adata_2.uns["training_genes"]
    training_genes = adata_1.uns["training_genes"]

    logging.info("Annotating sparsity...")
    ut.annotate_gene_sparsity(adata_1)
    ut.annotate_gene_sparsity(adata_2)
    xs = adata_1[:, training_genes].var["sparsity"].values
    ys = adata_2[:, training_genes].var["sparsity"].values
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1)
    ax.set_xlabel("sparsity (" + xlabel + ")")
    ax.set_ylabel("sparsity (" + ylabel + ")")
    ax.set_title(f"Gene sparsity ({len(xs)} genes)")
    ax.scatter(xs, ys, s=s, marker="x")


def ordered_predictions(xs, ys, preds, reverse=False):
    """
    Utility function that orders 2d points based on values associated to each point.

    Args:
        xs (Pandas series): Sequence of x coordinates (floats).
        ys (Pandas series): Sequence of y coordinates (floats).
        preds (Pandas series): Sequence of spatial prediction.
        reverse (bool): Optional. False will sort ascending, True will sort descending. Default is False.
        
    Returns:
        Returns the ordered xs, ys, preds.
    """
    assert len(xs) == len(ys) == len(preds)
    return list(
        zip(
            *[
                (x, y, z)
                for x, y, z in sorted(
                    zip(xs, ys, preds), key=lambda pair: pair[2], reverse=reverse
                )
            ]
        )
    )


def convert_adata_array(adata):
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        adata.X = adata.X.toarray()


def construct_obs_plot(df_plot, adata, perc=0, suffix=None):
    # clip
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)

    # normalize
    df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())

    if suffix:
        df_plot = df_plot.add_suffix(" ({})".format(suffix))
    adata.obs = pd.concat([adata.obs, df_plot], axis=1)


def plot_cell_annotation_sc(
    adata_sp, 
    annotation_list, 
    x="x", 
    y="y", 
    spot_size=None, 
    scale_factor=None, 
    perc=0,
    alpha_img=1.0,
    bw=False,
    ax=None
):
        
    # remove previous df_plot in obs
    adata_sp.obs.drop(annotation_list, inplace=True, errors="ignore", axis=1)

    # construct df_plot
    df = adata_sp.obsm["tangram_ct_pred"][annotation_list]
    construct_obs_plot(df, adata_sp, perc=perc)
    
    #non visium data 
    if 'spatial' not in adata_sp.obsm.keys():
        #add spatial coordinates to obsm of spatial data 
        coords = [[x,y] for x,y in zip(adata_sp.obs[x].values,adata_sp.obs[y].values)]
        adata_sp.obsm['spatial'] = np.array(coords)
    
    if 'spatial' not in adata_sp.uns.keys() and spot_size == None and scale_factor == None:
        raise ValueError("Spot Size and Scale Factor cannot be None when ad_sp.uns['spatial'] does not exist")
    
    #REVIEW
    if 'spatial' in adata_sp.uns.keys() and spot_size != None and scale_factor != None:
        raise ValueError("Spot Size and Scale Factor should be None when ad_sp.uns['spatial'] exists")
    
    sc.pl.spatial(
        adata_sp, color=annotation_list, cmap="viridis", show=False, frameon=False, spot_size=spot_size,
        scale_factor=scale_factor, alpha_img=alpha_img, bw=bw, ax=ax
    )

    adata_sp.obs.drop(annotation_list, inplace=True, errors="ignore", axis=1)


def plot_cell_annotation(
    adata_map,
    adata_sp,
    annotation="cell_type",
    x="x",
    y="y",
    nrows=1,
    ncols=1,
    s=5,
    cmap="viridis",
    subtitle_add=False,
    robust=False,
    perc=0,
    invert_y=True,
):
    """
    Transfer an annotation for a single cell dataset onto space, and visualize
    corresponding spatial probability maps.

    Args:
        adata_map (AnnData): cell-by-spot AnnData containing mapping result
        adata_sp (AnnData): spot-by-gene spatial AnnData
        annotation (str): Optional. Must be a column in `adata_map.obs`. Default is 'cell_type'.
        x (str): Optional. Column name for spots x-coordinates (must be in `adata_map.var`). Default is 'x'.
        y (str): Optional. Column name for spots y-coordinates (must be in `adata_map.var`). Default is 'y'.
        nrows (int): Optional. Number of rows of the subplot grid. Default is 1.
        ncols (int): Optional. Number of columns of the subplot grid. Default is 1.
        s (float): Optional. Marker size. Default is 5.
        cmap (str): Optional. Name of colormap. Default is 'viridis'.
        subtitle_add (bool): Optional. If add annotation name as the subtitle. Default is False.
        robust (bool): Optional. If True, the colormap range is computed with given percentiles instead of extreme values.
        perc (float): Optional. percentile used to calculate colormap range, only used when robust is True. Default is zero.
        invert_y (bool): Optional. If invert the y axis for the plot. Default is True.

    Returns:
        None
    """

    # TODO ADD CHECKS for x and y

    if not robust and perc != 0:
        raise ValueError("Arg perc is zero when robust is False.")

    if robust and perc == 0:
        raise ValueError("Arg perc cannot be zero when robust is True.")

    ut.project_cell_annotations(adata_map, adata_sp, annotation=annotation)

    df_annotation = adata_sp.obsm["tangram_ct_pred"]

    #### Colorbar:
    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.subplots_adjust(top=0.5)

    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=1,)

    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="horizontal", label="Probability"
    )
    #### Colorbar

    if nrows is None or ncols is None:
        ncols = 1
        nrows = len(df_annotation.columns)

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 3, nrows * 3), sharex=True, sharey=True
    )

    axs_f = axs.flatten()
    if invert_y is True:
        axs_f[0].invert_yaxis()

    [ax.axis("off") for ax in axs_f]

    if len(df_annotation.columns) > nrows * ncols:
        logging.warning(
            "Number of panels smaller than annotations. Increase `nrows`/`ncols`."
        )

    iterator = zip(df_annotation.columns, range(nrows * ncols))
    for ann, index in iterator:
        xs, ys, preds = ordered_predictions(
            adata_map.var[x], adata_map.var[y], df_annotation[ann]
        )

        if robust:
            vmin, vmax = q_value(preds, perc=perc)
        else:
            vmin, vmax = q_value(preds, perc=0)

        axs_f[index].scatter(x=xs, y=ys, c=preds, s=s, cmap=cmap, vmin=vmin, vmax=vmax)
        axs_f[index].set_title(ann)
        axs_f[index].set_aspect(1)

    if subtitle_add is True:
        fig.suptitle(annotation)


def plot_genes_sc(
    genes, 
    adata_measured, 
    adata_predicted,
    x="x",
    y = "y",
    spot_size=None, 
    scale_factor=None, 
    cmap="inferno", 
    perc=0,
    alpha_img=1.0,
    bw=False,
    return_figure=False
):

    # remove df_plot in obs
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )

    # prepare adatas
    convert_adata_array(adata_measured)

    adata_measured.var.index = [g.lower() for g in adata_measured.var.index]
    adata_predicted.var.index = [g.lower() for g in adata_predicted.var.index]

    adata_predicted.obsm = adata_measured.obsm
    adata_predicted.uns = adata_measured.uns

    # remove previous df_plot in obs
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )

    # construct df_plot
    data = []
    for ix, gene in enumerate(genes):
        if gene not in adata_measured.var.index:
            data.append(np.zeros_like(np.array(adata_measured[:, 0].X).flatten()))
        else:
            data.append(np.array(adata_measured[:, gene].X).flatten())

    df = pd.DataFrame(
        data=np.array(data).T, columns=genes, index=adata_measured.obs.index,
    )
    construct_obs_plot(df, adata_measured, suffix="measured")

    df = pd.DataFrame(
        data=np.array(adata_predicted[:, genes].X),
        columns=genes,
        index=adata_predicted.obs.index,
    )
    construct_obs_plot(df, adata_predicted, perc=perc, suffix="predicted")

    fig = plt.figure(figsize=(7, len(genes) * 3.5))
    gs = GridSpec(len(genes), 2, figure=fig)
    
    #non visium data
    if 'spatial' not in adata_measured.obsm.keys():
        #add spatial coordinates to obsm of spatial data 
        coords = [[x,y] for x,y in zip(adata_measured.obs[x].values,adata_measured.obs[y].values)]
        adata_measured.obsm['spatial'] = np.array(coords)
        coords = [[x,y] for x,y in zip(adata_predicted.obs[x].values,adata_predicted.obs[y].values)]
        adata_predicted.obsm['spatial'] = np.array(coords)

    if ("spatial" not in adata_measured.uns.keys()) and (spot_size==None and scale_factor==None):
        raise ValueError("Spot Size and Scale Factor cannot be None when ad_sp.uns['spatial'] does not exist")
        
    for ix, gene in enumerate(genes):
        ax_m = fig.add_subplot(gs[ix, 0])
        sc.pl.spatial(
            adata_measured,
            spot_size=spot_size,
            scale_factor=scale_factor,
            color=["{} (measured)".format(gene)],
            frameon=False,
            ax=ax_m,
            show=False,
            cmap=cmap,
            alpha_img=alpha_img,
            bw=bw
        )
        ax_p = fig.add_subplot(gs[ix, 1])
        sc.pl.spatial(
            adata_predicted,
            spot_size=spot_size,
            scale_factor=scale_factor,
            color=["{} (predicted)".format(gene)],
            frameon=False,
            ax=ax_p,
            show=False,
            cmap=cmap,
            alpha_img=alpha_img,
            bw=bw
        )
        
    #     sc.pl.spatial(adata_measured, color=['{} (measured)'.format(gene) for gene in genes], frameon=False)
    #     sc.pl.spatial(adata_predicted, color=['{} (predicted)'.format(gene) for gene in genes], frameon=False)

    # remove df_plot in obs
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    if return_figure==True:
        return fig


def plot_genes(
    genes,
    adata_measured,
    adata_predicted,
    x="x",
    y="y",
    s=5,
    log=False,
    cmap="inferno",
    robust=False,
    perc=0,
    invert_y=True,
):
    """
    Utility function to plot and compare original and projected gene spatial pattern ordered by intensity of the gene signal.

    Args:
        genes (list): list of gene names (str).
        adata_measured (AnnData): ground truth gene spatial AnnData
        adata_predicted (AnnData): projected gene spatial AnnData, can also be adata_ge_cv AnnData returned by cross_validation under 'loo' mode
        x (str): Optional. Column name for spots x-coordinates (must be in `adata_measured.var` and `adata_predicted.var`). Default is 'x'.
        y (str): Optional. Column name for spots y-coordinates (must be in `adata_measured.var` and `adata_predicted.var`). Default is 'y'.
        s (float): Optional. Marker size. Default is 5.
        log: Optional. Whether to apply the log before plotting. Default is False.
        cmap (str): Optional. Name of colormap. Default is 'inferno'.
        robust (bool): Optional. If True, the colormap range is computed with given percentiles instead of extreme values.
        perc (float): Optional. percentile used to calculate colormap range, only used when robust is True. Default is zero.
        invert_y (bool): Optional. If invert the y axis for the plot. Default is True.

    Returns:
        None
    """
    # TODO: not very elegant and slow as hell

    if not robust and perc != 0:
        raise ValueError("Arg perc is zero when robust is False.")

    if robust and perc == 0:
        raise ValueError("Arg perc cannot be zero when robust is True.")

    if isinstance(adata_measured.X, csc_matrix) or isinstance(
        adata_measured.X, csr_matrix
    ):
        adata_measured.X = adata_measured.X.toarray()

    adata_measured.var.index = [g.lower() for g in adata_measured.var.index]
    adata_predicted.var.index = [g.lower() for g in adata_predicted.var.index]

    #### Colorbar:
    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.subplots_adjust(top=0.5)

    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=1,)

    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="horizontal", label="Expression Level"
    )
    #### Colorbar

    fig, axs = plt.subplots(nrows=len(genes), ncols=2, figsize=(6, len(genes) * 3))

    for ix, gene in enumerate(genes):
        if gene not in adata_measured.var.index:
            vs = np.zeros_like(np.array(adata_measured[:, 0].X).flatten())
        else:
            vs = np.array(adata_measured[:, gene].X).flatten()

        xs, ys, vs = ordered_predictions(
            adata_measured.obs[x], adata_measured.obs[y], vs,
        )

        if log:
            vs = np.log(1 + np.asarray(vs))
        axs[ix, 0].scatter(xs, ys, c=vs, cmap=cmap, s=s)
        axs[ix, 0].set_title(gene + " (measured)")
        axs[ix, 0].axis("off")
        axs[ix, 0].set_aspect(1)

        xs, ys, vs = ordered_predictions(
            adata_predicted.obs[x],
            adata_predicted.obs[y],
            np.array(adata_predicted[:, gene].X).flatten(),
        )

        if robust:
            vmin, vmax = q_value(vs, perc=perc)
        else:
            vmin, vmax = q_value(vs, perc=0)

        if log:
            vs = np.log(1 + np.asarray(vs))
        axs[ix, 1].scatter(xs, ys, c=vs, cmap=cmap, s=s, vmin=vmin, vmax=vmax)
        axs[ix, 1].set_title(gene + " (predicted)")
        axs[ix, 1].axis("off")
        axs[ix, 1].set_aspect(1)

        if invert_y is True:
            axs[ix, 0].invert_yaxis()
            axs[ix, 1].invert_yaxis()


def quick_plot_gene(
    gene, adata, x="x", y="y", s=50, log=False, cmap="viridis", robust=False, perc=0
):
    """
    Utility function to quickly plot a gene in a AnnData structure ordered by intensity of the gene signal.
    
    Args:
        gene (str): Gene name.
        adata (AnnData): spot-by-gene spatial data.
        x (str): Optional. Column name for spots x-coordinates (must be in `adata.var`). Default is 'x'.
        y (str): Optional. Column name for spots y-coordinates (must be in `adata.var`). Default is 'y'.
        s (float): Optional. Marker size. Default is 5.
        log: Optional. Whether to apply the log before plotting. Default is False.
        cmap (str): Optional. Name of colormap. Default is 'viridis'.
        robust (bool): Optional. If True, the colormap range is computed with given percentiles instead of extreme values.
        perc (float): Optional. percentile used to calculate colormap range, only used when robust is True. Default is zero.

    Returns:
        None
    """
    if not robust and perc != 0:
        raise ValueError("Arg perc is zero when robust is False.")

    if robust and perc == 0:
        raise ValueError("Arg perc cannot be zero when robust is True.")

    xs, ys, vs = ordered_predictions(
        adata.obs[x], adata.obs[y], np.array(adata[:, gene].X).flatten()
    )
    if robust:
        vmin, vmax = q_value(vs, perc=perc)
    else:
        vmin, vmax = q_value(vs, perc=0)
    if log:
        vs = np.log(1 + np.asarray(vs))
    plt.scatter(xs, ys, c=vs, cmap=cmap, s=s, vmin=vmin, vmax=vmax)


def plot_annotation_entropy(adata_map, annotation="cell_type"):
    """
    Utility function to plot entropy box plot by each annotation.

    Args:
        adata_map (AnnData): cell-by-voxel tangram mapping result.
        annotation (str): Optional. Must be a column in `adata_map.obs`. Default is 'cell_type'.

    Returns:
        None
    """
    qk = np.ones(shape=(adata_map.n_obs, adata_map.n_vars))
    adata_map.obs["entropy"] = entropy(adata_map.X, base=adata_map.X.shape[1], axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.set_ylim(0, 1)
    sns.boxenplot(x=annotation, y="entropy", data=adata_map.obs, ax=ax)
    plt.xticks(rotation=30)


def plot_test_scores(df_gene_score, bins=10, alpha=0.7):
    """
    Plots gene level test scores with each gene's sparsity for mapping result.
    
    Args:
        df_gene_score (Pandas dataframe): returned by compare_spatial_geneexp(adata_ge, adata_sp, adata_sc); 
                       with "gene names" as the index and "score", "sparsity_sc", "sparsity_sp", "sparsity_diff" as the columns
        bins (int or string): Optional. Default is 10.
        alpha (float): Optional. Ranges from 0-1, and controls the opacity. Default is 0.7.

    Returns:
        None
    """

    # check if df_gene_score has all required columns
    if not set(["score", "sparsity_sc", "sparsity_sp", "sparsity_diff"]).issubset(
        set(df_gene_score.columns)
    ):
        raise ValueError(
            "There are missing columns in df_gene_score. Run `compare_spatial_geneexp` with `adata_ge`, `adata_sp`, and `adata_sc` to produce complete dataframe input."
        )

    if "is_training" in df_gene_score.keys():
        df = df_gene_score[df_gene_score["is_training"] == False]
    else:
        df = df_gene_score

    df.rename({"score": "test_score"}, axis="columns", inplace=True)

    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axs_f = axs.flatten()

    # set limits for axis
    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    sns.histplot(data=df, y="test_score", bins=bins, ax=axs_f[0])

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(data=df, y="test_score", x="sparsity_sc", ax=axs_f[1], alpha=alpha)

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(data=df, y="test_score", x="sparsity_sp", ax=axs_f[2], alpha=alpha)

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(
        data=df, y="test_score", x="sparsity_diff", ax=axs_f[3], alpha=alpha
    )
    plt.tight_layout()

    
def plot_auc(df_all_genes, test_genes=None):
    """
        Plots auc curve which is used to evaluate model performance.
    
    Args:
        df_all_genes (Pandas dataframe): returned by compare_spatial_geneexp(adata_ge, adata_sp); 
        test_genes (list): list of test genes, if not given, test_genes will be set to genes where 'is_training' field is False

    Returns:
        None
    """
    metric_dict, ((pol_xs, pol_ys), (xs, ys)) = ut.eval_metric(df_all_genes, test_genes)
    
    fig = plt.figure()
    plt.figure(figsize=(6, 5))

    plt.plot(pol_xs, pol_ys, c='r')
    sns.scatterplot(xs, ys, alpha=0.5, edgecolors='face')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(.5)
    plt.xlabel('score')
    plt.ylabel('spatial sparsity')
    plt.tick_params(axis='both', labelsize=8)
    plt.title('Prediction on test transcriptome')
    
    textstr = 'auc_score={}'.format(np.round(metric_dict['auc_score'], 3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    # place a text box in upper left in axes coords
    plt.text(0.03, 0.1, textstr, fontsize=11, verticalalignment='top', bbox=props);

    
# Colors used in the manuscript for deterministic assignment.
mapping_colors = {
    "L6 CT": (0.19215686274509805, 0.5098039215686274, 0.7411764705882353),
    "L6 IT": (0.4196078431372549, 0.6823529411764706, 0.8392156862745098),
    "L5/6 NP": (0.6196078431372549, 0.792156862745098, 0.8823529411764706),
    "L6b": "#0000c2ff",
    "L2/3 IT": (0.9019607843137255, 0.3333333333333333, 0.050980392156862744),
    "L5 IT": (0.19215686274509805, 0.6392156862745098, 0.32941176470588235),
    "L5 ET": (0.4549019607843137, 0.7686274509803922, 0.4627450980392157),
    "Oligo": (0.4588235294117647, 0.4196078431372549, 0.6941176470588235),
    "Vip": (0.6196078431372549, 0.6039215686274509, 0.7843137254901961),
    "Astro": "#ffdd55ff",
    "Micro-PVM": "#000000ff",
    "Pvalb": (0.38823529411764707, 0.38823529411764707, 0.38823529411764707),
    "Lamp5": (0.5882352941176471, 0.5882352941176471, 0.5882352941176471),
    "Sst": (0.7411764705882353, 0.7411764705882353, 0.7411764705882353),
    "Sst Chodl": (0.8509803921568627, 0.8509803921568627, 0.8509803921568627),
    "Sncg": (0.5176470588235295, 0.23529411764705882, 0.2235294117647059),
    "Peri": (0.6784313725490196, 0.28627450980392155, 0.2901960784313726),
    "VLMC": (0.8392156862745098, 0.3803921568627451, 0.4196078431372549),
    "Endo": (0.9058823529411765, 0.5882352941176471, 0.611764705882353),
    "Meis2": "#FFA500ff",
    "SMC": "#000000ff",
    "L6 PT": "#4682B4ff",
    "L5 PT": "#a1ed7bff",
    "L5 NP": "#6B8E23ff",
    "L4": "#d61f1dff",
    "Macrophage": "#2b2d2fff",
    "CR": "#000000ff",
}
