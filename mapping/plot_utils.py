"""
This module includes plotting utility functions.
"""
import numpy as np
import matplotlib.pyplot as plt


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