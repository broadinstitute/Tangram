import numpy as np
import pandas as pd
import scanpy as sc

def ctg(adata_sc):
    sc.tl.rank_genes_groups(adata_sc, groupby="cell_subclass", use_raw=False)
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:150, :]
    return list(np.unique(markers_df.melt().value.values))