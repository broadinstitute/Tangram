import scanpy as sc

def hvg(adata_sc):
    sc.pp.highly_variable_genes(adata_sc, n_top_genes=4000)
    return list(adata_sc[:,adata_sc.var["highly_variable"]].var_names)