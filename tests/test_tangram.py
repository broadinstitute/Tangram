import scanpy as sc
import tangram as tg
import numpy as np

import pytest
import os
import wget

# to run test_tangram.py on your local machine, please set up as follow:
# - create test environment according to environment.yml (conda create env -f environment.yml) to make sure environment matches developing environment
# - install extra package: pytest, pytest-cov
# - install editable version of tangram package (pip install -e .)
#      - check tangram version (conda list tangram), make sure it is the developing version
# - make sure the test data are ready under test_data folder


# mapping input data (anndata formated single cell data)
@pytest.fixture
def ad_sc():
    ad_sc = sc.read_h5ad('test_data/ad_sc_readytomap.h5ad')
    return ad_sc

# mapping input data (anndata formated spatial data)
@pytest.fixture
def ad_sp():
    ad_sp = sc.read_h5ad('test_data/ad_sp_readytomap.h5ad')
    return ad_sp

# test mapping function with different parameters
@pytest.mark.parametrize('mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale, e', [
    ('clusters', 'subclass', 1, 0, 0, True, np.float32(0.00033864976)), 
    ('clusters', 'subclass', 1, 0, 0, False, np.float32(1.0042528e-05)),
    ('clusters', 'subclass', 1, 1, 0, True, np.float32(1.7422922e-06)), 
    ('clusters', 'subclass', 1, 1, 0, False, np.float32(6.644411e-06)),
    ('clusters', 'subclass', 1, 1, 1, True, np.float32(0.0013598711)),
    ('clusters', 'subclass', 1, 1, 1, False, np.float32(4.3795535e-06)),
])
def test_map_cells_to_space(ad_sc, ad_sp, mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale, e):
    
    # mapping with defined random_state
    ad_map = tg.map_cells_to_space(
                    adata_cells=ad_sc,
                    adata_space=ad_sp,
                    device='cpu',
                    mode=mode,
                    cluster_label=cluster_label,
                    lambda_g1=lambda_g1,
                    lambda_g2=lambda_g2,
                    lambda_d=lambda_d,
                    scale=scale,
                    random_state=42,
                    num_epochs=500)

    # check if first element of output_admap.X is equal to expected value
    assert ad_map.X[0,0] == e

# test mapping exception with assertion
@pytest.mark.parametrize('mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale, e', [
    ('clusters', 'subclass', 0, 0, 0, True, 'lambda_g1 cannot be 0.'), 
    ('not_clusters_or_cells', None, 1, 0, 0, True, 'Argument "mode" must be "cells" or "clusters"'), 
    ('clusters', None, 1, 0, 0, True, 'An cluster_label must be specified if mode = clusters.'),
])
def test_invalid_map_cells_to_space(ad_sc, ad_sp, mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale, e):
    with pytest.raises(ValueError) as exc_info:
        tg.map_cells_to_space(
                    adata_cells=ad_sc,
                    adata_space=ad_sp,
                    device='cpu',
                    mode=mode,
                    cluster_label=cluster_label,
                    lambda_g1=lambda_g1,
                    lambda_g2=lambda_g2,
                    lambda_d=lambda_d,
                    scale=scale,
                    random_state=42,
                    num_epochs=500)
        assert e in str(exc_info.value)

# test to see if the average training score matches between the one in training history and the one from compare_spatial_geneexp function
# test mapping function with different parameters
@pytest.mark.parametrize('mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale', [
    ('clusters', 'subclass', 1, 0, 0, True),
    ('clusters', 'subclass', 1, 0, 0, False),
    ('clusters', 'subclass', 1, 1, 0, True),
    ('clusters', 'subclass', 1, 1, 0, False),
    ('clusters', 'subclass', 1, 0, 1, True),
    ('clusters', 'subclass', 1, 0, 1, False),
])
def test_map_cells_to_space(ad_sc, ad_sp, mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale):
    
    # mapping with defined random_state
    ad_map = tg.map_cells_to_space(
                    adata_cells=ad_sc,
                    adata_space=ad_sp,
                    device='cpu',
                    mode=mode,
                    cluster_label=cluster_label,
                    lambda_g1=lambda_g1,
                    lambda_g2=lambda_g2,
                    lambda_d=lambda_d,
                    scale=scale,
                    random_state=42,
                    num_epochs=500)

    # call project_genes to project input ad_sc data to ad_ge spatial data with ad_map
    ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=ad_sc, cluster_label=cluster_label, scale=scale)
    df_all_genes = tg.compare_spatial_geneexp(ad_ge, ad_sp)
    
    avg_score_df = df_all_genes['score'].mean()
    avg_score_train_hist = list(ad_map.uns['training_history']['main_loss'].values)[-1]

    # check if raining score matches between the one in training history and the one from compare_spatial_geneexp function
    assert avg_score_df == avg_score_train_hist






