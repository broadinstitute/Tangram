import pytest
import scanpy as sc
import tangram as tg
import numpy as np

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





