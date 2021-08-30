import scanpy as sc
import tangram as tg
import numpy as np
import pandas as pd
import pytest
from pytest import approx

# to run test_tangram.py on your local machine, please set up as follow:
# - create test environment according to environment.yml (conda create env -f environment.yml) to make sure environment matches developing environment
# - install extra package: pytest, pytest-cov
# - install editable version of tangram package (pip install -e .)
#      - check tangram version (conda list tangram), make sure it is the developing version
# - make sure the test data are ready under test_data folder

# test data


@pytest.fixture
def adatas():
    ad_sc = sc.read_h5ad("data/test_ad_sc.h5ad")
    ad_sp = sc.read_h5ad("data/test_ad_sp.h5ad")
    return (ad_sc, ad_sp)


@pytest.fixture
def df_all_genes():
    df_all_genes = pd.read_csv("data/test_df.csv", index_col=0)
    return df_all_genes


@pytest.fixture
def ad_sc_mock():
    X = np.array([[0, 1, 1], [0, 1, 1]])
    obs = pd.DataFrame(index=["cell_1", "cell_2"])
    var = pd.DataFrame(index=["gene_a", "gene_b", "gene_d"])
    ad_sc_mock = sc.AnnData(X=X, obs=obs, var=var)
    return ad_sc_mock


@pytest.fixture
def ad_sp_mock():
    X = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    obs = pd.DataFrame(index=["voxel_1", "voxel_2"])
    var = pd.DataFrame(index=["gene_c", "gene_b", "gene_a", "gene_d"])
    ad_sp_mock = sc.AnnData(X=X, obs=obs, var=var)

    return ad_sp_mock


# test pp_data


@pytest.mark.parametrize("genes", [(None), (["gene_a", "gene_b"]),])
def test_pp_data(ad_sc_mock, ad_sp_mock, genes):
    tg.pp_adatas(ad_sc_mock, ad_sp_mock, genes)

    assert ad_sc_mock.uns["training_genes"] == ad_sp_mock.uns["training_genes"]
    assert ad_sc_mock.uns["overlap_genes"] == ad_sp_mock.uns["overlap_genes"]
    assert ad_sc_mock.X.any(axis=0).all() and ad_sp_mock.X.any(axis=0).all()
    assert "rna_count_based_density" in ad_sp_mock.obs.keys()
    assert "uniform_density" in ad_sp_mock.obs.keys()


# test mapping function with different parameters


@pytest.mark.parametrize(
    "lambda_g1, lambda_g2, lambda_d, density_prior, scale, e",
    [
        (1, 0, 0, None, True, np.float32(8.280743e-06)),
        (1, 0, 0, None, False, np.float32(2.785552e-07)),
        (1, 1, 0, None, True, np.float32(8.376801e-06)),
        (1, 1, 0, None, False, np.float32(2.4095453e-07)),
        (1, 1, 1, "uniform", True, np.float32(8.376801e-06)),
        (1, 1, 1, "uniform", False, np.float32(2.4095453e-07)),
        (1, 0, 2, "uniform", True, np.float32(1.3842443e-06)),
        (1, 0, 1, "rna_count_based", True, np.float32(0.0023217443)),
        (1, 0, 1, "uniform", True, np.float32(8.280743e-06)),
    ],
)
def test_map_cells_to_space(
    adatas, lambda_g1, lambda_g2, lambda_d, density_prior, scale, e,
):

    # mapping with defined random_state
    ad_map = tg.map_cells_to_space(
        adata_sc=adatas[0],
        adata_sp=adatas[1],
        device="cpu",
        mode="clusters",
        cluster_label="subclass_label",
        lambda_g1=lambda_g1,
        lambda_g2=lambda_g2,
        lambda_d=lambda_d,
        density_prior=density_prior,
        scale=scale,
        random_state=42,
        num_epochs=500,
        verbose=True,
    )

    # check if first element of output_admap.X is equal to expected value
    assert round(ad_map.X[0, 0], 3) == round(e, 3)


# test mapping exception with assertion


@pytest.mark.parametrize(
    "mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale, e",
    [
        ("clusters", "subclass", 0, 0, 0, True, "lambda_g1 cannot be 0."),
        (
            "not_clusters_or_cells",
            None,
            1,
            0,
            0,
            True,
            'Argument "mode" must be "cells" or "clusters"',
        ),
        (
            "clusters",
            None,
            1,
            0,
            0,
            True,
            "An cluster_label must be specified if mode = clusters.",
        ),
    ],
)
def test_invalid_map_cells_to_space(
    adatas, mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale, e
):
    with pytest.raises(ValueError) as exc_info:

        tg.map_cells_to_space(
            adata_sc=adatas[0],
            adata_sp=adatas[1],
            device="cpu",
            mode=mode,
            cluster_label=cluster_label,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_d=lambda_d,
            scale=scale,
            random_state=42,
            num_epochs=500,
            verbose=False,
        )
        assert e in str(exc_info.value)


# test to see if the average training score matches between the one in
# training history and the one from compare_spatial_geneexp function


@pytest.mark.parametrize(
    "mode, cluster_label, lambda_g1, lambda_g2, lambda_d, density_prior, scale",
    [
        ("clusters", "subclass_label", 1, 0, 0, None, True),
        ("clusters", "subclass_label", 1, 0, 0, None, False),
        ("clusters", "subclass_label", 1, 1, 0, None, True),
        ("clusters", "subclass_label", 1, 1, 0, None, False),
        ("clusters", "subclass_label", 1, 0, 1, "uniform", True),
        ("clusters", "subclass_label", 1, 0, 1, "rna_count_based", False),
    ],
)
def test_train_score_match(
    adatas, mode, cluster_label, lambda_g1, lambda_g2, lambda_d, density_prior, scale,
):

    # mapping with defined random_state
    ad_map = tg.map_cells_to_space(
        adata_sc=adatas[0],
        adata_sp=adatas[1],
        device="cpu",
        mode=mode,
        cluster_label=cluster_label,
        lambda_g1=lambda_g1,
        lambda_g2=lambda_g2,
        lambda_d=lambda_d,
        density_prior=density_prior,
        scale=scale,
        random_state=42,
        num_epochs=500,
        verbose=False,
    )

    # call project_genes to project input ad_sc data to ad_ge spatial data
    # with ad_map
    ad_ge = tg.project_genes(
        adata_map=ad_map,
        adata_sc=adatas[0],
        cluster_label="subclass_label",
        scale=scale,
    )

    df_all_genes = tg.compare_spatial_geneexp(ad_ge, adatas[1])

    avg_score_df = round(
        df_all_genes[df_all_genes["is_training"] == True]["score"].mean(), 3
    )
    avg_score_train_hist = round(
        np.float(list(ad_map.uns["training_history"]["main_loss"])[-1]), 3
    )

    # check if raining score matches between the one in training history and the one from compare_spatial_geneexp function
    assert avg_score_df == approx(avg_score_train_hist)


# test cross validation function
def test_eval_metric(df_all_genes):
    auc_score = tg.eval_metric(df_all_genes)[0]["auc_score"]
    assert auc_score == approx(0.750597829464878)

