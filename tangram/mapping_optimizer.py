"""
Library for instantiating and running the optimizer for Tangram. The optimizer comes in two flavors,
which correspond to two different classes:
- Mapper: optimizer without filtering (i.e., all single cells are mapped onto space). At the end, the learned mapping
matrix M is returned.
- MapperConstrained: optimizer with filtering (i.e., only a subset of single cells are mapped onto space).
At the end, the learned mapping matrix M and the learned filter F are returned.
"""
import numpy as np
import logging
import torch
from torch.nn.functional import softmax, cosine_similarity

class Mapper:
    """
    Allows instantiating and running the optimizer for Tangram, without filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping result.
    """
    def __init__(
        self,
        S,
        G,
        train_genes_idx=None,
        val_genes_idx=None,
        d=None,
        d_source=None,
        lambda_g1=1.0,
        lambda_d=0,
        lambda_g2=0,
        lambda_r=0,
        lambda_l1=0,
        lambda_l2=0,
        lambda_neighborhood_g1=0,
        voxel_weights=None,
        lambda_getis_ord=0,
        lambda_geary=0,
        lambda_moran=0,
        neighborhood_filter=None,
        ct_encode=None,
        lambda_ct_islands=0,
        spatial_weights=None,
        device="cpu",
        adata_map=None,
        random_state=None,
    ):
        """
        Instantiate the Tangram optimizer (without filtering).

        Args:
            S (ndarray): Single nuclei matrix, shape = (number_cell, number_genes).
            G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
                Spots can be single cells or they can contain multiple cells.
            train_genes_idx (ndarray): Optional. Gene indices used for training.
            val_genes_idx (ndarray): Optional. Gene indices used for validation.
            d (ndarray): Spatial density of cells, shape = (number_spots,). If not provided, the density term is ignored.
                This array should satisfy the constraints d.sum() == 1.
            d_source (ndarray): Density of single cells in single cell clusters. To be used when S corresponds to cluster-level expression.
                This array should satisfy the constraint d_source.sum() == 1.
            lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
            lambda_d (float): Optional. Strength of density regularizer. Default is 0.
            lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
            lambda_r (float): Optional. Strength of entropy regularizer. An higher entropy promotes
                              probabilities of each cell peaked over a narrow portion of space.
                              lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
            lambda_l1 (float): Optional. Strength of L1 regularizer. Default is 0.
            lambda_l2 (float): Optional. Strength of L2 regularizer. Default is 0.
            lambda_neighborhood_g1 (float): Optional. Strength of neighborhood weighted gene expression comparison. Default is 0.
            voxel_weights (ndarray): Optional. Spatial weight used for neighborhood weighting, shape = (number_spots, number_spots).
            lambda_getis_ord (float): Optional. Strength of Getis-Ord G* preservation. Default is 0.
            lambda_geary (float): Optional. Strength of Geary's C preservation. Default is 0.
            lambda_moran (float): Optional. Strength of Moran's I preservation. Default is 0.
            spatial_weights (ndarray): Optional. Spatial weight used for local spatial indicator preservation, shape = (number_spots, number_spots).
            lambda_ct_islands: Optional. Strength of ct islands enforcement. Default is 0.
            neighborhood_filter (ndarray): Optional. Neighborhood filter used for cell type island preservation, shape = (number_spots, number_spots).
            ct_encode(ndarray): Optional. One-hot encoding of cell types used for cell type island preservation, shape = (number_cells, number_celltypes).
            device (str or torch.device): Optional. Device is 'cpu'.
            adata_map (scanpy.AnnData): Optional. Mapping initial condition (for resuming previous mappings). Default is None.
            random_state (int): Optional. pass an int to reproduce training. Default is None.
        """
        self.device = device
        self.random_state = random_state

        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)

        # setup training and validation data
        if train_genes_idx is not None:
            self.S_train = self.S[:, train_genes_idx].clone()
            self.G_train = self.G[:, train_genes_idx].clone()
        else:
            self.S_train = self.S.clone()
            self.G_train = self.G.clone()
        if val_genes_idx is not None:
            self.S_val = self.S[:, val_genes_idx].clone()
            self.G_val = self.G[:, val_genes_idx].clone()
        else:
            self.S_val = self.S.clone()
            self.G_val = self.G.clone()
        
        # loss weights
        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_neighborhood_g1 = lambda_neighborhood_g1
        self.lambda_ct_islands = lambda_ct_islands
        self.lambda_getis_ord = lambda_getis_ord
        self.lambda_geary = lambda_geary
        self.lambda_moran = lambda_moran

        # density terms
        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(d, device=device, dtype=torch.float32)

        self.source_density_enabled = d_source is not None
        if self.source_density_enabled:
            self.d_source = torch.tensor(d_source, device=device, dtype=torch.float32)
        
        self._density_criterion = torch.nn.KLDivLoss(reduction="sum")

        # spatial weights for neighborhood weighting 
        self.voxel_weights = voxel_weights
        if self.voxel_weights is not None:
            self.voxel_weights = torch.tensor(voxel_weights, device=device, dtype=torch.float32)
       
        # cell type islands
        self.neighborhood_filter = neighborhood_filter
        if self.neighborhood_filter is not None:
            self.neighborhood_filter = torch.tensor(neighborhood_filter, device=device, dtype=torch.float32)
        
        self.ct_encode = ct_encode
        if self.ct_encode is not None:
            self.ct_encode = torch.tensor(ct_encode, device=device, dtype=torch.float32)

        # spatial autocorrelation weights
        self.spatial_weights = spatial_weights
        if self.spatial_weights is not None:
            self.spatial_weights = torch.tensor(spatial_weights, device=device, dtype=torch.float32)

        # precompute spatial local indicators for target preservation
        self.getis_ord_G_star_ref, self.moran_I_ref, self.gearys_C_ref = self._spatial_local_indicators(self.G_train)

        # initialize mapping matrix
        if adata_map is None:
            if self.random_state:
                np.random.seed(seed=self.random_state)
            self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        else:
            raise NotImplemented
            self.M = adata_map.X  # doesn't work. maybe apply inverse softmax

        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )

    def _spatial_local_indicators(self, G):
        """
        Compute spatial local indicators for given gene expression matrix.

        Args:
            G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
        
        Returns:
            Tuple of 3 Arrays: getis_ord_G_star, moran_I, gearys_C
        """
        getis_ord_G_star = None
        if self.lambda_getis_ord > 0:
            getis_ord_G_star = (self.spatial_weights @ G) / G.sum(axis=0)
        
        moran_I = None
        if self.lambda_moran > 0:
            z = (G - G.mean(axis=0))
            moran_I = (G.shape[0] * z * (self.spatial_weights @ z)) / (z * z).sum(axis=0)
        
        gearys_C = None
        if self.lambda_geary > 0:
            n_spots, n_genes = G.shape
            m2 = ((G - G.mean(axis=0)) ** 2).sum(axis=0) / (n_spots - 1)
            G_row_dup = G[None, :, :].expand(n_spots, n_spots, n_genes)
            G_col_dup = G[:, None, :].expand(n_spots, n_spots, n_genes)
            weighted_diff_sq = self.spatial_weights.unsqueeze(2) * ((G_row_dup - G_col_dup) ** 2)
            gearys_C = weighted_diff_sq.sum(dim=(0, 1)) / (2 * m2)
        
        return getis_ord_G_star, moran_I, gearys_C

    def _loss_fn(self, verbose=True):
        """
        Evaluates the loss function.

        Args:
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed. Default is True.

        Returns:
            Tuple of 5 Floats: Total loss, gv_loss, vg_loss, kl_reg, entropy_reg
        """
        G = self.G_train
        S = self.S_train
        M_probs = softmax(self.M, dim=1)
        G_pred = torch.matmul(M_probs.t(), S)
        
        # gene expression similarity terms
        gv_term = self.lambda_g1 * cosine_similarity(G_pred, G, dim=0).mean()
        vg_term = self.lambda_g2 * cosine_similarity(G_pred, G, dim=1).mean()
        expression_term = gv_term + vg_term
        main_loss = (gv_term / self.lambda_g1).tolist()
        vg_reg = (vg_term / self.lambda_g2).tolist()
        
        # density regularization terms
        if self.target_density_enabled:
            # KL wants the log in first argument
            if self.source_density_enabled:
                d_pred = torch.log(self.d_source @ M_probs)
            else:
                d_pred = torch.log(M_probs.sum(dim=0) / self.M.shape[0])
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)
            kl_reg = (density_term / self.lambda_d).tolist()
        else:
            density_term, kl_reg = 0, np.nan
        
        # entropy regularization terms
        entropy_term = self.lambda_r * -(torch.log(M_probs) * M_probs).sum()
        entropy_reg = (entropy_term / self.lambda_r).tolist()

        # l1 and l2 regularization terms
        l1_term = self.lambda_l1 * self.M.abs().sum()
        l1_reg = (l1_term / self.lambda_l1).tolist()
        l2_term = self.lambda_l2 * (self.M ** 2).sum()
        l2_reg = (l2_term / self.lambda_l2).tolist()

        # spatial neighborhood-based gene expression term
        if self.lambda_neighborhood_g1 > 0:
            gv_neighborhood_term = self.lambda_neighborhood_g1 * cosine_similarity(self.voxel_weights @ G_pred, 
                                                                                   self.voxel_weights @ G, dim=0).mean()
            gv_neighborhood_sim = (gv_neighborhood_term / self.lambda_neighborhood_g1).tolist()
        else:
            gv_neighborhood_term, gv_neighborhood_sim = 0, np.nan

        # cell type island enforcement
        if self.lambda_ct_islands > 0:
            ct_map = (M_probs.T @ self.ct_encode)
            ct_island_term = self.lambda_ct_islands * (torch.max((ct_map) - (self.neighborhood_filter @ ct_map),
                                                                 torch.tensor([0], dtype=torch.float32, device=self.device)).mean())
            ct_island_penalty = (ct_island_term / self.lambda_ct_islands).tolist()
        else:
            ct_island_term, ct_island_penalty = 0, np.nan

        # spatial autocorrelation metrics
        getis_ord_G_star_pred, moran_I_pred, gearys_C_pred = self._spatial_local_indicators(G_pred)

        getis_ord_term, moran_term, gearys_term = 0, 0, 0
        getis_ord_sim, moran_sim, gearys_sim = np.nan, np.nan, np.nan
        if self.lambda_getis_ord > 0:
            getis_ord_term = self.lambda_getis_ord * cosine_similarity(self.getis_ord_G_star_ref, getis_ord_G_star_pred, dim=0).mean()
            getis_ord_sim = (getis_ord_term / self.lambda_getis_ord).tolist()
        if self.lambda_moran > 0:
            moran_term = self.lambda_moran * cosine_similarity(self.moran_I_ref, moran_I_pred, dim=0).mean()
            moran_sim = (moran_term / self.lambda_moran).tolist()
        if self.lambda_geary > 0:
            gearys_term = self.lambda_geary * cosine_similarity(self.gearys_C_ref, gearys_C_pred, dim=0).mean()
            gearys_sim = (gearys_term / self.lambda_geary).tolist()

        # total loss aggregation
        total_loss = -expression_term \
                 + density_term + entropy_term \
                 + l1_term + l2_term \
                 + ct_island_term - gv_neighborhood_term \
                 - getis_ord_term - moran_term - gearys_term

        if verbose:
            term_numbers = [
                main_loss,
                vg_reg, 
                kl_reg, 
                entropy_reg, 
                l1_reg, 
                l2_reg, 
                gv_neighborhood_sim, 
                ct_island_penalty, 
                getis_ord_sim, 
                moran_sim, 
                gearys_sim
            ]
            term_names = [
                "Gene-voxel score", 
                "Voxel-gene score", 
                "Cell densities reg", 
                "Entropy reg", 
                "L1 reg", 
                "L2 reg",
                "Spatial weighted score", 
                "Cell type islands penalty", 
                "Getis-Ord score", 
                "Moran score", 
                "Geary score"
            ]

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        return total_loss, main_loss, vg_reg, kl_reg, entropy_reg

    def _val_loss_fn(self, verbose=False):
        """
        Evaluates the validation loss function. Used during hyperparameter tuning.

        Args:
            verbose (bool): Optional. Whether to print the val results. If True, the loss for each individual term is printed as. Default is True.

        Returns:
            Tuple of 5 Floats: expression_sim, gv_sim, sp_sparsity_weighted_gv_sim, entropy
        """
        G = self.G_train
        S = self.S_train
        M_probs = softmax(self.M, dim=1)
        G_pred = torch.matmul(M_probs.t(), S)

        gv_sim = cosine_similarity(G_pred, G, dim=0).mean().tolist()
        vg_sim = cosine_similarity(G_pred, G, dim=1).mean().tolist()
        expression_sim = gv_sim + vg_sim

        gene_sparsity = 1 - ((G != 0).sum(axis=0) / G.shape[0]).reshape((-1,))
        sp_sparsity_weighted_gv_sim = ((cosine_similarity(G_pred, G, dim=0) * (1-gene_sparsity)) / (1-gene_sparsity).sum()).sum().tolist()

        entropy = -((torch.log(M_probs) * M_probs).sum(axis=1) / np.log(M_probs.shape[1])).mean().tolist()

        if verbose:
            term_numbers = [
                gv_sim, 
                sp_sparsity_weighted_gv_sim, 
                entropy
            ]
            term_names = [
                "Val gene-voxel score", 
                "Val gene-voxel sparsity-weighted score", 
                "Val map entropy"
            ]

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        return expression_sim, gv_sim, sp_sparsity_weighted_gv_sim, entropy

    def train(self, num_epochs, learning_rate=0.1, print_each=100, val_each=None):
        """
        Run the optimizer and returns the mapping outcome.

        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.
            val_each (int): Optional: Evaluate the val loss each val_each epochs. If None, the val loss is never computed. Default is None.
        Returns:
            output (ndarray): The optimized mapping matrix M (ndarray), with shape (number_cells, number_spots).
            training_history (dict): loss for each epoch
        """
        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)

        if print_each:
            logging.info(f"Printing scores every {print_each} epochs.")

        keys = ["total_loss", "main_loss", "vg_reg", "kl_reg", "entropy_reg"]
        val_keys = ["val_total_loss", "val_gene_sim", "val_sp_sparsity_weighted_sim", "val_entropy"]
        training_history = {key: [] for key in keys + val_keys}

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)

            loss = run_loss[0]

            training_history[keys[0]].append(run_loss[0].clone().detach().cpu().numpy())
            for i in range(1,len(keys)):
                training_history[keys[i]].append(run_loss[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if val_each is not None:
                with torch.no_grad():
                    if val_each is not None and t % val_each == 0:
                        val_loss = self._val_loss_fn(verbose=False)
                        for i in range(len(val_keys)):
                            training_history[val_keys[i]].append(val_loss[i])

        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            return output, training_history


class MapperConstrained:
    """
    Allows instantiating and running the optimizer for Tangram, with filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping and filter results.
    """

    def __init__(
        self,
        S,
        G,
        d,
        lambda_d=1,
        lambda_g1=1,
        lambda_g2=1,
        lambda_r=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        device="cpu",
        adata_map=None,
        random_state=None,
    ):
        """
        Instantiate the Tangram optimizer (with filtering).

        Args:
            S (ndarray): Single nuclei matrix, shape = (number_cell, number_genes).
            G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
                Spots can be single cells or they can contain multiple cells.
            d (ndarray): Spatial density of cells, shape = (number_spots,).
                         This array should satisfy the constraints d.sum() == 1.
            lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 1.
            lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
            lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 1.
            lambda_r (float): Optional. Entropy regularizer for the learned mapping matrix. An higher entropy promotes
                              probabilities of each cell peaked over a narrow portion of space.
                              lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
            lambda_count (float): Optional. Regularizer for the count term. Default is 1.
            lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Default is 1.
            target_count (int): Optional. The number of cells to be filtered. If None, this number defaults to the number of
                                voxels inferred by the matrix 'G'. Default is None.
            device (str or torch.device): Optional. Device is 'cpu'.
            adata_map (scanpy.AnnData): Optional. Mapping initial condition (for resuming previous mappings). Default is None.
            random_state (int): Optional. pass an int to reproduce training. Default is None.
        """
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)

        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(d, device=device, dtype=torch.float32)

        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_count = lambda_count
        self.lambda_f_reg = lambda_f_reg
        self._density_criterion = torch.nn.KLDivLoss(reduction="sum")
        self.random_state = random_state

        if adata_map is None:
            if self.random_state:
                np.random.seed(seed=self.random_state)
            self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        else:
            raise NotImplemented
            self.M = adata_map.X  # doesn't work. maybe apply inverse softmax

        if target_count is None:
            self.target_count = self.G.shape[0]
        else:
            self.target_count = target_count

        self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )

        self.F = np.random.normal(0, 1, S.shape[0])
        self.F = torch.tensor(
            self.F, device=device, requires_grad=True, dtype=torch.float32
        )

    def _loss_fn(self, verbose=True):
        """
        Evaluates the loss function.

        Args:
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed as:
                density_term, gene-voxel similarity term, voxel-gene similarity term. Default is True.

        Returns:
            Tuple of 7 Floats: Total loss, gv_loss, vg_loss, kl_reg, entropy_reg, count_reg, lambda_f_reg
        """
        M_probs = softmax(self.M, dim=1)
        F_probs = torch.sigmoid(self.F)

        M_probs_filtered = M_probs * F_probs[:, np.newaxis]

        if self.target_density_enabled:
            d_pred = torch.log(
                M_probs_filtered.sum(axis=0) / (F_probs.sum())
            )  # KL wants the log in first argument
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)
        else:
            density_term = None

        S_filtered = self.S * F_probs[:, np.newaxis]

        G_pred = torch.matmul(M_probs.t(), S_filtered)
        gv_term = self.lambda_g1 * cosine_similarity(G_pred, self.G, dim=0).mean()
        vg_term = self.lambda_g2 * cosine_similarity(G_pred, self.G, dim=1).mean()
        expression_term = gv_term + vg_term

        entropy_term = self.lambda_r * (torch.log(M_probs) * M_probs).sum()

        _count_term = F_probs.sum() - self.target_count
        count_term = self.lambda_count * torch.abs(_count_term)

        f_reg_t = F_probs - F_probs * F_probs
        f_reg = self.lambda_f_reg * f_reg_t.sum()

        main_loss = (gv_term / self.lambda_g1).tolist()
        kl_reg = (
            (density_term / self.lambda_d).tolist()
            if self.target_density_enabled
            else np.nan
        )
        entropy_reg = (entropy_term / self.lambda_r).tolist()
        main_loss = (gv_term / self.lambda_g1).tolist()
        vg_reg = (vg_term / self.lambda_g2).tolist()
        count_reg = (count_term / self.lambda_count).tolist()
        lambda_f_reg = (f_reg / self.lambda_f_reg).tolist()

        if verbose:
            term_numbers = [
                main_loss,
                vg_reg,
                kl_reg,
                entropy_reg,
                count_reg,
                lambda_f_reg,
            ]
            term_names = [
                "Score",
                "VG reg",
                "KL reg",
                "Entropy reg",
                "Count reg",
                "Lambda f reg",
            ]

            score_dict = dict(zip(term_names, term_numbers))
            clean_dict = {
                k: score_dict[k] for k in score_dict if not np.isnan(score_dict[k])
            }
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        total_loss = -expression_term - entropy_term + count_term + f_reg
        if self.target_density_enabled:
            total_loss = total_loss + density_term

        return (
            total_loss,
            main_loss,
            vg_reg,
            kl_reg,
            entropy_reg,
            count_reg,
            lambda_f_reg,
        )

    def train(self, num_epochs, learning_rate=0.1, print_each=100):
        """
        Run the optimizer and returns the mapping outcome.

        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.

        Returns:
            A tuple (output, F_out, training_history), with:
                M (ndarray): is the optimized mapping matrix, shape = (number_cells, number_spots).
                f (ndarray): is the optimized filter, shape = (number_cells,).
                training_history (dict): loss for each epoch
        """

        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.M, self.F], lr=learning_rate)

        keys = [
            "total_loss",
            "main_loss",
            "vg_reg",
            "kl_reg",
            "entropy_reg",
            "count_reg",
            "lambda_f_reg",
        ]
        values = [[] for i in range(len(keys))]
        training_history = {key: value for key, value in zip(keys, values)}

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)

            loss = run_loss[0]

            for i in range(len(keys)):
                training_history[keys[i]].append(str(run_loss[i]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            F_out = torch.sigmoid(self.F).cpu().numpy()
            return output, F_out, training_history
