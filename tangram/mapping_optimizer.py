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
from sklearn.metrics import auc

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
        lambda_sparsity_g1=0,
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
            lambda_sparsity_g1 (float): Optional. Strength of sparsity weighted gene expression comparison. Default is 0.
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

        self.S_all = torch.tensor(S, device=device, dtype=torch.float32)
        self.G_all = torch.tensor(G, device=device, dtype=torch.float32)

        if train_genes_idx is not None:
            self.S_train = self.S_all[:,train_genes_idx].clone()
            self.G_train = self.G_all[:,train_genes_idx].clone()
        else:
            self.S_train = self.S_all.clone()
            self.G_train = self.G_all.clone()
        if val_genes_idx is not None:
            self.S_val = self.S_all[:,val_genes_idx].clone()
            self.G_val = self.G_all[:,val_genes_idx].clone()
        else:
            self.S_val = None
            self.G_val = None

        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(d, device=device, dtype=torch.float32)

        self.source_density_enabled = d_source is not None
        if self.source_density_enabled:
            self.d_source = torch.tensor(d_source, device=device, dtype=torch.float32)
        
        self._density_criterion = torch.nn.KLDivLoss(reduction="sum")

        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_sparsity_g1 = lambda_sparsity_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

        self.lambda_neighborhood_g1 = lambda_neighborhood_g1
        self.voxel_weights = voxel_weights
        if self.voxel_weights is not None: 
            self.voxel_weights = torch.tensor(voxel_weights, device=device, dtype=torch.float32)
        else:
            self.voxel_weights = torch.zeros((self.G_train.shape[0],self.G_train.shape[0]), device=device, dtype=torch.float32)

        self.lambda_ct_islands = lambda_ct_islands
        self.neighborhood_filter = neighborhood_filter
        if self.neighborhood_filter is not None: 
            self.neighborhood_filter = torch.tensor(neighborhood_filter, device=device, dtype=torch.float32)
        self.ct_encode = ct_encode
        if self.ct_encode is not None: 
            self.ct_encode = torch.tensor(ct_encode, device=device, dtype=torch.float32)

        self.spatial_weights = spatial_weights
        if self.spatial_weights is not None:
            self.spatial_weights = torch.tensor(spatial_weights, device=device, dtype=torch.float32)
        
        self.lambda_getis_ord = lambda_getis_ord
        if self.lambda_getis_ord > 0:
            self.G_star = (self.spatial_weights @ self.G_train) / self.G_train.sum(axis=0)
        else:
            self.G_star = torch.zeros_like(self.G_train, device=device, dtype=torch.float32)

        self.lambda_moran = lambda_moran
        if self.lambda_moran > 0:
            z = (self.G_train - self.G_train.mean(axis=0))
            self.moran_I = (self.G_train.shape[0] * z * (self.spatial_weights @ z)) / (z * z).sum(axis=0)
        else:
            self.moran_I = torch.zeros_like(self.G_train, device=device, dtype=torch.float32)
        
        self.lambda_geary = lambda_geary
        if self.lambda_geary > 0:
            m2 = ((self.G_train - self.G_train.mean(axis=0)) ** 2).sum(axis=0) / (self.G_train.shape[0] - 1)
            G_row_dup = self.G_train[None,:,:].expand(self.G_train.shape[0], self.G_train.shape[0], self.G_train.shape[1])
            G_col_dup = self.G_train[:,None,:].expand(self.G_train.shape[0], self.G_train.shape[0], self.G_train.shape[1])
            self.gearys_C = 1 / m2.unsqueeze(1) * torch.diagonal(self.spatial_weights @ ((G_row_dup - G_col_dup) ** 2), 0)
        else:
            self.gearys_C = torch.zeros_like(self.G_train, device=device, dtype=torch.float32)

        self.random_state = random_state

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

        if self.target_density_enabled and self.source_density_enabled:
            d_pred = torch.log(
                self.d_source @ M_probs
            )  # KL wants the log in first argument
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)

        elif self.target_density_enabled and not self.source_density_enabled:
            d_pred = torch.log(
                M_probs.sum(axis=0) / self.M.shape[0]
            )  # KL wants the log in first argument
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)
        else:
            density_term = None

        G_pred = torch.matmul(M_probs.t(), S)
        gv_term = self.lambda_g1 * cosine_similarity(G_pred, G, dim=0).mean()
        gv_neighborhood_term = self.lambda_neighborhood_g1 * cosine_similarity(self.voxel_weights @ G_pred, 
                                                                               self.voxel_weights @ G, dim=0).mean()
        mask = G != 0
        gene_sparsity = mask.sum(axis=0) / G.shape[0]
        gene_sparsity = 1 - gene_sparsity.reshape((-1,))
        sp_sparsity_weighted_score = self.lambda_sparsity_g1 * ((cosine_similarity(G_pred, G, dim=0) * (1-gene_sparsity)) / (1-gene_sparsity).sum()).sum()
        
        vg_term = self.lambda_g2 * cosine_similarity(G_pred, G, dim=1).mean()

        expression_term = gv_term + gv_neighborhood_term + vg_term + sp_sparsity_weighted_score

        regularizer_term = self.lambda_r * (torch.log(M_probs) * M_probs).sum()
        l1_regularizer_term = self.lambda_l1 * self.M.abs().sum() 
        l2_regularizer_term = self.lambda_l2 * (self.M ** 2).sum()

        main_loss = (gv_term / self.lambda_g1).tolist()
        gv_neighborhood = (
            (gv_neighborhood_term / self.lambda_sparsity_g1).tolist()
            if self.lambda_sparsity_g1 != 0
            else np.nan
        )
        kl_reg = (
            (density_term / self.lambda_d).tolist()
            if density_term is not None
            else np.nan
        )
        vg_reg = (vg_term / self.lambda_g2).tolist()

        entropy_reg = (regularizer_term / self.lambda_r).tolist()
        l1_reg = (l1_regularizer_term / self.lambda_l1).tolist()
        l2_reg = (l2_regularizer_term / self.lambda_l2).tolist()

        if self.lambda_ct_islands > 0:
            ct_map = (M_probs.T @ self.ct_encode)
            ct_island_penalty = self.lambda_ct_islands * (torch.max((ct_map) - (self.neighborhood_filter @ ct_map), 
                                                        torch.tensor([0], dtype=torch.float32, device=self.device)).mean())
            ct_island_penalty_report = (ct_island_penalty / self.lambda_ct_islands).tolist()
        else:
            ct_island_penalty = 0
            ct_island_penalty_report = np.nan

        if self.lambda_getis_ord > 0:
            G_star_pred = (self.spatial_weights @ G_pred) / (G_pred.sum(axis=0))
            G_star_sim = self.lambda_getis_ord * cosine_similarity(self.G_star, G_star_pred, dim=0).mean()
            G_star_sim_report = (G_star_sim / self.lambda_getis_ord).tolist()
        else:
            G_star_sim = 0
            G_star_sim_report = np.nan

        if self.lambda_moran > 0:
            z = (G_pred - G_pred.mean(axis=0))
            moran_I_pred = (G_pred.shape[0] * z * (self.spatial_weights @ z)) / (z * z).sum(axis=0)
            moran_I_sim = self.lambda_moran * cosine_similarity(self.moran_I, moran_I_pred, dim=0).mean()
            moran_I_sim_report = (moran_I_sim / self.lambda_moran).tolist()
        else:
            moran_I_sim = 0
            moran_I_sim_report = np.nan

        if self.lambda_geary > 0:
            m2 = ((G_pred - G_pred.mean(axis=0)) ** 2).sum(axis=0) / (G_pred.shape[0] - 1)
            G_row_dup = G_pred[None,:,:].expand(G_pred.shape[0], G_pred.shape[0], G_pred.shape[1])
            G_col_dup = G_pred[:,None,:].expand(G_pred.shape[0], G_pred.shape[0], G_pred.shape[1])
            gearys_C_pred = 1 / m2.unsqueeze(1) * torch.diagonal(self.spatial_weights @ ((G_row_dup - G_col_dup) ** 2), 0)
            gearys_C_sim = self.lambda_geary * cosine_similarity(self.gearys_C, gearys_C_pred, dim=0).mean()
            gearys_C_sim_report = (gearys_C_sim / self.lambda_geary).tolist()
        else:
            gearys_C_sim = 0
            gearys_C_sim_report = np.nan

        total_loss = -expression_term - regularizer_term 
        total_loss += l1_regularizer_term 
        total_loss += l2_regularizer_term
        if density_term is not None:
            total_loss += density_term
        total_loss += ct_island_penalty
        total_loss -= G_star_sim
        total_loss -= moran_I_sim
        total_loss -= gearys_C_sim

        if verbose:
            term_numbers = [main_loss, vg_reg, kl_reg, 
                            entropy_reg, l1_reg, l2_reg, 
                            gv_neighborhood, ct_island_penalty_report, G_star_sim_report, moran_I_sim_report, gearys_C_sim_report]
            term_names = ["Gene-voxel score", "Voxel-gene score", "Cell densities reg", 
                          "Entropy reg", "L1 reg", "L2 reg", 
                          "Spatial weighted score", "Cell type islands score", "Getis-Ord G* score", "Moran\'s I score", "Geary\'s C score"]

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
        Evaluates the val loss function. Used during hyperparameter tuning.

        Args:
            verbose (bool): Optional. Whether to print the val results. If True, the loss for each individual term is printed as:
                density_term, gene-voxel similarity term, voxel-gene similarity term. Default is True.

        Returns:
            Tuple of 5 Floats: total_loss, gene_score, sp_sparsity_weighted_score, auc_score, prob_entropy
        """

        G = self.G_val
        S = self.S_val
        
        M_probs = softmax(self.M, dim=1)
        G_pred = torch.matmul(M_probs.t(), S)
        
        gv_scores = cosine_similarity(G_pred, G, dim=0)
        vg_scores = cosine_similarity(G_pred, G, dim=1)

        total_loss = (gv_scores.mean() + vg_scores.mean()).tolist()
        gene_score = gv_scores.mean().tolist()
        
        mask = G != 0
        gene_sparsity = mask.sum(axis=0) / G.shape[0]
        gene_sparsity = 1 - gene_sparsity.reshape((-1,))
        sp_sparsity_weighted_score = ((gv_scores * (1-gene_sparsity)) / (1-gene_sparsity).sum()).sum().tolist()

        xs = list(gv_scores.clone().detach().cpu().numpy())
        ys = list(gene_sparsity.clone().detach().cpu().numpy())
        pol_deg = 2
        pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
        pol_xs = np.linspace(0, 1, 10)  # x linearly spaced
        pol = np.poly1d(pol_cs)  # build polynomial as function
        pol_ys = [pol(x) for x in pol_xs]  # compute polys
        if pol_ys[0] > 1:
            pol_ys[0] = 1
        # if real root when y = 0, add point (x, 0):
        roots = pol.r
        root = None
        for i in range(len(roots)):
            if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
                root = roots[i]
                break
        if root is not None:
            pol_xs = np.append(pol_xs, root)
            pol_ys = np.append(pol_ys, 0)       
        np.append(pol_xs, 1)
        np.append(pol_ys, pol(1))
        # remove point that are out of [0,1]
        del_idx = []
        for i in range(len(pol_xs)):
            if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
                del_idx.append(i)
        pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
        pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]
        # Compute are under the curve of polynomial
        auc_score = np.real(auc(pol_xs, pol_ys)).tolist()

        prob_entropy = -((torch.log(M_probs) * M_probs).sum(axis=1) / np.log(M_probs.shape[1])).mean().tolist()

        if verbose:
            term_numbers = [total_loss, gene_score, sp_sparsity_weighted_score, auc_score, prob_entropy]
            term_names = ["total_loss", "gene_score", "sp_sparsity_weighted_score", "auc_score", "prob_entropy"]

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        return total_loss, gene_score, sp_sparsity_weighted_score, auc_score, prob_entropy

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
        if val_each is not None:
            assert(self.S_val is not None and self.G_val is not None)

        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)

        if print_each:
            logging.info(f"Printing scores every {print_each} epochs.")

        keys = ["total_loss", "main_loss", "vg_reg", "kl_reg", "entropy_reg"]
        val_keys = ["val_total_loss", "val_gene_score", "val_sp_sparsity_weighted_score", "val_auc_score", "val_prob_entropy"]
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

        regularizer_term = self.lambda_r * (torch.log(M_probs) * M_probs).sum()

        _count_term = F_probs.sum() - self.target_count
        count_term = self.lambda_count * torch.abs(_count_term)

        f_reg_t = F_probs - F_probs * F_probs
        f_reg = self.lambda_f_reg * f_reg_t.sum()

        main_loss = (gv_term / self.lambda_g1).tolist()
        kl_reg = (
            (density_term / self.lambda_d).tolist()
            if density_term is not None
            else np.nan
        )
        entropy_reg = (regularizer_term / self.lambda_r).tolist()
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

        total_loss = -expression_term - regularizer_term + count_term + f_reg
        if density_term is not None:
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
