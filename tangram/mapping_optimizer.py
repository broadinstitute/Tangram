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
        d=None,
        d_source=None,
        lambda_g1=1.0,
        lambda_d=0,
        lambda_g2=0,
        lambda_r=0,
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
            device (str or torch.device): Optional. Device is 'cpu'.
            adata_map (scanpy.AnnData): Optional. Mapping initial condition (for resuming previous mappings). Default is None.
            random_state (int): Optional. pass an int to reproduce training. Default is None.
        """
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)

        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(d, device=device, dtype=torch.float32)

        self.source_density_enabled = d_source is not None
        if self.source_density_enabled:
            self.d_source = torch.tensor(d_source, device=device, dtype=torch.float32)

        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self._density_criterion = torch.nn.KLDivLoss(reduction="sum")

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
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed as:
                density_term, gene-voxel similarity term, voxel-gene similarity term. Default is True.

        Returns:
            Tuple of 5 Floats: Total loss, gv_loss, vg_loss, kl_reg, entropy_reg
        """
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

        G_pred = torch.matmul(M_probs.t(), self.S)
        gv_term = self.lambda_g1 * cosine_similarity(G_pred, self.G, dim=0).mean()
        vg_term = self.lambda_g2 * cosine_similarity(G_pred, self.G, dim=1).mean()

        expression_term = gv_term + vg_term

        regularizer_term = self.lambda_r * (torch.log(M_probs) * M_probs).sum()

        main_loss = (gv_term / self.lambda_g1).tolist()
        kl_reg = (
            (density_term / self.lambda_d).tolist()
            if density_term is not None
            else np.nan
        )
        vg_reg = (vg_term / self.lambda_g2).tolist()

        entropy_reg = (regularizer_term / self.lambda_r).tolist()

        if verbose:

            term_numbers = [main_loss, vg_reg, kl_reg, entropy_reg]
            term_names = ["Score", "VG reg", "KL reg", "Entropy reg"]

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        total_loss = -expression_term - regularizer_term
        if density_term is not None:
            total_loss = total_loss + density_term

        return total_loss, main_loss, vg_reg, kl_reg, entropy_reg

    def train(self, num_epochs, learning_rate=0.1, print_each=100):
        """
        Run the optimizer and returns the mapping outcome.

        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.

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
