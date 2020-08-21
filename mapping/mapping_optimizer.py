"""
Library for instantiating and running the optimizer for Tangram. The optimizer comes in two flavors,
which correspond to two different classes:
- Mapper: optimizer without filtering (i.e., all single cells are mapped onto space). At the end, the learned mapping
matrix M is returned.
- MapperConstrained: optimizer with filtering (i.e., only a subset of single cells are mapped onto space).
At the end, the learned mapping matrix M and the learned filter F are returned.
"""
import numpy as np
import torch
from torch.nn.functional import softmax, cosine_similarity


class Mapper:
    """
    Allows instantiating and running the optimizer for Tangram, without filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping result.
    """
    def __init__(self, S, G, d, lambda_d=1, lambda_g1=1, lambda_g2=1, lambda_r=0, device='cpu'):
        """
        Instantiate the Tangram optimizer (without filtering).
        Args:
            S (ndarray): Single nuclei matrix, shape = (number_cell, number_genes).
            G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
                Spots can be single cells or they can contain multiple cells.
            d (ndarray): Spatial density of cells, shape = (number_spots,).
                This array should satisfy the constraints d.sum() == 1.
            lambda_d (float): Optional. Hiperparameter for the density term of the optimizer. Default is 1.
            lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
            lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 1.
            lambda_r (float): Optional. Entropy regularizer for the learned mapping matrix. An higher entropy promotes
                probabilities of each cell peaked over a narrow portion of space.
                lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
            device (str or torch.device): Optional. Device is 'cpu'.
        """
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)
        self.d = torch.tensor(d, device=device, dtype=torch.float32)

        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self._density_criterion = torch.nn.KLDivLoss(reduction='sum')

        self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        self.M = torch.tensor(self.M, device=device, requires_grad=True, dtype=torch.float32)

    def _loss_fn(self, verbose=True):
        """
        Evaluates the loss function.
        Args:
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed as:
                density_term, gene-voxel similarity term, voxel-gene similarity term. Default is True.
        Returns:
            Total loss (float).
        """
        M_probs = softmax(self.M, dim=1)

        d_pred = torch.log(M_probs.sum(axis=0) / self.M.shape[0])  # KL wants the log in first argument
        density_term = self.lambda_d * self._density_criterion(d_pred, self.d)

        G_pred = torch.matmul(M_probs.t(), self.S)
        gv_term = self.lambda_g1 * cosine_similarity(G_pred, self.G, dim=0).mean()
        vg_term = self.lambda_g2 * cosine_similarity(G_pred, self.G, dim=1).mean()
        expression_term = gv_term + vg_term

        regularizer_term = self.lambda_r * (torch.log(M_probs) * M_probs).sum()

        if verbose:
            print((density_term / self.lambda_d).tolist(),
                  (gv_term / self.lambda_g1).tolist(),
                  (vg_term / self.lambda_g2).tolist(),)
        return density_term - expression_term - regularizer_term

    def train(self, num_epochs, learning_rate=0.1, print_each=100):
        """
        Run the optimizer and returns the mapping outcome.
        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.
        Returns:
            The optimized mapping matrix M (ndarray), with shape (number_cells, number_spots).
        """
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                loss = self._loss_fn(verbose=False)
            else:
                loss = self._loss_fn(verbose=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            return output


class MapperConstrained:
    """
    Allows instantiating and running the optimizer for Tangram, with filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping and filter results.
    """
    def __init__(self, S, G, d, lambda_d=1, lambda_g1=1, lambda_g2=1, lambda_r=0, lambda_count=1, lambda_f_reg=1,
                 device='cpu', target_count=None):
        """
        Instantiate the Tangram optimizer (with filtering).
        Args:
            S (ndarray): Single nuclei matrix, shape = (number_cell, number_genes).
            G (ndarray): Spatial transcriptomics matrix, shape = (number_spots, number_genes).
                Spots can be single cells or they can contain multiple cells.
            d (ndarray): Spatial density of cells, shape = (number_spots,).
                This array should satisfy the constraints d.sum() == 1.
            lambda_d (float): Optional. Hiperparameter for the density term of the optimizer. Default is 1.
            lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
            lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 1.
            lambda_r (float): Optional. Entropy regularizer for the learned mapping matrix. An higher entropy promotes
                probabilities of each cell peaked over a narrow portion of space.
                lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
            lambda_count (float): Optional. Regularizer for the count term. Default is 1.
            lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Default is 1.
            target_count (int): Optional. The number of cells to be filtered. If None, this number defaults to the number of
                voxels inferred by the matrix 'G'. Default is None.
        """
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)
        self.d = torch.tensor(d, device=device, dtype=torch.float32)

        self.lambda_d = lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.lambda_count = lambda_count
        self.lambda_f_reg = lambda_f_reg
        self._density_criterion = torch.nn.KLDivLoss(reduction='sum')

        if target_count is None:
            self.target_count = self.G.shape[0]
        else:
            self.target_count = target_count

        self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        self.M = torch.tensor(self.M, device=device, requires_grad=True, dtype=torch.float32)

        self.F = np.random.normal(0, 1, S.shape[0])
        self.F = torch.tensor(self.F, device=device, requires_grad=True, dtype=torch.float32)

    def _loss_fn(self, verbose=True):
        """
        Evaluates the loss function.
        Args:
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed as:
                density_term, gene-voxel similarity term, voxel-gene similarity term. Default is True.
        Returns:
            Total loss (float).
        """
        M_probs = softmax(self.M, dim=1)
        F_probs = torch.sigmoid(self.F)

        M_probs_filtered = M_probs * F_probs[:, np.newaxis]
        d_pred = torch.log(M_probs_filtered.sum(axis=0) / (F_probs.sum()))  # KL wants the log in first argument
        density_term = self.lambda_d * self._density_criterion(d_pred, self.d)

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

        if verbose:
            print((density_term / self.lambda_d).tolist(),
                  (gv_term / self.lambda_g1).tolist(),
                  (vg_term / self.lambda_g2).tolist(),
                  (count_term / self.lambda_count).tolist(),
                  (f_reg / self.lambda_f_reg).tolist()
                  )

        return density_term - expression_term - regularizer_term + count_term + f_reg

    def train(self, num_epochs, learning_rate=0.1, print_each=100):
        """
        Run the optimizer and returns the mapping outcome.
        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.
        Returns:
            A tuple (M, f), with:
                M (ndarray) is the optimized mapping matrix, shape = (number_cells, number_spots).
                f (ndarray) is the optimized filter, shape = (number_cells,).
        """
        optimizer = torch.optim.Adam([self.M, self.F], lr=learning_rate)

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                loss = self._loss_fn(verbose=False)
            else:
                loss = self._loss_fn(verbose=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            F_out = torch.sigmoid(self.F).cpu().numpy()
            return output, F_out
