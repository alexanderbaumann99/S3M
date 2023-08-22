import numpy as np


class SSM:
    def __init__(self, correspondences: np.ndarray) -> None:
        """
        Compute the SSM based on eigendecomposition.
        Args:
            correspondences:    Corresponded shapes
        """
        self.mean = np.mean(correspondences, 0)

        data_centered = correspondences - self.mean
        cov_dual = np.matmul(data_centered, data_centered.transpose()) / (
            data_centered.shape[0] - 1
        )
        evals, evecs = np.linalg.eigh(cov_dual)
        evecs = np.matmul(data_centered.transpose(), evecs)
        # Normalize the col-vectors
        evecs /= np.sqrt(np.sum(np.square(evecs), 0))

        # Sort
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]

        # Remove the last eigenpair (it should have zero eigenvalue)
        self.variances = evals[:-1]
        self.modes_norm = evecs[:, :-1]
        # Compute the modes scaled by corresp. std. dev.
        self.modes_scaled = np.multiply(self.modes_norm, np.sqrt(self.variances))

    def generate_random_samples(self, n_samples: int = 1, n_modes=None) -> np.ndarray:
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
        weights = np.random.standard_normal([n_samples, n_modes])
        samples = self.mean + np.matmul(weights, self.modes_scaled.transpose())
        return np.squeeze(samples)

    def get_reconstruction(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction
        """
        shape = shape.reshape(-1)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            if n_modes > self.length:
                n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        data_proj_re = data_proj.reshape(-1, 1)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)
        data_proj = self.mean + np.matmul(
            weights.transpose(1, 0), evecs.transpose(1, 0)
        )
        data_proj = data_proj.reshape(-1, 3)
        return data_proj
