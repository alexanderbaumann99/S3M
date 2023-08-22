import torch
import numpy as np
from sslap import auction_solve


def pmf(
    corres_idx: torch.Tensor, dist_x: torch.Tensor, dist_y: torch.Tensor, var
) -> np.ndarray:
    """
    Implementation of product manifold filter (PMF).
    Args:
        points_x:       points of shape x
        points_y:       points of shape y
        corres_idx:     non-bijective correspondences between x and y
        dist_x:         geodesic distance of shape x
        dist_y:         geodesic distance of shape y
        var:            Variance of Gaussian Kernel 
    Returns:
        bij_corres:     bijective correspondences between x and y

    """
    k_x = torch.exp(-torch.pow(dist_x[:, corres_idx[:, 0]], 2) / (2 * var))
    k_y = torch.exp(-torch.pow(dist_y[:, corres_idx[:, 1]], 2) / (2 * var))

    density = torch.matmul(k_y, torch.transpose(k_x, 0, 1))

    density = density.cpu().numpy().astype(np.float64)
    bij_corres = auction_solve(density, problem="max")["sol"]

    return bij_corres
