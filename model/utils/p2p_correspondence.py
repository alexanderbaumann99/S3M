# Code from: https://github.com/RobinMagnet/pyFM/tree/master
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from scipy import linalg


def icp_iteration(
    c_xy: np.ndarray,
    evecs_x: np.ndarray,
    evecs_y: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Performs an iteration of ICP.
    Conversion from a functional map to a pointwise map is done by comparing
    embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T.
    The diracs are transposed using the functional map or its adjoint.
    Args:
        c_xy:       Functional map in reduced basis
                    Shape: (k_y,k_x)
        evecs_x:    First k' eigenvectors of the first basis  (k_x'>k_x).
                    Shape: (n_x,k_x')
        evecs_y:    First k' eigenvectors of the second basis (k_y'>k_y)
                    Shape:(n_y,k_y')
        n_jobs:     Number of parallel jobs. Use -1 to use all processes
    Returns:
        c_refined:  An orthogonal functional map after one step of refinement
                    Shape: (k_y,k_x)
    """
    k_y, k_x = c_xy.shape
    p2p_yx = FM_to_p2p(c_xy, evecs_x, evecs_y, n_jobs=n_jobs)
    c_icp = p2p_to_FM(p2p_yx, evecs_x, evecs_y)
    u, _, vt = linalg.svd(c_icp)
    return u @ np.eye(k_y, k_x) @ vt


def icp_refine(
    c_xy: np.ndarray,
    evecs_x: np.ndarray,
    evecs_y: np.ndarray,
    nit: int = 10,
    tol: float = 1e-10,
    n_jobs: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Refine a functional map using the standard ICP algorithm.

    Args:
        c_xy:       Functional map in reduced basis
                    Shape: (k_y,k_x)
        evecs_x:    First k' eigenvectors of the first basis  (k_x'>k_x).
                    Shape: (n_x,k_x')
        evecs_y:    First k' eigenvectors of the second basis (k_y'>k_y)
                    Shape:(n_y,k_y')
        nit:        Number of iterations to perform. If not specified, uses the tol parameter
        tol:        Maximum change in a functional map to stop refinement
                    (only used if nit is not specified)
        n_jobs:     Number of parallel jobs. Use -1 to use all processes
        return_p2p: If True returns the vertex to vertex map from 2 to 1
    Returns:
        c_xy_icp:   ICP-refined functional map
        p2p_yx_icp: Only if return_p2p is set to True - the refined pointwise map
                    from basis 2 to basis 1
    """
    c_xy_curr = c_xy.copy()
    iteration = 1
    if verbose:
        start_time = time.time()

    if nit is not None and nit > 0:
        myrange = tqdm(range(nit)) if verbose else range(nit)
    else:
        myrange = range(10000)

    for i in myrange:
        c_xy_icp = icp_iteration(c_xy_curr, evecs_x, evecs_y, n_jobs=n_jobs)

        if nit is None or nit == 0:
            if verbose:
                print(
                    f"iteration : {1+i} - mean : {np.square(c_xy_curr - c_xy_icp).mean():.2e}"
                    f" - max : {np.max(np.abs(c_xy_curr - c_xy_icp)):.2e}"
                )
            if np.max(np.abs(c_xy_curr - c_xy_icp)) <= tol:
                break

        c_xy_curr = c_xy_icp.copy()

    if nit is None or nit == 0 and verbose:
        run_time = time.time() - start_time
        print(f"ICP done with {iteration:d} iterations - {run_time:.2f} s")

    p2p_yx_icp = FM_to_p2p(c_xy_icp, evecs_x, evecs_y, n_jobs=n_jobs)

    return p2p_yx_icp


def knn_query(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 1,
    return_distance: bool = False,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Query nearest neighbors.
    Args:
        x:                  First collection
                            Shape: n_x x p
        y:                  Second collection
                            Shape: n_y x p
        k:                  Number of neighbors to look for
        return_distance:    Whether to return the nearest neighbor distance
        n_job:              Number of parallel jobs. Set to -1 to use all processes
    Returns:
        dists:              Nearest neighbor distance. Only if return_distance = True
                            Shape: n_y x k
        matches:            Nearest neighbor
                            Shape: n_y x k
    """

    tree = NearestNeighbors(
        n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs
    )
    tree.fit(x)
    dists, matches = tree.kneighbors(y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches


def FM_to_p2p(
    c_xy: np.ndarray, evecs_x: np.ndarray, evecs_y: np.ndarray, n_jobs: int = -1
) -> np.ndarray:
    """
    Obtain a point to point map from a functional map C.
    Compares embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T
    Args:
        c_xy:       Functional map in reduced basis
                    Shape: (k_y,k_x)
        evecs_x:    First k' eigenvectors of the first basis  (k_x'>k_x).
                    Shape: (n_x,k_x')
        evecs_y:    First k' eigenvectors of the second basis (k_y'>k_y)
                    Shape:(n_y,k_y')
        n_jobs:     Number of parallel jobs. Use -1 to use all processes
    Returns:
        p2p_yx:     match vertex i on shape y to vertex p2p_yx[i] on shape x,
                    Shape: n_y
    """
    k_y, k_x = c_xy.shape

    assert (
        k_x <= evecs_x.shape[1]
    ), f"At least {k_x} should be provided, here only {evecs_x.shape[1]} are given"
    assert (
        k_y <= evecs_y.shape[1]
    ), f"At least {k_y} should be provided, here only {evecs_y.shape[1]} are given"

    emb1 = evecs_x @ c_xy.T
    emb2 = evecs_y

    p2p_yx = knn_query(emb1, emb2, k=1, n_jobs=n_jobs)
    return p2p_yx  # (n_y,)


def p2p_to_FM(
    p2p_yx: np.ndarray, evecs_x: np.ndarray, evecs_y: np.ndarray
) -> np.ndarray:
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).
    Can compute with the pseudo inverse of eigenvectors (if no subsampling) or least square.
    Args:
        p2p_yx:     Vertex to vertex map from target to source.
                    Shape: n_y
        evecs_x:    Eigenvectors on source mesh
                    Shape: n_x x k_x
        evecs_y:    Eigenvectors on target mesh. Possibly subsampled on the first dimension.
                    Shape: n_y x k_y
    Returns:
        c_xy:       (k_y,k_x) functional map corresponding to the p2p map given.
                    Solved with pseudo inverse if A2 is given, else using least square.
    """
    # Pulled back eigenvectors
    evecs_x_pb = (
        evecs_x[p2p_yx, :] if np.asarray(p2p_yx).ndim == 1 else p2p_yx @ evecs_x
    )

    # Solve with least square
    return linalg.lstsq(evecs_y, evecs_x_pb)[0]  # (k_y,k_x)
