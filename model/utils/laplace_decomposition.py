import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import robust_laplacian

def laplace_decomposition(verts, faces, neig=50):
    W, A = robust_laplacian.mesh_laplacian(np.array(verts), np.array(faces))
    try:
        evals, evecs = eigsh(W, neig, A, 1e-6)
    except RuntimeError as e:
        c = np.ones(verts.shape[0]) * 1e-5
        damping = diags(c)
        W += damping
        evals, evecs = eigsh(W, neig, A, 1e-6)

    evecs = np.array(evecs, ndmin=2)
    evecs_trans = evecs.T @ A
    evals = np.array(evals)
    return evals, evecs, evecs_trans, np.sqrt(A.diagonal().sum())
