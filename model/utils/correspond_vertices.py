import os
from pathlib import Path
import scipy.io as sio
import numpy as np


def save_corresponding_vertices(mat_savedir: str, n_ref: int):
    """
    Corresponds the point clouds based on the predictions
    from the saved mat files.
    Args:
        mat_savedir:    directory where mat files from inference are saved
        n_ref:          index of reference shape
    """
    corres_verts = []

    # Load the vertices from the reference shape
    ref_mat = sio.loadmat(
        os.path.join(mat_savedir, "out_{:03d}_{:03d}.mat".format(n_ref, 0))
    )
    for vert in ref_mat["verts_x"]:
        corres_verts.append([vert])

    # Load and add corresponded vertices from other shapes
    for shape_idx in range(len(os.listdir(mat_savedir))):
        if shape_idx != n_ref:
            mat = sio.loadmat(
                os.path.join(mat_savedir, "out_{:03d}_{:03d}.mat".format(n_ref, shape_idx))
            )
            correspondence_idx = list(mat["y2x_pmf"].squeeze().astype(int))
            for idx_y, idx_ref in enumerate(correspondence_idx):
                corres_verts[idx_ref].append(mat["verts_y"][idx_y])

    # Convert to NumPy array for easier manipulation
    corres_verts = np.array(corres_verts)

    # Rearrange the array to match the desired format
    rearranged = np.empty_like(corres_verts)
    rearranged[:, :n_ref] = corres_verts[:, 1 : (n_ref + 1)]
    rearranged[:, n_ref] = corres_verts[:, 0]
    rearranged[:, (n_ref + 1) :] = corres_verts[:, (n_ref + 1) :]

    # Save the rearranged array as a NumPy binary file
    np.save(Path(mat_savedir).parent / "corres_verts.npy", rearranged)
