import os
from typing import List, Tuple
import numpy as np
import trimesh


def get_correspondended_vertices(
    idx_train_shapes: List[int], path: str
) -> Tuple[np.ndarray, int]:
    """
    Get corresponded vertices from predictions
    Args:
        idx_train_shapes:               Indices of shapes used to build the SSM
        path:                           Path to saved npy file with correspondences
    Return:
        corresponded_vertices_train:    Corresponded vertices of specific shapes
        n_points:                       Number of points in SSM (flattened)
    """
    corresponded_vertices_all = np.transpose(np.load(path), (0, 2, 1))
    n_shapes = corresponded_vertices_all.shape[2]
    corresponded_vertices_all = corresponded_vertices_all.reshape(-1, n_shapes)
    corresponded_vertices_train = corresponded_vertices_all[:, idx_train_shapes]
    n_points = corresponded_vertices_train.shape[0]

    return corresponded_vertices_train, n_points


def get_target_point_cloud(path: str, val_idx: int) -> trimesh.Trimesh:
    """
    Get original target point cloud to measure error.
    Args:
        path:       Path to directory which contain the original meshes
        val_idx:    Index of target shape
    Returns:
        target:     Centered target point cloud
    """
    target = trimesh.load(os.path.join(path, "{:03d}.ply".format(val_idx))).vertices
    target -= np.mean(target, axis=0)
    return target


def get_test_point_cloud(path: str, val_idx: int) -> np.ndarray:
    """
    Get test point cloud to be reconstructed by the SSM.
    Already in correspondence with other shapes.
    Args:
        path:               Path to saved npy file with correspondences
        val_idx:            Index of test shape
    Returns:
        test_point_cloud:   Flattened point cloud in correspondence
    """
    corresponded_vertices_all = np.transpose(np.load(path), (0, 2, 1))
    test_point_cloud = corresponded_vertices_all[..., val_idx]
    return test_point_cloud.reshape(-1)
