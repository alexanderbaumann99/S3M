from pathlib import Path
from typing import List
from tempfile import TemporaryDirectory
from uuid import uuid4
import scipy.io as sio
from joblib.parallel import Parallel, delayed, cpu_count
from tqdm import tqdm
import numpy as np
import potpourri3d as pp3d
import trimesh
import networkx as nx
import torch
from dgl.geometry import farthest_point_sampler as fps
from model.utils.laplace_decomposition import laplace_decomposition
from model.utils.mesh_deformation import modify_vert


def compute_geodesic_distances(verts: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """
    Computes geodesic distance matrix on a chunk of source points
    """
    solver = pp3d.PointCloudHeatSolver(verts, 0.000000001)
    chunked_distance_matrix = [
        solver.compute_distance(i).astype(np.float32) for i in sources
    ]
    return np.array(chunked_distance_matrix, dtype=np.float32)


def compute_geodesic_distance_matrix(verts: np.ndarray) -> np.ndarray:
    """
    Computes geodesic distance matrix of a point cloud
    based on the heat method.
    Args:
        verts:      Points of point cloud
                    Shape: n x 3
    Returns:
        geo_mat:    Geodesic distance matrix
                    Shape: n x n
    """
    n_chunks = cpu_count()
    chunk_size = int(np.ceil(len(verts) / float(n_chunks)))
    sources = np.arange(len(verts))
    distance_mats = Parallel(n_chunks)(
        delayed(compute_geodesic_distances)(verts, sources[i : i + chunk_size])
        for i in range(0, len(verts), chunk_size)
    )
    geo_mat = np.vstack(distance_mats)
    return geo_mat


def write_mesh(path: str, verts: np.ndarray, faces: np.ndarray):
    """Saves mesh"""
    trimesh.Trimesh(verts, faces).export(path)


def deform_mesh(
    mesh_path: str,
    save_dir: str,
    number_deformations: int = 4,
    radius: int = 10,
    sigma: float = 70,
):
    """
    Data Augmentation: Deforms the mesh by Gaussians.
    """
    # Load mesh
    mesh = trimesh.load_mesh(mesh_path)
    # Get adjacency matrix
    adj = (nx.adjacency_matrix(trimesh.graph.vertex_adjacency_graph(mesh))).toarray()
    norms = mesh.vertex_normals.copy()
    init_vert = mesh.vertices.copy()
    vertex_colors = np.repeat([[1, 1, 1.0]], init_vert.shape[0], axis=0)
    new_vert = init_vert
    # Deform mesh number_deformations times
    for _ in range(number_deformations):
        idx = np.random.choice(init_vert.shape[0])
        gauss_mul = np.random.choice([-10, 10])
        # Augment vertices by Gaussiana
        new_vert, vertex_colors = modify_vert(
            idx, adj, radius, norms, new_vert, sigma, gauss_mul, vertex_colors
        )
    # Save  mesh with augmented vertices and same faces
    write_mesh(
        save_dir / f"{int(mesh_path.stem)}_augmented_{uuid4().hex[:8]}.ply",
        new_vert,
        mesh.faces,
    )


def copy_mesh(mesh: str, save_dir: str):
    "Copies and renames the mesh"
    verts, faces = trimesh.load(mesh).vertices, trimesh.load(mesh).faces
    write_mesh(
        save_dir / f"{int(mesh.stem)}_augmented_{uuid4().hex[:8]}.ply", verts, faces
    )


def rotate_mesh(mesh: str, save_dir: str, angle: float = None):
    """Data Augmentation: Rotating the mesh by a (random) angle"""
    verts, faces = trimesh.load(mesh).vertices, trimesh.load(mesh).faces
    # Define rotation matrix
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    c = np.cos(angle)
    s = np.sin(angle)
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # Apply rotation matrix w.r.t. center of shape
    mean = np.mean(verts, axis=0)
    rot_verts = np.dot((verts - mean), np.transpose(rot_mat)) + mean
    # Save rotated mesh
    write_mesh(
        save_dir / f"{int(mesh.stem)}_augmented_{uuid4().hex[:8]}.ply", rot_verts, faces
    )


def process_mesh(
    mesh: str,
    save_dir: str,
    num_eigen: int,
    distances: bool = False,
    n_points: int = 4000,
    scaling=None,
):
    """
    Takes a mesh, samples a subset of vertices, calculates the LBOs,
    and saves the data.
    """
    # Loading mesh
    verts, faces = trimesh.load(mesh).vertices, trimesh.load(mesh).faces
    if scaling is not None:
        verts *= np.array(scaling)
    # center shape
    mean = np.mean(verts, axis=0)
    verts -= mean
    # compute LBO decomposition
    evals, evecs, evecs_trans, _ = laplace_decomposition(verts, faces, num_eigen)
    # Subsample to same number of vertices using FPS
    verts_idx = fps(torch.Tensor(verts).view(1, -1, 3), n_points).numpy().squeeze()
    to_save = {
        "pos": verts[verts_idx],
        "evals": evals.flatten(),
        "evecs": evecs[verts_idx],
        "evecs_trans": evecs_trans[..., verts_idx],
    }
    # For inference, compute geodesic distances due to PMF
    if distances:
        dist = compute_geodesic_distance_matrix(verts[verts_idx])
        to_save["dist"] = dist
    # Save in mat file
    sio.savemat(save_dir / f"{mesh.stem}.mat", to_save)


def preprocess_train_set_main(
    file_paths: List[str],
    save_dir: str,
    n_eigen: int = 20,
    n_jobs: int = 1,
    n_points: int = 4000,
    scaling=None,
):
    """
    Preprocessing pipeline with augmentations, processing and saving for training purposes
    Args:
        file_paths:     List of paths of meshes
        save_dir:       Output directory
        n_eigen:        Number of used LBO eigenfunctions
        n_jobs:         Number of jobs for multiprocessing
        n_points:       Number of points to sample from mesh
        scaling:        Scaling vector for unit conversion
    """

    save_root_mat = Path(save_dir)

    with TemporaryDirectory() as save_root_mesh:
        save_root_mesh = Path(save_root_mesh)
        save_root_mat.mkdir(parents=True, exist_ok=True)

        meshes = [Path(p) for p in file_paths]
        meshes = sorted(meshes, key=lambda x: int(str(x.stem)))

        _ = Parallel(n_jobs=n_jobs)(
            delayed(copy_mesh)(mesh, save_root_mesh) for mesh in meshes
        )
        _ = Parallel(n_jobs=n_jobs)(
            delayed(rotate_mesh)(mesh, save_root_mesh, angle=None) for mesh in meshes
        )
        _ = Parallel(n_jobs=n_jobs)(
            delayed(deform_mesh)(mesh, save_root_mesh) for mesh in meshes
        )

        meshes = list(save_root_mesh.iterdir())
        _ = Parallel(n_jobs=n_jobs)(
            delayed(process_mesh)(
                mesh,
                save_root_mat,
                n_eigen,
                distances=False,
                n_points=n_points,
                scaling=scaling,
            )
            for mesh in meshes
        )


def preprocess_test_set_main(
    file_paths: List[str],
    save_dir: str,
    n_eigen: int = 20,
    n_jobs: int = 1,
    n_points: int = 4000,
    scaling: List[float] = None,
    distances: bool = False,
):
    """
    Preprocessing pipeline for inference purposes
    Args:
        file_paths:     List of paths of meshes
        save_dir:       Output directory
        n_eigen:        Number of used LBO eigenfunctions
        n_jobs:         Number of jobs for multiprocessing
        n_points:       Number of points to sample from mesh
        scaling:        Scaling vector for unit conversion
        distances:      If geodesic distances must be computed
    """
    save_root_mat = Path(save_dir)
    save_root_mat.mkdir(parents=True, exist_ok=True)

    meshes = [Path(p) for p in file_paths]
    meshes = sorted(meshes, key=lambda x: int(x.stem))

    _ = Parallel(n_jobs=n_jobs)(
        delayed(process_mesh)(
            mesh,
            save_root_mat,
            n_eigen,
            distances=distances,
            n_points=n_points,
            scaling=scaling,
        )
        for mesh in meshes
    )
