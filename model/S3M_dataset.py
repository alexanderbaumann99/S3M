from typing import Tuple, List
from pathlib import Path
import os
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import KNNGraph
import scipy.io as sio
from dgl.geometry import farthest_point_sampler as fps


class S3MNetDataset(Dataset):
    """Torch dataset used for S3M"""

    def __init__(
        self,
        root: str,
        n_eigen: int,
        mode: str,
        n_points: int = None,
        ref_shape: int = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.mode = mode
        self.root = root
        self.n_eigen = n_eigen
        self.ref_shape = ref_shape

        if n_points is not None:
            self.sampling = Sampling(n_points, device)
        else:
            self.sampling = None
        self.graph_former = KNNGraph(k=10)

        self.samples = [Path(root) / Path(p) for p in os.listdir(root)]
        self.samples = sorted(self.samples, key=lambda x: int(x.stem.split("_")[0]))
        self.combinations = self.get_data_pairs()

    def len(self):
        return len(self.combinations)

    def get(self, idx):
        """
        From a given shape pair, the function creates two graphs with
        its associating LBO information.
        Args:
            idx
        Returns:
            graph_x:            Graph of shape x
            evals_x:            LBO eigenvalues of shape x
            evecs_x:            LBO eigenvectors of shape x
            evecs_trans_x:      Transposed LBO eigenvectors of shape x
            graph_y:            Graph of shape x
            evals_y:            LBO eigenvalues of shape x
            evecs_y:            LBO eigenvectors of shape x
            evecs_trans_y:      Transposed LBO eigenvectors of shape x
        """
        idx1, idx2 = self.combinations[idx]
        out1 = self.loader(self.samples[idx1])
        out2 = self.loader(self.samples[idx2])

        if self.sampling:
            evals_x, evecs_x, evecs_trans_x, verts_x, dist_x = self.sampling(out1)
            evals_y, evecs_y, evecs_trans_y, verts_y, dist_y = self.sampling(out2)
        else:
            evals_x, evecs_x, evecs_trans_x, verts_x, dist_x = out1
            evals_y, evecs_y, evecs_trans_y, verts_y, dist_y = out2

        graph_x = self.create_graph(verts_x)
        graph_y = self.create_graph(verts_y)

        if self.mode == "test":
            return [
                graph_x,
                evals_x,
                evecs_x,
                evecs_trans_x,
                dist_x,
                graph_y,
                evals_y,
                evecs_y,
                evecs_trans_y,
                dist_y,
            ]
        else:
            return [
                graph_x,
                evals_x,
                evecs_x,
                evecs_trans_x,
                graph_y,
                evals_y,
                evecs_y,
                evecs_trans_y,
            ]

    def get_data_pairs(self) -> List:
        """
        Defines the pairs of shapes between which a functional
        mapping will be calculated.
        """

        def _get_id(path):
            return Path(path).stem.split("_")[-1]

        pairs = []
        for i in range(len(self.samples)):
            for j in range(len(self.samples)):
                if self.mode == "test" and self.ref_shape is not None:
                    _ref_id = _get_id(self.samples[i])
                    if _ref_id == self.ref_shape:
                        pairs.append((i, j))
                else:
                    pairs.append((i, j))
        return pairs

    def create_graph(self, vertices: torch.Tensor) -> torch_geometric.data.Data:
        """
        Creates a graph from a given input point cloud through kNN.
        Args:
            vertices:       input point cloud
        Returns:
            graph:          output graph
        """
        graph = Data(pos=vertices)
        graph.x = vertices
        graph = self.graph_former(graph)
        edges = graph.edge_index
        edge_attr = torch.ones(edges.shape[1], dtype=torch.float32)
        graph.edge_attr = edge_attr
        return graph

    def loader(self, path: str) -> Tuple[torch.Tensor]:
        """
        Loads a mat file which was defined during preprocessing and returns
        the relevant data.
        Args:
            path:   path to mat file
        Returns:
            LBO eigenvalues
            LBO eigenvectors
            LBO transposed eigenvectors
            Vertices of shape
            Geodesic distance of vertices
        """
        mat = sio.loadmat(path)
        dist = torch.Tensor(mat["dist"]).float() if "dist" in mat else None
        return (
            torch.Tensor(mat["evals"]).flatten()[: self.n_eigen].float(),
            torch.Tensor(mat["evecs"])[:, : self.n_eigen].float(),
            torch.Tensor(mat["evecs_trans"])[: self.n_eigen, :].float(),
            torch.Tensor(mat["pos"]).float(),
            dist,
        )


class Sampling:
    """Class for subsampling a point cloud instance"""

    def __init__(self, num_vertices: int, device: str):
        self.num_vertices = num_vertices
        self.device = device

    def __call__(self, sample: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Subsamples a shape with its LBO information through farthest point
        sampling.
        Args:
            sample:     Includes information from S3MNetDataset.loader
        Returns:
            Subsampled LBO eigenvalues
            Subsampled LBO eigenvectors
            Subsampled LBO transposed eigenvectors
            Subsampled vertices of shape
            Geodesic distance of subsampled vertices
        """
        evals, evecs, evecs_trans, verts, dist = sample
        sampled_indices = (
            fps(verts.to(self.device).view(1, -1, 3), self.num_vertices)
            .reshape(-1)
            .cpu()
        )
        evecs = evecs[sampled_indices, :]
        evecs_trans = evecs_trans[:, sampled_indices]
        verts = verts[sampled_indices, :]
        if dist is not None:
            dist = dist[np.ix_(sampled_indices, sampled_indices)]

        return evals, evecs, evecs_trans, verts, dist
