import numpy as np
import torch
from torch import nn


class S3MNetLoss(nn.Module):
    """
    Calculate the loss as presented in the SURFMNet paper.
    Code from: https://github.com/pvnieo/SURFMNet-pytorch
    """

    def __init__(
        self,
        w_bij: float = 1,
        w_orth: float = 1,
        w_lap: float = 1e-3,
        w_pre: float = 1e2,
        sub_pre: float = 0.2,
    ):
        """
        Args::
            w_bij:      Bijectivity penalty weight 
            w_orth:     Orthogonality penalty weight
            w_lap:      Laplacian commutativity penalty weight
            w_pre:      Descriptor preservation via commutativity penalty weight
            sub_pre:    Percentage of subsampled vertices used to compute
                        descriptor preservation via commutativity penalty 
        """
        super().__init__()
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap
        self.w_pre = w_pre
        self.sub_pre = sub_pre

    def forward(
        self,
        c_xy,
        c_yx,
        feat_x,
        feat_y,
        evecs_x,
        evecs_y,
        evecs_trans_x,
        evecs_trans_y,
        evals_x,
        evals_y,
    ):
        """Compute soft error loss

        Args:
            c_xy:                   Matrix representation of functional correspondence
                                    Shape: batch_size x num-eigenvectors x num-eigenvectors
            c_yx:                   Matrix representation of functional correspondence
                                    Shape: batch_size x num-eigenvectors x num-eigenvectors
            feat_x:                 Learned feature od shape x.
                                    Shape: batch-size x num-vertices x num-features
            feat_y:                 Learned feature of shape y
                                    Shape: batch-size x num-vertices x num-features
            evecs__x:               Eigenvectors decomposition of shape x
                                    Shape: batch-size x num-vertices x num-eigenvectors
            evecs_y:                Eigenvectors decomposition of shape y
                                    Shape: batch-size x num-vertices x num-eigenvectors
            evecs_trans_x:          Inverse eigenvectors decomposition of shape x. defined as evecs_x.t() @ mass_matrix
                                    Shape: batch-size x num-eigenvectors x num-vertices
            evecs_trans_y:          Inverse eigenvectors decomposition of shape y. defined as evecs_y.t() @ mass_matrix
                                    Shape: batch-size x num-eigenvectors x num-vertices
            evals_x:                Eigenvalues of shape x
                                    Shape: batch-size x num-eigenvectors
            evals_y:                Eigenvalues of shape y
                                    Shape: batch-size x num-eigenvectors
        Returns:
            total_loss:             Total weighted loss
        """
        criterion = FrobeniusLoss()
        eye = torch.eye(c_xy.size(1), c_xy.size(2)).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=c_xy.size(0), dim=0).to(
            c_xy.device
        )

        # Bijectivity penalty
        bijectivity_penalty = criterion(torch.bmm(c_xy, c_yx), eye_batch) + criterion(
            torch.bmm(c_yx, c_xy), eye_batch
        )
        bijectivity_penalty *= self.w_bij

        # Orthogonality penalty
        orthogonality_penalty = criterion(
            torch.bmm(c_xy.transpose(1, 2), c_xy), eye_batch
        )
        orthogonality_penalty += criterion(
            torch.bmm(c_yx.transpose(1, 2), c_yx), eye_batch
        )
        orthogonality_penalty *= self.w_orth

        # Laplacian commutativity penalty
        laplacian_penalty = criterion(
            torch.einsum("abc,ac->abc", c_xy, evals_x),
            torch.einsum("ab,abc->abc", evals_y, c_xy),
        )
        laplacian_penalty += criterion(
            torch.einsum("abc,ac->abc", c_yx, evals_y),
            torch.einsum("ab,abc->abc", evals_x, c_yx),
        )
        laplacian_penalty *= self.w_lap

        # Descriptor preservation via commutativity
        num_desc = int(feat_x.size(2) * self.sub_pre)
        descs = np.random.choice(feat_x.size(2), num_desc)
        feat_x = feat_x[:, :, descs].transpose(1, 2).unsqueeze(2)
        feat_y = feat_y[:, :, descs].transpose(1, 2).unsqueeze(2)
        m_x = torch.einsum("abcd,ade->abcde", feat_x, evecs_x)
        m_x = torch.einsum("afd,abcde->abcfe", evecs_trans_x, m_x)
        m_y = torch.einsum("abcd,ade->abcde", feat_y, evecs_y)
        m_y = torch.einsum("afd,abcde->abcfe", evecs_trans_y, m_y)
        c_xy_expand = torch.repeat_interleave(
            c_xy.unsqueeze(1).unsqueeze(1), repeats=num_desc, dim=1
        )
        c_yx_expand = torch.repeat_interleave(
            c_yx.unsqueeze(1).unsqueeze(1), repeats=num_desc, dim=1
        )
        source1, target1 = torch.einsum(
            "abcde,abcef->abcdf", c_xy_expand, m_x
        ), torch.einsum("abcef,abcfd->abced", m_y, c_xy_expand)
        source2, target2 = torch.einsum(
            "abcde,abcef->abcdf", c_yx_expand, m_y
        ), torch.einsum("abcef,abcfd->abced", m_x, c_yx_expand)
        preservation_penalty = criterion(source1, target1) + criterion(source2, target2)
        preservation_penalty *= self.w_pre

        return (
            bijectivity_penalty
            + orthogonality_penalty
            + laplacian_penalty
            + preservation_penalty
        )


class FrobeniusLoss(nn.Module):
    """Forbenius distance for matrices"""

    def __init__(self):
        super().__init__()

    def forward(self, mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mat_1:  Matrix 1
                    Shape: n x n
            mat_2:  Matrix 2
                    Shape:  n x n
        Returns:
            frobenius_distance
        """
        frobenius_distance = torch.mean((mat_1 - mat_2) ** 2, axis=(1, 2))
        return torch.mean(frobenius_distance)
