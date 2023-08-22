import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
import torch_geometric


class GNNFeatExtractor(torch.nn.Module):
    """
    Defines the GNN feature extractor including the three
    TAG Conv layers.
    """

    def __init__(self, feat_size: int, in_channels: int = 3):
        super().__init__()
        self.tag_conv_1 = TAGConv(in_channels, feat_size // 4, 1)
        self.tag_conv_2 = TAGConv(feat_size // 4, feat_size // 2, 2)
        self.tag_conv_3 = TAGConv(feat_size // 2, feat_size, 3)
        self.fc_layer = nn.Linear(feat_size, feat_size)
        self.activation = nn.ReLU()

    def forward(self, data: torch_geometric.data.Data):
        """
        Forward pass of GNN. Takes a graph and processes it through
        the TAG Conv Layers and outputs the extracted features.
        Args:
            data:   Input Graph
        Return:
            x:      Extracted features per node
        """
        feat, edge_index, edge_attr = (data.x, data.edge_index, data.edge_attr)
        feat = self.activation(self.tag_conv_1(feat, edge_index, edge_attr))
        feat = self.activation(self.tag_conv_2(feat, edge_index, edge_attr))
        feat = self.activation(self.tag_conv_3(feat, edge_index, edge_attr))

        feat = self.fc_layer(feat)
        feat = F.normalize(feat, dim=-1)

        return feat


class FunctionalMapNet(nn.Module):
    """
    Computes the functional map matrix representation.
    Code from: https://github.com/pvnieo/SURFMNet-pytorch
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        feat_x: torch.Tensor,
        feat_y: torch.Tensor,
        evecs_trans_x: torch.Tensor,
        evecs_trans_y: torch.Tensor,
    ):
        """One pass in functional map net.

        Arguments:
            feat_x:             Learned feature of shape X
                                Shape: batch-size x num-vertices x num-features
            feat_y:             Learned feature of shape Y
                                Shape: batch-size x num-vertices x num-features
            evecs_trans_x:      Transposed LBO eigenvectors of shape X
                                Shape: batch-size x num-eigenvectors x num-vertices
            evecs_trans_y:      Transposed LBO eigenvectors of shape Y
                                Shape: batch-size x num-eigenvectors x num-vertices

        Returns:
            C1:                 Matrix representation of functional correspondence.
                                Shape: batch_size x num-eigenvectors x num-eigenvectors.
            C2:                 Matrix representation of functional correspondence.
                                Shape: batch_size x num-eigenvectors x num-eigenvectors.
        """
        # compute linear operator matrix representation C1 and C2
        f_hat = torch.bmm(evecs_trans_x, feat_x)
        g_hat = torch.bmm(evecs_trans_y, feat_y)
        f_hat, g_hat = f_hat.transpose(1, 2), g_hat.transpose(1, 2)

        cs_xy = []
        for i in range(feat_x.size(0)):
            c = torch.linalg.inv(f_hat[i].t() @ f_hat[i]) @ f_hat[i].t() @ g_hat[i]
            cs_xy.append(c.t().unsqueeze(0))
        c_xy = torch.cat(cs_xy, dim=0)

        cs_yx = []
        for i in range(feat_x.size(0)):
            c = torch.linalg.inv(g_hat[i].t() @ g_hat[i]) @ g_hat[i].t() @ f_hat[i]
            cs_yx.append(c.t().unsqueeze(0))
        c_yx = torch.cat(cs_yx, dim=0)

        return c_xy , c_yx


class S3MNet(nn.Module):
    """
    Network to compute the functional maps based on GNN shape descriptors
    """

    def __init__(self, feat_size: int):
        super().__init__()
        self.feat_extractor = GNNFeatExtractor(feat_size, in_channels=3)
        self.funcmap_net = FunctionalMapNet()

    def forward(
        self,
        graph_x: torch_geometric.data.Data,
        graph_y: torch_geometric.data.Data,
        evecs_trans_x: torch.Tensor,
        evecs_trans_y: torch.Tensor,
    ):
        """
        Based on two shapes, the network extracts shape descriptors through a GNN and
        calculates the functional mapping.
        Args:
            graph_x:        Graph of shape X
            graph_y:        Graph of shape Y
            evecs_trans_x:  Transposed LBO eigenvectors of shape X
            evecs_trans_y:  Transposed LBO eigenvectors of shape Y
        Returns:
            C1:         Functional mapping from X to Y
            C2:         Functional mapping from Y to X
            feat_x:     Features per node of X
            feat_y:     Features per node of Y
        """
        batch_size = evecs_trans_x.shape[0]
        feat_x = self.feat_extractor(graph_x)
        feat_x = feat_x.view(batch_size, -1, feat_x.shape[1])
        feat_y = self.feat_extractor(graph_y)
        feat_y = feat_y.view(batch_size, -1, feat_y.shape[1])
        c_xy, c_yx = self.funcmap_net(feat_x, feat_y, evecs_trans_x, evecs_trans_y)

        return c_xy, c_yx, feat_x, feat_y
