import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,KarateClub
from torch_geometric.nn import GCNConv

# from _base_network import BaseEncoder
from ._base_network import BaseEncoder


class LSTMEncoder(BaseEncoder):
    """
    Parameters
    ----------
    input_dim: int
        Number of features of input.
    embed_dim: int
        Number of features of hidden layer.
    """

    def __init__(self, input_dim, embed_dim, device=None) -> None:
        super(LSTMEncoder, self).__init__(input_dim=input_dim,
                                          embed_dim=embed_dim,
                                          device=device)
        self.input_dim = input_dim
        self.hidden_dim = embed_dim
        self.device = device
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            bias=True,
                            batch_first=True
                            ).to(device=device)

    def forward(self, x) -> torch.Tensor:
        """

        Parameters
        ----------
        x:
            [Batch Size, Sequence Length, Features]
        """

        x = x.permute(0, 2, 1)
        output = self.embedding(x).permute(0, 2, 1)
        output, (_, _) = self.lstm(output)

        return output


class MLPEncoder(BaseEncoder):
    """
    Feed-forward neural networks----MLP

    """

    def __init__(self, input_dim, embed_dim, hidden_dim,
                 device=None) -> None:
        super(MLPEncoder, self).__init__(input_dim=input_dim,
                                         embed_dim=embed_dim,
                                         hidden_dim=hidden_dim,
                                         device=device)
        self.input_dim = input_dim
        self.embed_dim = embed_dim  # also is output_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, x) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        output = self.embedding(x)
        output = self.feedforward_conv1d(output)
        output = self.bn(output).permute(0, 2, 1)

        return output


        # # Positional encoding
        # W_pos = torch.empty((q_len, d_model), device=default_device())
        # nn.init.uniform_(W_pos, -0.02, 0.02)
        # self.W_pos = nn.Parameter(W_pos, requires_grad=True)

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len, device):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout).to(device=device)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 256)).to(device=device)  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02).to(device=device)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe
        # x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(BaseEncoder):
    """Transformer Encoder"""

    def __init__(self, input_dim, embed_dim, hidden_dim,
                 heads=8, blocks=3, max_length=5, device=None) -> None:
        super(TransformerEncoder, self).__init__(input_dim=input_dim,
                                                 embed_dim=embed_dim,
                                                 hidden_dim=hidden_dim,
                                                 device=device)
        self.input_dim = input_dim
        self.heads = heads
        self.embed_dim = embed_dim  # also is output_dim
        self.hidden_dim = hidden_dim
        self.blocks = blocks
        self.device = device
        self.attention = MultiHeadAttention(input_dim=embed_dim,
                                            output_dim=embed_dim,
                                            heads=heads,
                                            dropout_rate=0.0,
                                            device=device)
        # learnable
        self.pos_enc = LearnablePositionalEncoding(d_model=hidden_dim, dropout=0.1, max_len=max_length,device=device)

    def forward(self, x) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        output = self.embedding(x).permute(0, 2, 1)
        #加入位置编码
        output = self.pos_enc(output)
        for i in range(self.blocks):
            enc = self.attention(output)
            enc = enc.permute(0, 2, 1)
            output = self.feedforward_conv1d(enc)
            output += enc  # Residual connection
            output = self.bn(output).permute(0, 2, 1)

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi head attention mechanism

    Parameters
    ----------
    input_dim: int
        input dimension
    output_dim: int
        output dimension
    heads: int
        head numbers of multi head attention mechanism
    dropout_rate: float, int
        If not 0, append `Dropout` layer on the outputs of each LSTM layer
        except the last layer. Default 0. The range of dropout is (0.0, 1.0).

    """

    def __init__(self, input_dim, output_dim, heads=8, dropout_rate=0.1,
                 device=None) -> None:
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.device = device

        self.w_q = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.w_k = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.w_v = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.bn = nn.BatchNorm1d(num_features=output_dim, device=device)

    def forward(self, x) -> torch.Tensor:
        Q = self.w_q(x)  # [batch_size, seq_length, n_hidden]
        K = self.w_k(x)
        V = self.w_v(x)

        # Split and concat
        Q_ = torch.cat(torch.split(Q,
                                   split_size_or_sections=Q.shape[2]//self.heads,
                                   dim=2),
                       dim=0)
        K_ = torch.cat(torch.split(K,
                                   split_size_or_sections=K.shape[2]//self.heads,
                                   dim=2),
                       dim=0)
        V_ = torch.cat(torch.split(V,
                                   split_size_or_sections=V.shape[2]//self.heads,
                                   dim=2),
                       dim=0)
        # Multiplication # [num_heads*batch_size, seq_length, seq_length]
        output = torch.matmul(Q_, K_.permute(0, 2, 1))

        # Scale
        output = output / (K_.shape[-1] ** 0.5)

        # Activation  # [num_heads*batch_size, seq_length, seq_length]
        output = F.softmax(output, dim=1)

        # Dropouts
        output = F.dropout(output, p=self.dropout_rate)

        # Weighted sum # [num_heads*batch_size, seq_length, n_hidden/num_heads]
        output = torch.matmul(output, V_)

        # Restore shape
        output = torch.cat(torch.split(output,
                                       split_size_or_sections=output.shape[0]//self.heads,
                                       dim=0),
                           dim=2)  # [batch_size, seq_length, n_hidden]
        # Residual connection
        output += x  # [batch_size, seq_length, n_hidden]
        output = self.bn(output.permute(0, 2, 1)).permute(0, 2, 1)

        return output

# class GCNEncoder(torch.nn.Module):
#     def __init__(self):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = GCNConv(dataset.num_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)





if __name__ == '__main__':
    # dataset = torch.rand(5,100)
    # dataset = KarateClub()
    # print(dataset.x.shape)
    # print(dataset.edge_index.shape)
    # model = GCNEncoder()
    # out = model(dataset.x, dataset.edge_index)
    # print(out.shape)


    batch_size = 64
    input_dim = 100
    hidden_dim = 128
    input = torch.rand(64,10,100)
    mdoel = TransformerEncoder(input_dim, 64, hidden_dim)
    out = mdoel(input)
    print(out.shape)
