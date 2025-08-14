import torch
import torch.nn as nn
import math
from jaxtyping import Float
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                weight: Float[torch.Tensor, " d_out d_in"] | None = None,
                device: torch.device | None = None, dtype: torch.dtype | None = None,):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super(Linear, self).__init__()
        # initialize the weights, y = Wx, W [d_out, d_in], x[d_in]
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight if weight != None else nn.Parameter(nn.init.trunc_normal_(
            tensor = torch.rand( out_features, in_features, device=device, dtype=dtype),
            mean = 0.0, std = math.sqrt(2/(in_features+out_features)), a = -3.0, b = 3.0
        ))
    
    def set_weight(self, weight: Float[torch.Tensor, " d_out d_in"]):
        self.weight = nn.Parameter(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "Apply the linear transformation to the input."
        "y = Wx"
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out" )


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Construct an embedding module. This function should accept the following parameters:
        - num_embeddings: int Size of the vocabulary
        - embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        - device: torch.device | None = None Device to store the parameters on
        - dtype: torch.dtype | None = None Data type of the parameters
        """
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            tensor = torch.rand( num_embeddings, embedding_dim, device=device, dtype=dtype),
            mean = 0.0, std = 1.0, a = -3.0, b = 3.0
        ))

    def set_weight(self, weight: Float[torch.Tensor, " vocab_size d_model"]):
        self.weight = nn.Parameter(weight)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        "Lookup the embedding vectors for the given token IDs."
        ""
        return self.weight[token_ids]

