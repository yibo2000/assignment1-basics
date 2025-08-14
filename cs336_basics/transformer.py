import torch
import torch.nn as nn
from jaxtyping import Float
from einops import reduce
from cs336_basics.basic import Linear

# Root Mean Square Layer Normalization
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter( torch.randn(d_model, device=device, dtype=dtype) ) # Float[Tensor, " d_model"]

    def set_gain(self, weight: Float[torch.Tensor, " d_model"]):
        self.gain = nn.Parameter(weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32) # upcast your input to torch.float32 before performing the normalization
        # RMS_a = torch.sqrt( torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps )
        RMS_a = torch.sqrt(reduce( torch.pow(x, 2), "... d_model -> ... 1", "mean") + self.eps)
        result = x * self.gain / RMS_a
        return result.to(in_dtype) # downcast to the original dtype
    

# Feedforward Neural Network
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
    w1_weight: Float[torch.Tensor, " d_ff d_model"],
    w2_weight: Float[torch.Tensor, " d_model d_ff"],
    w3_weight: Float[torch.Tensor, " d_ff d_model"],
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.l_w1 = Linear(self.d_model, self.d_ff, w1_weight)
        self.l_w2 = Linear(self.d_ff, self.d_model, w2_weight)
        self.l_w3 = Linear(self.d_model, self.d_ff, w3_weight)

    # x.shape -> d_model
    def SiLU(self, x: torch.Tensor): # SiLU(x) = x·sigmod(x)
        return x * torch.sigmoid(x)
    
    # x.shape -> d_model
    def GLU(self, x: torch.Tensor): # GLU(x, W1, W2) = sigmod(W1 x) ⊙ W2 x
        raise NotImplementedError
    
    def SwiGLU(self, x: torch.Tensor): # FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3 x)
        return self.l_w2.forward( (self.SiLU( self.l_w1.forward(x)  ) * (self.l_w3.forward(x))) )
    

# RoPE: Rotary Position Embedding
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape."""