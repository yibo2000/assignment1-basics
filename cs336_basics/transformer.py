import torch
import torch.nn as nn
from jaxtyping import Float, Int
from einops import reduce, einsum, rearrange
from cs336_basics.basic import Linear, Embedding

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
    

# x.shape -> d_model
def SiLU(x: torch.Tensor): # SiLU(x) = x·sigmod(x)
    return x * torch.sigmoid(x)


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
    def GLU(self, x: torch.Tensor): # GLU(x, W1, W2) = sigmod(W1 x) ⊙ W2 x
        raise NotImplementedError
    
    def SwiGLU(self, x: torch.Tensor): # FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3 x)
        return self.l_w2.forward( (SiLU( self.l_w1.forward(x)  ) * (self.l_w3.forward(x))) )
    

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
        self.device = device

        # buffer
        k = torch.arange(0, self.d_k, dtype=torch.float32)
        k = (k - k%2) / self.d_k # 0,0,2,2,4,4,...
        angles = self.theta ** -k # theta_i_k
        self.register_buffer("angles", angles, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape."""
        angles = self.angles.to(self.device) # load buffer
        angles = token_positions.unsqueeze(-1).float() * angles
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # x: x1, x2, x3, x4, ..., x_d-1, x_d
        # y: -x2, x1, -x4, x3, ..., -x_d, x_d-1
        y = torch.rand(x.shape)
        y[..., 0::2] = -x[..., 1::2]
        y[..., 1::2] = x[..., 0::2]
        res = x * cos + y * sin
        return res


def my_softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    """The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution."""
    # return torch.softmax(in_features, dim)
    in_max = torch.max(in_features, dim=dim, keepdim=True)
    in_exp = torch.exp(in_features - in_max.values)
    in_sum = torch.sum(in_exp, dim=dim, keepdim=True)
    return in_exp / in_sum


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    pre_softmax = ( einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") ) / \
                torch.sqrt(torch.tensor(data=Q.shape[-1]))
    #masked_pre_softmax = pre_softmax + torch.where( mask == 0, torch.tensor(float("-inf")), torch.tensor(0) )
    if(mask!= None): masked_pre_softmax = pre_softmax.masked_fill(mask == 0, float('-inf'))
    else: masked_pre_softmax = pre_softmax
    return einsum(my_softmax(masked_pre_softmax, dim = -1), V, "... queries keys, ... keys d_v -> ... queries d_v")


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[torch.Tensor, " d_k d_in"],
    k_proj_weight: Float[torch.Tensor, " d_k d_in"],
    v_proj_weight: Float[torch.Tensor, " d_v d_in"],
    o_proj_weight: Float[torch.Tensor, " d_model d_v"],
    in_features: Float[torch.Tensor, " ... sequence_length d_in"],
    max_seq_len: int | None = None,
    theta: float | None = None,
    token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
) -> Float[torch.Tensor, " ... sequence_length d_out"]:        
    qw = einsum(in_features, q_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
    kw = einsum(in_features, k_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
    vw = einsum(in_features, v_proj_weight, "... sequence_length d_in, d_v d_in -> ... sequence_length d_v")

    qs = rearrange(qw, "... seq_len (h d_head) -> ... h seq_len d_head", h=num_heads)
    ks = rearrange(kw, "... seq_len (h d_head) -> ... h seq_len d_head", h=num_heads)
    vs = rearrange(vw, "... seq_len (h d_head) -> ... h seq_len d_head", h=num_heads)

    if(theta != None):
        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    if(token_positions != None):
        qs = rope.forward(qs, token_positions)
        ks = rope.forward(ks, token_positions)

    casual_mask = torch.triu(torch.ones(in_features.shape[-2], in_features.shape[-2]), diagonal=1).bool()
    casual_mask = casual_mask[None, None, :, :]
    o = scaled_dot_product_attention(qs, ks, vs, ~casual_mask)
    o = rearrange(o, "... h seq_len d_head ->  ... seq_len (h d_head)", h=num_heads )
    return einsum(o, o_proj_weight, "... seq_len d_v, d_model d_v -> ... seq_len d_model" )


class TransformerBlock(nn.Module):
    """
    Two sub-layer: MLA + FFN
    """
    def __init__(self,  
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.weights = weights
        self.rms1 = RMSNorm(self.d_model)
        self.rms1.set_gain(self.weights["ln1.weight"])
        self.rms2 = RMSNorm(self.d_model)
        self.rms2.set_gain(self.weights["ln2.weight"])
        self.ffn = FFN(self.d_model, self.d_ff, self.weights["ffn.w1.weight"], self.weights["ffn.w2.weight"], self.weights["ffn.w3.weight"])

    def forward(self, in_features: Float[torch.Tensor, " batch sequence_length d_model"]) \
        -> Float[torch.Tensor, " batch sequence_length d_model"]:
        
        data = in_features + multihead_self_attention(self.d_model, self.num_heads,
            self.weights['attn.q_proj.weight'], self.weights['attn.k_proj.weight'], self.weights['attn.v_proj.weight'],
            self.weights['attn.output_proj.weight'], 
            self.rms1.forward(in_features), 
            self.max_seq_len, self.theta, torch.arange(0, in_features.size(-2)) )
        
        data = data + self.ffn.SwiGLU(self.rms2.forward(data))
        return data
    

class TransformerLM(nn.Module):
    """The Full Transformer LM"""
    def __init__(self,     
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = [ {
            "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
            "ln1.weight": weights[f"layers.{i}.ln1.weight"],
            "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
            "ln2.weight": weights[f"layers.{i}.ln2.weight"],
            } for i in range(num_layers)
        ]
        # word embedding
        self.word_embedding = Embedding(self.vocab_size, self.d_model)
        self.word_embedding.set_weight(weights["token_embeddings.weight"])
        # num_layers Transformer Blocks
        self.tfblocks = nn.ModuleList(
            [ TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, w) for w in self.weights ]
        )
        # Norm
        self.ln_final = RMSNorm(self.d_model)
        self.ln_final.set_gain(weights["ln_final.weight"])
        # Linear (Output Embedding)
        self.lm_head = Linear(in_features=self.d_model, out_features=self.vocab_size, weight=weights["lm_head.weight"])
    

    def forward(self, in_indices: Int[torch.Tensor, " batch_size sequence_length"],):
        vector = self.word_embedding.forward(in_indices) # word embedding
        for transformer in self.tfblocks:
            vector = transformer.forward(vector)
        vector = self.ln_final.forward(vector)
        vector = self.lm_head.forward(vector)
        #return my_softmax(vector, dim=-1)
        return vector
    