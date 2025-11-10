from typing import IO, Any, BinaryIO
import torch.nn
from jaxtyping import Float, Int

class Linear(torch.nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        sigma = (2/(in_features+out_features)) ** 0.5
        self.weights = torch.nn.Parameter(
            torch.zeros(out_features, in_features, dtype=dtype, device = device)
        )
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, 
                x: torch.Tensor, 
                ) -> torch.Tensor:
        # no bias in the linear layer
        return  x @ self.weights.T
        
        

        
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_mat = torch.nn.Parameter(
            torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.embedding_mat, mean=0, std=1.0, a=-3, b=3)

    def forward(self, token_ids:torch.Tensor)->torch.Tensor:
        """
        input:(batch_size, sequence_length) 
        output:(batch_size, sequence_length, d_model)
        """
        return self.embedding_mat[token_ids]




class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        input:(batch_size, sequence_length, d_model) 
        output:(batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = ((x**2).sum(dim=-1)/self.d_model + self.eps)**0.5
        y = (x*self.weights) / RMS.unsqueeze(-1)
        return y.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter(
            torch.zeros(d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.zeros(d_model, d_ff, device=device, dtype=dtype)
        )
        self.w3 = torch.nn.Parameter(
            torch.zeros(d_ff, d_model, device=device, dtype=dtype)
        )

    def SiLU(self, x:torch.Tensor) -> torch.Tensor: 
        return x * torch.sigmoid(x)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        input:(batch_size, sequence_length, d_model) 
        output:(batch_size, sequence_length, d_model)
        """
        return  (self.SiLU(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T

        
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_model = int(d_k)
        self.max_seq_len = max_seq_len
        RoPE = torch.zeros(max_seq_len, self.d_model, self.d_model, device=device, dtype=dtype)
        for i in range(max_seq_len):
            for k in range(self.d_model//2):
                angle = torch.Tensor([i/self.theta**(2*k/self.d_model)])
                RoPE[i][2*k][2*k] = torch.cos(angle)
                RoPE[i][2*k+1][2*k] = torch.sin(angle)
                RoPE[i][2*k][2*k+1] = -torch.sin(angle)
                RoPE[i][2*k+1][2*k+1] = torch.cos(angle)
        # self.register_buffer('RoPE_mat', RoPE)
        self.rope = RoPE

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        input:(batch_size, sequence_length, d_model) 
        output:(batch_size, sequence_length, d_model)
        """
        rope_positioned = self.rope[token_positions]          # (seq_len, d_model, d_model)
        return (rope_positioned @ x.unsqueeze(-1)).squeeze(-1)


class RotaryPositionalEmbedding_Optimized(torch.nn.Module):
    def __init__(self, base: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        # 1. compute inv_freq for half the dims
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        # 2. build a [max_seq_len, d_k//2] table of angles
        positions = torch.arange(max_seq_len, device=device).float()[:, None]
        angles = positions @ inv_freq[None, :]        # (max_seq_len, d_k//2)
        # 3. interleave sin/cos → (max_seq_len, d_k)
        sin = angles.sin().repeat_interleave(2, dim=-1)
        cos = angles.cos().repeat_interleave(2, dim=-1)
        # 4. register as buffers so they move with .to(device)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def forward(self, x: torch.Tensor, token_positions: torch.LongTensor) -> torch.Tensor:
        # lookup sin/cos → shape (..., seq_len, d_k)
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        # rotate each pair [a, b] → [–b, a]
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        x_rot  = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)
        # apply:  x * cos + rotate_half(x) * sin
        return x * cos + x_rot * sin


def softmax(x:torch.Tensor, dimension:int) -> torch.Tensor:
    x_normalized = x - torch.max(x,dim=dimension, keepdim=True).values
    sum_dimension = torch.exp(x_normalized).sum(dim=dimension, keepdim=True)
    return torch.exp(x_normalized)/sum_dimension

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, 
                                 mask:torch.Tensor):
    '''
    Q: (batch_size, ..., seq_len_n, d_k)
    K: (batch_size, ..., seq_len_m, d_k)
    V: (batch_size, ..., seq_len_m, d_v)
    mask: (batch_size, ..., seq_len_n, seq_len_m)

    returns: (batch_size, ..., d_v)
    '''
    pre_softmax_x = torch.einsum('...qd,...kd->...qk', Q, K)/Q.size(-1)**0.5
    # print(pre_softmax_x.size(), "@"*100)
    pre_softmax_x.masked_fill_(~mask, float('-inf'))
    attention = softmax(pre_softmax_x, -1) @ V
    return attention


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model:int, h:int, device=None, dtype=None) -> torch.Tensor:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = self.d_v = self.d_model//self.h
        self.Wq = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wk = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wv = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wo = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
    
    def MultiHead(self, 
                  Q:Float[torch.Tensor, " ... h*d_k"], 
                  K:Float[torch.Tensor, " ... h*d_k"], 
                  V:Float[torch.Tensor, " ... h*d_v"], 
                  mask):
        Q_i = torch.split(Q, self.d_model//self.h, dim=-1)
        K_i = torch.split(K, self.d_model//self.h, dim=-1)
        V_i = torch.split(V, self.d_model//self.h, dim=-1)
        results = [scaled_dot_product_attention(q, k, v, mask) for q, k, v in zip(Q_i, K_i, V_i)]
        return torch.cat(results, -1)

    def forward(self, in_features:Float[torch.Tensor, " ... sequence_length d_in"]
                ) -> Float[torch.Tensor, " ... sequence_length d_out"]:
        L = in_features.size(1)
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=in_features.device))
        MHA = self.MultiHead(torch.matmul(in_features, self.Wq), 
                                        torch.matmul(in_features, self.Wk), 
                                        torch.matmul(in_features, self.Wv), mask)
        return torch.matmul(MHA, self.Wo)
        
class MultiheadSelfAttentionWithRoPE(torch.nn.Module):
    def __init__(self, d_model:int, h:int, 
                 max_seq_len:int = 2048, theta:int = 10000, device=None, dtype=None) -> torch.Tensor:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = self.d_v = self.d_model//self.h
        self.rope = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len, device=device, dtype=dtype)
        self.Wq = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wk = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wv = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wo = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
    
    def MultiHead(self, 
                  Q:Float[torch.Tensor, " ... h*d_k"], 
                  K:Float[torch.Tensor, " ... h*d_k"], 
                  V:Float[torch.Tensor, " ... h*d_v"], 
                  mask, token_positions):
        Q_i = torch.split(Q, self.d_model//self.h, dim=-1)
        K_i = torch.split(K, self.d_model//self.h, dim=-1)
        V_i = torch.split(V, self.d_model//self.h, dim=-1)
        results = [scaled_dot_product_attention(self.rope.forward(q, token_positions), 
                                                self.rope.forward(k, token_positions),
                                                v, mask) for q, k, v in zip(Q_i, K_i, V_i)]
        return torch.cat(results, -1)

    def forward(self, in_features:Float[torch.Tensor, " ... sequence_length d_in"], token_positions,
                ) -> Float[torch.Tensor, " ... sequence_length d_out"]:
        L = in_features.size(1)
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=in_features.device))
        MHA = self.MultiHead(torch.matmul(in_features, self.Wq), 
                                        torch.matmul(in_features, self.Wk), 
                                        torch.matmul(in_features, self.Wv), mask, token_positions)
        return torch.matmul(MHA, self.Wo)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model:int, h:int, d_ff:int, 
                 max_seq_len:int = 2048, theta:int = 10000, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        
        self.mha = MultiheadSelfAttentionWithRoPE(d_model, h, 
                                                  max_seq_len=max_seq_len, theta=theta, 
                                                  device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)

    def forward(self, in_features):
        x = self.rms1.forward(in_features)
        pos = torch.arange(in_features.size(-2), device=in_features.device)
        pos = torch.broadcast_to(pos, in_features.size()[:-1])
        y1 = in_features + self.mha.forward(x, pos)
        x2 = self.rms2.forward(y1)
        y2 = y1 + self.swiglu.forward(x2)
        return y2

    
class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size:int, context_length:int, num_layers: int, 
                 d_model:int, h:int, d_ff:int, 
                 theta:int):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, )

        self.transformers = torch.nn.ModuleList([
            TransformerBlock(d_model=d_model,
                                              h=h, 
                                              d_ff=d_ff, 
                                              max_seq_len=context_length, 
                                              theta=theta) for _ in range(num_layers)
        ])
        self.rms_final = RMSNorm(d_model=d_model, )
        self.linear = Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, token_ids:Int[torch.Tensor, " batch_size sequence_length"]):
        tokens_embedded = self.embedding.forward(token_ids)
        input = tokens_embedded
        for layer_depth in range(self.num_layers):
            output = self.transformers[layer_depth].forward(input)
            input = output
        normalized_output = self.rms_final.forward(output)
        output_embedded = self.linear.forward(normalized_output)
        return output_embedded




def main():
    pass


if __name__ == "__main__":
    main()


