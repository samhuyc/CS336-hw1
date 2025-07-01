from typing import Any
import torch.nn

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
        print('start')
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        RoPE = torch.zeros(max_seq_len, int(d_k*2), int(d_k*2), device=device, dtype=dtype)
        for i in range(max_seq_len):
            for k in range(int(d_k)):
                angle = torch.Tensor([i/theta**(2*(k-1)/(d_k*2))])
                RoPE[i][2*k][2*k] = torch.cos(angle)
                RoPE[i][2*k+1][2*k] = torch.sin(angle)
                RoPE[i][2*k][2*k+1] = -torch.sin(angle)
                RoPE[i][2*k+1][2*k+1] = torch.cos(angle)
        # self.register_buffer('RoPE_mat', RoPE)
        self.rope = RoPE
        print('init comple')

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        input:(batch_size, sequence_length, d_model) 
        output:(batch_size, sequence_length, d_model)
        """
        print('forward begin')
        rope = self.rope[token_positions]          # (seq_len, d_model, d_model)
        print(rope)
        return x



def main():
    n = RMSNorm(4, )
    x = torch.Tensor([[[1,2,3,4],[4,5,6,7],[3,2,1,0]]])
    n.forward(x)



if __name__ == "__main__":
    main()


