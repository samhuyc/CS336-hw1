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
        self.weight = torch.nn.Parameter(torch.clamp(torch.normal(0, sigma, (out_features, in_features), dtype=dtype, device = device)
                                  , min=-3*sigma, max=3*sigma))


    def forward(self, 
                x: torch.Tensor, 
                ) -> torch.Tensor:
        # no bias
        return  x @ self.weight.T
        
        

        
    






