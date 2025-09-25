
import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from typing import Optional, Callable

import math
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import Module

from cs336_basics.Model import softmax


def cross_entropy(inputs:Float[Tensor, " batch_size vocab_size"], 
                  targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    
    x_normalized = inputs - torch.max(inputs,dim=-1, keepdim=True).values
    sum_dimension = torch.exp(x_normalized).sum(dim=-1, keepdim=True)
    left = x_normalized[torch.arange(inputs.size(0)), targets]
    right = torch.log(sum_dimension)
    return torch.mean(- left + right)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay":weight_decay, "betas":betas, "eps":eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group['eps']

            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))
                t = state.get('t', 1)

                grad = p.grad.data
                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1-beta2) * (grad **2)
                lr_t = lr * math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data -= lr_t * (m/v**0.5 + epsilon) 
                p.data -= weight_decay * lr * p.data
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
                
        return loss