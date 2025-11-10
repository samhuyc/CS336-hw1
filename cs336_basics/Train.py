
import os
from typing import IO, Any, BinaryIO, TypedDict
from collections.abc import Iterable
from jaxtyping import Float, Int
from typing import Optional, Callable
from pathlib import Path

import math
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import Module

from cs336_basics.Model import softmax
from datetime import datetime

### optimizer definition and helpers


def cross_entropy(inputs:Float[Tensor, " batch_size*context_length(number of predicted token) vocab_size"], 
                  targets: Int[Tensor, " batch_size*context_length(number of predicted token)"]) -> Float[Tensor, ""]:
    
    x_normalized = inputs - torch.max(inputs,dim=-1, keepdim=True).values
    sum_dimension = torch.exp(x_normalized).sum(dim=-1, keepdim=True)
    left = x_normalized[torch.arange(inputs.size(0)), targets]
    right = torch.log(sum_dimension)
    return torch.mean(- left + right)

def learning_rate_schedule(t, lr_max, lr_min, T_warmup, T_cooled):
    if t < T_warmup:
        return lr_max * (t/T_warmup)
    elif t <= T_cooled:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(((t-T_warmup)/(T_cooled-T_warmup))*torch.pi))
    else:
        return lr_min
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm, epsilon=1e-6):
    global_l2_norm = 0
    for p in parameters:
        if p.grad == None:
            continue

        global_l2_norm += torch.linalg.norm(p.grad) ** 2

    if global_l2_norm >= max_l2_norm ** 2:
        for p in parameters:
            if p.grad == None:
                continue
            p.grad *= max_l2_norm/(math.sqrt(global_l2_norm)+epsilon)


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
                p.data -= lr_t * (m/(v**0.5 + epsilon)) 
                p.data -= weight_decay * lr * p.data
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v




### experiment helpers

def data_load(input, batch_size:int, context_length:int, device:str) -> tuple[Tensor]:
    input_sequences =[]
    corresponding_sequences = []
    for b in range(batch_size):
        start_ind = np.random.randint(0, len(input) - context_length )

        input_sequences.append(input[start_ind:start_ind+context_length])
        corresponding_sequences.append(input[start_ind+1:start_ind+1+context_length])

    arg1 = torch.from_numpy(np.array(input_sequences))
    arg1.to(device=device)
    arg2 = torch.from_numpy(np.array(corresponding_sequences))
    arg2.to(device=device)
    return (arg1, arg2)


def my_save_checkpoint(model, optimizer, iteration, out):
    ret = {}
    ret["model"] = model.state_dict()
    ret["optimizer"] = optimizer.state_dict()
    ret['iter'] = iteration
    torch.save(ret, out)
    return

def my_load_checkpoint(src, model, optimizer):
    ret = torch.load(src) 
    model.load_state_dict(ret["model"])
    optimizer.load_state_dict(ret["optimizer"])
    return ret['iter']



### Training Loop definition

class ModelParams(TypedDict, total=False):  # total=False → keys are optional
    vocab_size:int
    context_length:int
    num_layers: int
    d_model:int
    h:int
    d_ff:int
    theta:int

class OptimParams(TypedDict, total=False):
    ## learning rate scheduling TODO
    weight_decay:float
    betas:tuple[float]
    eps:float
    lr:float

class TrainParams(TypedDict, total = False):
    total_steps:int
    batch_size:int
    device:int

def training_together(
    tokens: list[int],
    model_params:ModelParams | None = None,
    optim_params:OptimParams | None = None,
    train_params:TrainParams | None = None, 
    use_pretrained:bool = False,
    pretrained_path:Path = None,
    use_memmap:bool = True,
    use_wandb: bool = False,
    bandb_username: str = None,
    use_checkpoint: bool = False,
    checkpoint_method: str = 'relative',
    checkpoint_path:Path = None,
):
    
    # default_model as GPT2_medium
    default_model_params: ModelParams = {
        'vocab_size': 50257,
        'context_length': 1024,
        'num_layers': 24,
        'd_model': 1024, 
        'h': 12,
        'd_ff': 6400, 
        'theta': 10000,
    }

    if model_params is None: model_params = default_model_params
    else: 
        for k, v in default_model_params.items(): 
            model_params.setdefault(k, v)

    # default_optim as homemade AdamW
    default_optim_params: OptimParams = {
        'weight_decay':0.01, 
        'betas': (0.9, 0.95), 
        'eps': 1e-8, 
        'lr': 1e-3,
    }

    if optim_params is None: optim_params = default_optim_params
    else: 
        for k, v in default_optim_params.items(): 
            optim_params.setdefault(k, v)

    # default_train_params
    default_train_params: TrainParams = {
        'total_steps': 10e5,
        'batch_size': 16,
        'device': 'cpu',
    }

    if train_params is None: train_params = default_train_params
    else: 
        for k, v in default_train_params.items(): 
            train_params.setdefault(k, v)
    

    # if use_pretained: TODO
    # if use_memmap: TODO
    # if use_bandb: TODO

    if use_checkpoint:
        if checkpoint_path == None:
            default_checkpoint_dir = Path("./checkpoints")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = default_checkpoint_dir / timestamp


    ### start training loop
    from cs336_basics import Model, Train

    total_steps = train_params['total_steps']
    device = train_params['device']
    batch_size = train_params['batch_size']
    context_length = model_params['context_length']
    model = Model.TransformerLM(**model_params)
    optimizer = Train.AdamW(**optim_params)
    for step in range(total_steps):
        # checkpoint every 10% of progress
        if step == total_steps//(total_steps//10) and use_checkpoint and checkpoint_method == 'relative':
            my_save_checkpoint(model, optimizer, step, checkpoint_path)
        
        # eval out of sample datasets
        # TODO (make sure it correspond to each checkpoint)
        
        # 1. grab a new training data
        x, y = data_load(tokens, batch_size, context_length, device)
        # 2. get resulting logits to compute loss
        logits = model.forward(x)
        loss = cross_entropy(inputs = logits, targets=y) # the size of input is WRONG (need flatten), TODO
        # 3. get gradient of loss of each param, clip, then optimize
        loss.backward()
        gradient_clipping(model.parameters, )
        optimizer.step()





    
