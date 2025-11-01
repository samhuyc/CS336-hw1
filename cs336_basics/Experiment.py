import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from typing import Optional, Callable

import math
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import Module



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