# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:30:28 2023

@author: jamyl
"""

import torch as th
import torch.nn.functional as F

EPSILON = 1e-6

def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = th.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return mn
    
def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = th.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return mx
    

def contrast_adaptive_sharpening(x, amount=0.8, better_diagonals=True):
    """
    Performs a contrast adaptive sharpening on the batch of images x.
    The algorithm is directly implemented from FidelityFX's source code, 
    that can be found here
    https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/master/ffx-cas/ffx_cas.h

    Parameters
    ----------
    x : Tensor
        Image or stack of images, of shape [burst, channels, ny, nx].
        Burst and channel dimensions can be ommited.
    amount : int [0, 1]
        Amount of sharpening to do, 0 being minimum and 1 maximum 
    better_diagonals : bool, optional
        If False, the algorithm runs slightly faster, but
        won't consider diagonals. The default is True.

    Returns
    -------
    Tensor
        Processed stack of images.

    """
    assert x.dim() >= 2
    assert 0 <= amount <= 1
    assert x.max() <= 1
    assert x.min() >= 0
    
    x_padded = F.pad(x, pad=(1, 1, 1, 1))
    # each side gets padded with 1 pixel 
    # padding = same by default
    
    # Extracting the 3x3 neighborhood around each pixel
    # a b c
    # d e f
    # g h i
    
    b = x_padded[..., :-2, 1:-1]
    d = x_padded[..., 1:-1, :-2]
    e = x_padded[..., 1:-1, 1:-1]
    f = x_padded[..., 1:-1, 2:]
    h = x_padded[..., 2:, 1:-1]

    if better_diagonals:    
        a = x_padded[..., :-2, :-2]
        c = x_padded[..., :-2, 2:]
        g = x_padded[..., 2:, :-2]
        i = x_padded[..., 2:, 2:]
    
    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)
    
    if better_diagonals:
        diag = (a, c, g, i)
        mn2 = min_(diag)
        mx2 = max_(diag)
        
        mx = mx + mx2
        mn = mn + mn2
    
    # Computing local weight
    inv_mx = th.reciprocal(mx + EPSILON) # 1/mx
    
    if better_diagonals:
        amp = inv_mx * th.minimum(mn, (2 - mx))
    else:
        amp = inv_mx * th.minimum(mn, (1 - mx))
    
    # scaling
    amp = th.sqrt(amp)
    
    w = - amp * (amount * (1/5 - 1/8) + 1/8)    
    # w scales from 0 when amp=0 to K for amp=1
    # K scales from -1/5 when amount=1 to -1/8 for amount=0
    
    # The local conv filter is
    # 0 w 0
    # w 1 w
    # 0 w 0
    div = th.reciprocal(1 + 4*w)
    output = ((b + d + f + h)*w + e) * div
    
    # Clipping between 0 and 1. It fixes previous divisions by 0 too
    return output.clamp(0, 1)
