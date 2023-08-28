# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:34 2023

@author: jamyl
"""

import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cas import contrast_adaptive_sharpening

BINOMIAL_KERNEL = th.tensor([1, 2, 1]).float().view(1, 1, 1, 3).repeat(3, 1, 1, 1)/4


im = th.from_numpy(plt.imread("data/kodim13.png").copy())
im = im.unsqueeze(0).permute(0, -1, 1, 2)

blurry = F.conv2d(im, BINOMIAL_KERNEL, padding=(0, 1), groups=3)
blurry = F.conv2d(blurry, BINOMIAL_KERNEL.permute(0, 1, 3, 2), padding=(1, 0), groups=3)[0]

amount = 0.8

plt.imsave('data/amount={:3.2f}.png'.format(amount), contrast_adaptive_sharpening(blurry, amount).permute(1, 2, 0).numpy())
plt.imsave('data/blurry.png', blurry.permute(1, 2, 0).numpy())