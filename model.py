import torch
from torch import nn
from torch.nn import functional as F
# from torch.autograd import Function
import numpy as np

# import random
from dataloader import dataloader
from data_import import MultiResolutionDataset
from torchvision import transforms
# import torchvision.transforms.functional as visF

out_path = "C:/Users/Bene/PycharmProjects/StyleGAN/lmdb_corgis/"
transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # data loader needs tensors, arrays etc.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

dataset = MultiResolutionDataset(out_path, transform=transform, resolution=128)

loader = dataloader(dataset, 1, 128)
x = next(loader)
print(x.shape)
# 4 dimensional tensor: batch, channels, x, y

#https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L22
# def blur2d(x, f = [1,2,1], normalize = True, flip = False, stride = 1):
#     assert x.shape.__len__() == 4 and all(dim is not None for dim in x.shape[1:])
#     assert isinstance(stride, int) and stride >= 1
#
#     # filter kernel
#     f = np.array(f, dtype = np.float32)
#     if f.ndim == 1:
#         # convert vector to 2d arrays and do the equivalent of "outer" in R
#         f = np.outer(f, f)
#     assert f.ndim == 2
#     if normalize:
#         f /= np.sum(f)
#     if flip:
#         #https://numpy.org/doc/stable/reference/generated/numpy.flip.html
#         # original implementation corresponds to flip(f)
#         f = np.flip(f)
#     # here we want batch and channel first so we add axis in front
#     f = f[np.newaxis, np.newaxis, :, :]
#     # f.shape # (1,1,3,3)
#     f = torch.from_numpy(f.copy())

# lets rewrite as a nn.Module
class Blur(nn.Module):
    def __init__(self, channel):
        # channel is the number of channels in the image => 1 for greyscale imgs
        super().__init__()
        f = np.array([1,2,1], dtype = np.float32)
        f = np.outer(f,f)
        f /= np.sum(f)
        f = f[np.newaxis, np.newaxis, :, :]
        f = torch.from_numpy(f.copy())
        f_flip = torch.flip(f, [2,3])
        # f and f_flip are not learnable parameters
        # https://pytorch.org/docs/1.1.0/nn.html#torch.nn.Module.register_buffer
        # https://github.com/rosinality/style-based-gan-pytorch/blob/b01ffcdcbca6d8bcbc5eb402c5d8180f4921aae4/model.py#L174
        self.register_buffer("weight", f.repeat(channel, 1,1,1))
        self.register_buffer("weight_flip", f_flip.repeat(channel, 1,1,1))

    def forward(self, input):
        return F.conv2d(input, self.weight, padding = 1, groups=input.shape[1])
        # consider swapping this with the changes proposed in rosinality implementation

# blur = blur2d(3)
# blurtest = blur(x)
# blurtest = blurtest.squeeze(0)
# blurtest = visF.to_pil_image(blurtest)
# blurtest.show()
# seems to work

# upscale2d_conv2d:
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L174
# both operations are combined in the original implementation to save memory and speed up performance
# https://github.com/rosinality/style-based-gan-pytorch/blob/b01ffcdcbca6d8bcbc5eb402c5d8180f4921aae4/model.py#L56

class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding = 0):
        super().__init__()
        w = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)
        # get weight https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L135
        fan_in = kernel_size * kernel_size * in_channel
        he_std = np.sqrt(2) / np.sqrt(fan_in) # He initialization
        self.multiplier = he_std
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(bias)
        self.pad = padding

    def forward(self, input):
        # pad last 2 dimensions with 0 on each side => turn 1x1x3x3 to 1x1x5x5
        w = F.pad(self.w * self.multiplier, [1,1,1,1])
        # compare to https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L188
        # add weights element wise
        w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1]+ w[:, :, :-1, :-1]) * 0.25
        # original implementation performs "deconvolution" http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf
        out = F.conv_transpose2d(input, w, self.bias, stride = 2, padding = self.pad)
        return out

# upsamp = FusedUpsample(3,3,3, padding = 1)
# print(x.shape)
# test = upsamp(x)
# print(test.shape)
# # displaying to check if the output makes sense
# from utils import display_tensor
# display_tensor(test)
# # seems good

# fused downsample works just the same way but we use a convolution with stride 2


# https://arxiv.org/abs/1710.10196
# equalized learning rate
# we set w_i_hat = w_i/c where w_i are weights and c is a per layer normalization constant
# from He's initializer (He et al. 2015)
# Optimizers such as Adam normalize the gradient update by its estimated standard
# deviation => Update is independent of the scale of the parameters

# LR equalization ensures that tynamic range and thus learning speed is the same
# for all weights.
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L135
