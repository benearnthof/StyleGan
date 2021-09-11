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
        b = torch.zeros(out_channel)
        # get weight https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L135
        fan_in = kernel_size * kernel_size * in_channel
        he_std = np.sqrt(2) / np.sqrt(fan_in) # He initialization
        self.multiplier = he_std
        # weight and bias are learnable parameters
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)
        self.pad = padding

    def forward(self, input):
        # pad last 2 dimensions with 0 on each side => turn 1x1x3x3 to 1x1x5x5
        w = F.pad(self.w * self.multiplier, [1,1,1,1])
        # compare to https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L188
        # add weights element wise
        w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
        # original implementation performs "deconvolution" http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf
        out = F.conv_transpose2d(input, w, self.b, stride = 2, padding = self.pad)
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
class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        w = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        b = torch.zeros(out_channel)
        # get weight https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L135
        fan_in = kernel_size * kernel_size * in_channel
        he_std = np.sqrt(2) / np.sqrt(fan_in)  # He initialization
        self.multiplier = he_std
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)
        self.pad = padding

    def forward(self, input):
        w = F.pad(self.w * self.multiplier, [1,1,1,1])
        w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
        # do a regular convolution with stride 2 to downsample
        out = F.conv2d(input, w, self.b, stride=2, padding=self.pad)
        return out

# lets test if we return the correct shape
downsamp = FusedDownsample(3,3,3,1)
# we need to pad with 1 to obtain correct shape, else we downsample to 127x127
print(test.shape)
test2 = downsamp(test)
print(test2.shape)
display_tensor(test2)
# also seems to work, for proper testing we should probably initialize weights as 1

# the apply_bias function in the original implementation is used for style modulation,
# the mapping network (8 layer mlp)
# at the end of each layer (layer_epilogue)
# for the torgb and fromrgb layer to obtain rgb from single channel images
# and for all building blocks of the growing network.
# this is combined with the application of lrmul in the get_weight function

# this has been used already in the original progressive growing GAN paper:
# https://arxiv.org/abs/1710.10196
# equalized learning rate
# we set w_i_hat = w_i/c where w_i are weights and c is a per layer normalization constant
# from He's initializer (He et al. 2015)
# Optimizers such as Adam normalize the gradient update by its estimated standard
# deviation => Update is independent of the scale of the parameters

# LR equalization ensures that dynamic range and thus learning speed is the same
# for all weights.
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L135

# to solve this in pytorch we again use rosinalitys implementation
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        # we again use he initialization
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        # scale the obtained weights
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        # instance of the EqualLR class with the __call__ method, also important
        # for forward pass
        fn = EqualLR(name)
        #  get the original values
        weight = getattr(module, name)
        # delete the original values
        del module._parameters[name]
        # replace them with a renamed copy
        # to forward propagate we need a `name` attribute, this happens in the
        # init of EqualLR. To run it before the forward propagation we need to
        # registar a forward pre hook:
        # "The hook will be called every time before the forward() is invoked.
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

# leaky_relu https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L223
# has already been implemented for us in torch.nn.functional

# pixelwise feature vector normalization
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L239
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # reciprocal square root
        out = input * torch.rsqrt(torch.mean(torch.square(input), dim = 1,
                                             keepdim = True) + 1e-8)
        return out

# test_pxnorm = torch.randn(3,3)
# pxnorm = PixelNorm()
# res = pxnorm1(test_pxnorm)
# print(res, '\n' , test_pxnorm)
# # seems good

# instance_norm
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L247
