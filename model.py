import torch
from torch import nn
from torch.nn import functional as F
# from torch.autograd import Function
import numpy as np
import random

# import random
from dataloader import dataloader
from data_import import MultiResolutionDataset
from torchvision import transforms
# import torchvision.transforms.functional as visF
from math import sqrt

out_path = "C:/Users/Bene/PycharmProjects/StyleGAN/lmdb_corgis/"
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # data loader needs tensors, arrays etc.
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ]
)

dataset = MultiResolutionDataset(out_path, transform=transform, resolution=128)

loader = dataloader(dataset, 1, 128)
x = next(loader)
print(x.shape)


# 4 dimensional tensor: batch, channels, x, y

# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L22
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
        f = np.array([1, 2, 1], dtype=np.float32)
        f = np.outer(f, f)
        f /= np.sum(f)
        f = f[np.newaxis, np.newaxis, :, :]
        f = torch.from_numpy(f.copy())
        f_flip = torch.flip(f, [2, 3])
        # f and f_flip are not learnable parameters
        # https://pytorch.org/docs/1.1.0/nn.html#torch.nn.Module.register_buffer
        # https://github.com/rosinality/style-based-gan-pytorch/blob/b01ffcdcbca6d8bcbc5eb402c5d8180f4921aae4/model.py#L174
        self.register_buffer("weight", f.repeat(channel, 1, 1, 1))
        self.register_buffer("weight_flip", f_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])
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
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        w = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        b = torch.zeros(out_channel)
        # get weight https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L135
        fan_in = kernel_size * kernel_size * in_channel
        he_std = np.sqrt(2) / np.sqrt(fan_in)  # He initialization
        self.multiplier = he_std
        # weight and bias are learnable parameters
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)
        self.pad = padding

    def forward(self, input):
        # pad last 2 dimensions with 0 on each side => turn 1x1x3x3 to 1x1x5x5
        w = F.pad(self.w * self.multiplier, [1, 1, 1, 1])
        # compare to https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L188
        # add weights element wise
        w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
        # original implementation performs "deconvolution" http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf
        out = F.conv_transpose2d(input, w, self.b, stride=2, padding=self.pad)
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
        w = F.pad(self.w * self.multiplier, [1, 1, 1, 1])
        w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
        # do a regular convolution with stride 2 to downsample
        out = F.conv2d(input, w, self.b, stride=2, padding=self.pad)
        return out


# lets test if we return the correct shape
# downsamp = FusedDownsample(3,3,3,1)
# # we need to pad with 1 to obtain correct shape, else we downsample to 127x127
# print(test.shape)
# test2 = downsamp(test)
# print(test2.shape)
# display_tensor(test2)
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


# Wrappers for linear & conv layers with equal_lr applied to them
# https://github.com/rosinality/style-based-gan-pytorch/blob/b01ffcdcbca6d8bcbc5eb402c5d8180f4921aae4/model.py#L182

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


# leaky_relu https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L223
# has already been implemented for us in torch.nn.functional

# pixelwise feature vector normalization
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L239
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # reciprocal square root
        out = input * torch.rsqrt(torch.mean(torch.square(input), dim=1,
                                             keepdim=True) + 1e-8)
        return out


# test_pxnorm = torch.randn(3,3)
# pxnorm = PixelNorm()
# res = pxnorm1(test_pxnorm)
# print(res, '\n' , test_pxnorm)
# # seems good

# instance_norm
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L247
# already implemented in nn.InstanceNorm2d
# https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html

# Adaptive Instance Normalization
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        # normalize inputs
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        # style inputs are just two vectors => More efficient to store it in one
        # bias vector that will be split in two and updated in chunks
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        # convert to linear vector and split in two parts
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        # get instance normalized input
        out = self.norm(input)
        # multiply by gain and add style bias beta
        # as in 5. of https://arxiv.org/pdf/1703.06868.pdf
        out = gamma * out + beta

        return out


# Style Modulation is performed at the end of every layer
# the original implementation applies the style vector bias to the respective feature maps
# of the input. This is equivalent with the lrequalized AdaIn defined above.

# apply_noise
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L270
class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # batch, channel, height, width
        # here they add the option to randomize noise inputs,
        # we want noise parameters to be learnable though
        # "Single-Channel images consisting of uncorrelated Gaussian noise
        # [...] broadcasted to all feature maps using learned per feature scaling
        # factors, [...] added to the output of the corresponding conv."
        # End of 2. in https://arxiv.org/pdf/1812.04948.pdf
        # here the actual noise generation is decoupled from the noise application
        # weight parameter to scale noise, more flexible than having to obtain
        # shapes of input like in original implementation
        self.w = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, input, noise):
        assert len(input.shape) == 4
        return input + self.w + noise


# test = ApplyNoise(3)
# out = test(x, noise = 0)
# print(torch.equal(x, out))
# print(out.shape)
# seems to work as expected => Noise input shape depends on the stage of training
# so we need to generate noise dependent on the training progress.

# the generator starts from a learnable constant input
# the original implementation initializes the learnable constant as such :
# def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
# where fmap_base = 8192, fmap_decay = 0.9, and fmap_max = 512
# def nf(stage): return min(int(8192 / (2.0 ** (stage * 0.9))), 512)
# nf(1) # 512
# thus the constant layer is shaped like
# batch, 512, 4,4
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L507

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        # batch, 512, 4, 4
        return self.w.repeat(input.shape[0], 1, 1, 1)


# test = ConstantInput(nf(1))
# out = test(x)
# print(out.shape) # torch.Size([1, 512, 4, 4])
# works as intended

# Building Blocks:
# https://github.com/NVlabs/stylegan/blob/66813a32aac5045fcde72751522a0c0ba963f6f2/training/networks_stylegan.py#L602
def block(x, res):
    pass


# the original function splits the problem in 2 cases: 4x4 and everything else
# if 8x8 an above:
# x = activation(apply_bias(conv2d(input, nf(res-1), 3, gain, w_scale)
# with leaky relu as activation function and lr-equalized conv2d as defined above

# conv block used in the discriminator
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding,
                 kernel_size2=None, padding2=None, downsample=False,
                 fused=False):
        # fused is always true
        super().__init__()
        pad1, pad2 = padding, padding
        if padding2 is not None:
            pad2 = padding2
        # kernel sizes
        kernel1, kernel2 = kernel_size, kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            # leaky_relu is defined with an alpha of 0.2 in the original implementation
            # https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/training/networks_stylegan.py#L223
            nn.LeakyReLU(0.2),
        )

        # https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/training/networks_stylegan.py#L196
        # https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/training/networks_stylegan.py#L177
        # In the Generator:
        # if fused_scale is auto, we perform the fused operation if the resolution
        # is larger than 64, else we do the operation separately
        # In the Discriminator we perform fused downsampling if the resolution
        # is larger than, or equal to 64
        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2,
                                    padding=pad2),
                    nn.LeakyReLU(0.2)
                )
            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2,
                                padding=pad2),
                    # the original implementation uses average pooling here
                    # https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/training/networks_progan.py#L107
                    # with a kernel of [1, 1, 2, 2]
                    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
                    # nn.AvgPool2d only requires `2` as input because the images are squares
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2)
                )
        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        # input => upsample => conv3x3
        out = self.conv1(input)
        out = self.conv2(out)
        return out


# print(x.shape)
# convtest = ConvBlock(3, 3, 3, 1, downsample=False, fused=True)
# out = convtest(x)
# print(out.shape)
#
# convtest = ConvBlock(3,3,3,1, downsample=True, fused=True)
# out = convtest(x)
# print(out.shape) # torch.Size([1, 3, 64, 64])
#
# convtest = ConvBlock(3,3,3,1, downsample=True, fused=False)
# out2 = convtest(x)
# print(out.shape) # torch.Size([1, 3, 64, 64])
#
# torch.allclose(out, out2) # false
# display_tensor(out)
# display_tensor(out2)
# display_tensor(x)

# the main component of the generator
# https://github.com/rosinality/style-based-gan-pytorch/blob/b01ffcdcbca6d8bcbc5eb402c5d8180f4921aae4/model.py#L310
class StyledConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,
                 style_dim=512, initial=False, upsample=False, fused=False):
        super().__init__()

        if initial:
            # https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/training/networks_stylegan.py#L504
            self.conv1 = ConstantInput(in_channel)
        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )
                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size,
                            padding=padding
                        ),
                        Blur(out_channel),
                    )
            else:
                self.conf1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        # refer to Figure 1: https://arxiv.org/pdf/1812.04948.pdf
        # upsample & conv
        out = self.conv1(input)
        # add noise (we generate noise separately)
        out = self.noise1(out, noise)
        # pass through activation function `act` in original implementation
        # https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/training/networks_stylegan.py#L602
        out = self.lrelu1(out)
        # add style through AdaIN
        out = self.adain1(out, style)
        # regular conv
        out = self.conv2(out)
        # add noise
        out = self.noise2(out)
        # pass through activation function again
        out = self.lrelu2(out)
        # add style again
        out = self.adain2(out, style)
        return out

# to convert layers to RGB all we need is a EqualConv2d block with shape
# (n_input, 3, 1) as we want to convert n_input channels to 3 color channels
# by using 1x1 convolutions.

class Generator(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.progression = nn.ModuleList(
            # 512x4x4 constant input
            # as we keep on upsampling we decrease number of channels
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8 fused = false
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),
                StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused),
                StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),
                StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),

            ]
        )

        self.to_rgb = nn.ModuleList(
            # to rgb => convolution with n input channels and 3 output channels
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = sorted(
                random.sample(list(range(step)), len(style) - 1))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[
                    crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

            out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2,
                                             mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(
            self,
            input,
            noise=None,
            step=0,
            alpha=-1,
            mean_style=None,
            style_weight=0,
            mixing_range=(-1, -1),
    ):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(
                    torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(
                    mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha,
                              mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1),
                                     nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out