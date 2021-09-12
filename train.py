# Training Details highlighted in section C of
# https://arxiv.org/pdf/1812.04948.pdf
# Bilinear upsampling => Here we just use nearest neighbor upsampling
# Start from 8x8 instead of 4x4 images
# Non Saturating loss
# R1 regularization with gamma = 10
# learning rate of 0.003
# leaky relu with alpha 0.2
# equalized learning rate for all layers
# same feature map counts as in
# https://arxiv.org/pdf/1710.10196.pdf
# mapping network of 8 fully connected layers
# dimensionality of all input and output activations = 512
# decreasing the learning rate for the mapping network by two orders of magnitude
# constant input in synthesis network is initialized to one
# biases and noise scaling factor initialized to zero
#