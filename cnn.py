#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    """
    CNN Network
    """
    def __init__(self, output_features, char_embeddings=50, kernel_size=5):
        super(CNN, self).__init__()
        m_word = 21
        self.after_cnn = None
        self.after_relu = None
        self.cnn = nn.Conv1d(in_channels=char_embeddings, out_channels=output_features, kernel_size=kernel_size, bias=True)
        self.maxpool = nn.MaxPool1d(m_word - kernel_size+1)
        self.relu = nn.ReLU()

    def forward(self, input):
        """
        :param input: - x_reshaped of dim (e_char, m_word)
        :return: x_conv_out of dim (e_word)
        """
        self.after_cnn = self.cnn(input)
        self.after_relu = self.relu(self.after_cnn)
        return self.maxpool(self.after_relu)

### END YOUR CODE

def test_cnn_network():
    EMBED_SIZE = 2
    INPUT_DIM = 3
    m_word = 21
    char_embedding = 50
    output_features = 3
    kernel_size = 5
    BATCH_SIZE = 1
    cnn = CNN(char_embeddings=50, output_features=output_features, kernel_size=5)
    ## Check input and output dims
    input = torch.randn(BATCH_SIZE, char_embedding, m_word)
    output = cnn.forward(input)
    assert input.size() == torch.Size([BATCH_SIZE, char_embedding, m_word])
    assert output.size() == torch.Size([BATCH_SIZE, output_features, 1])

    ## Check intermediate dims
    assert cnn.after_cnn.size() == torch.Size([BATCH_SIZE, output_features, m_word-kernel_size+1])
    assert cnn.after_relu.size() == torch.Size([BATCH_SIZE, output_features, m_word-kernel_size+1])

    ## Compute the values using numpy (using np.convolve) to cross check.
    """
        1. Retrieve the kernel weights, bias and inputs from the above cnn
        2. for out_channel in out_channels:
            result = np.zeros((1, in_length-kernel_size+1))  -> create an empty result
            for in_channel in range(in_channels):
                result += convolve(in_channel[:,in_channel], kernel[out_channel][in_channel])
            output[:,out_channel] = result+bias[out_channel]
        3. Compare `output.T` against the `after_cnn` from above
        4. Similarly for relu - we take np.maximum(output, 0)
    """
    in_channels = char_embedding
    out_channels = output_features
    in_length = m_word


    ###################
    # Retrieve weights, inputs, bias and compute Conv1d using numpy convolve
    kernel  = cnn.cnn.weight.detach().numpy()
    ip_np = input.detach().numpy()[0].T
    bias = cnn.cnn.bias.detach().numpy()
    output = np.empty((in_length-kernel_size+1, out_channels))
    # output_maxpool = np.empty((1, out_channels))
    for out_channel in range(out_channels):
        result = np.zeros((1, in_length-kernel_size+1))
        for in_channel in range(in_channels):
            result += np.convolve(ip_np[:,in_channel], np.flip(kernel[out_channel][in_channel]), mode='valid')
        output[:,out_channel] = result+bias[out_channel]
    final_output = output.T
    cnn_output = cnn.after_cnn.squeeze().detach().numpy()

    assert (np.all(np.allclose(final_output, cnn_output))) == True
    # after RELU
    output_after_relu = np.maximum(output, 0)
    assert (np.all(np.allclose(output_after_relu.T, cnn.after_relu.squeeze().detach().numpy()))) == True

# test_cnn_network
