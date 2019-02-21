#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    """
    """
    def __init__(self, output_features, char_embeddings=50, kernel_size=5):
        super(CNN, self).__init__()
        m_word = 21
        self.cnn = nn.Conv1d(in_channels=char_embeddings, out_channels=output_features, kernel_size=kernel_size, bias=True)
        self.maxpool = nn.MaxPool1d(m_word - kernel_size+1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.cnn(input)
        x = self.relu(x)
        return self.maxpool(x)

### END YOUR CODE

