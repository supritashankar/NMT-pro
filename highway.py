#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Highway(nn.Module):
    """
    Highway network model implementation.
    """
    def __init__(self, word_embedding_size):
        """
            params:
                embedding_size: Size of the embedding. e_word in the document
            self.inter_x_projection is a linear layer and has dimensions (e_word, e_word) with bias of dim (e_word)
            self.inter_x_gate is a linear layer and has dimensions (e_word, e_word) with bias of dim (e_word)
        """
        super(Highway, self).__init__()
        self.x_projection = None
        self.x_gate = None
        self.x_highway = None
        self.x_word_emb = None
        self.x_projection_relu = None
        self.x_gate_sigmoid = None
        self.dropout_prob = 0.3
        self.inter_x_projection = nn.Linear(word_embedding_size, word_embedding_size, bias=True)
        self.inter_x_gate = nn.Linear(word_embedding_size, word_embedding_size, bias=True)
        self.relu_fn = nn.ReLU()
        self.sigmoid_fn = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, input):
        """
            params:
                x_conv_out - of dimension e_word
            return:
                x_word_emb of dimension e_word

        """
        self.x_projection = self.inter_x_projection(input)
        self.x_projection_relu = self.relu_fn(self.x_projection)
        self.x_gate = self.inter_x_gate(input)
        self.x_gate_sigmoid = self.sigmoid_fn(self.x_gate)
        self.x_highway = (self.x_gate_sigmoid * self.x_projection_relu) + (1 - self.x_gate_sigmoid) * input
        self.x_word_emb = self.dropout(self.x_highway)
        return self.x_word_emb
    
### END YOUR CODE 


"""
Unit tests for Highway network
"""

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def test_highway_network():
    EMBED_SIZE = 2
    h = Highway(word_embedding_size=EMBED_SIZE)
    INPUT_DIM = 3
    x_conv_out = torch.randn((INPUT_DIM, EMBED_SIZE))
    out = h.forward(x_conv_out)
    ## Check input and output dimensions
    assert x_conv_out.size() == torch.Size([INPUT_DIM, EMBED_SIZE])
    assert out.size() == torch.Size([INPUT_DIM, EMBED_SIZE])

    ## Check dimensions of intermediate outputs
    assert h.x_projection.size() == torch.Size([INPUT_DIM, EMBED_SIZE])
    assert h.x_gate.size() == torch.Size([INPUT_DIM, EMBED_SIZE])
    assert h.x_highway.size() == torch.Size([INPUT_DIM, EMBED_SIZE])
    assert h.x_word_emb.size() == torch.Size([INPUT_DIM, EMBED_SIZE])

    ## Compute the values using numpy to crosscheck again the returned values.
    W_proj = h.inter_x_projection.weight.detach().numpy() # Convert inter_x_projection.weight to numpy
    B_proj = h.inter_x_projection.bias.detach().numpy() # Convert inter_x_projection.bias to numpy
    W_gate = h.inter_x_gate.weight.detach().numpy() # Convert inter_x_gate.weight to numpy
    B_gate = h.inter_x_gate.bias.detach().numpy() # Convert inter_x_gate.bias to numpy
    y_proj = np.array([])
    y_gate = np.array([])
    y_ref = np.array([])

    # for each input dimension compute:
    #  y_proj (to be compared with x_projection)
    #  y_gate - to be compared with x_gate
    #  y_ref to be computed with x_highway
    for i in range(INPUT_DIM):
        highway_input = x_conv_out.numpy()[i]
        y_proj_ = np.maximum(np.dot(W_proj, highway_input)+ B_proj,0)
        y_gate_ = sigmoid(np.dot(W_gate, highway_input)+B_gate)
        y_proj = np.append(y_proj, y_proj_)
        y_gate = np.append(y_gate, y_gate_)
        y_ref = np.append(y_ref, np.multiply(y_gate_, y_proj_) + (1 - y_gate_) * highway_input)

    x_projection_relu = h.x_projection_relu.detach().numpy()
    x_gate_sigmoid = h.x_gate_sigmoid.detach().numpy()
    x_highway = h.x_highway.detach().numpy()
    assert np.all(np.array_equal(y_proj, x_projection_relu.astype(np.float32).flatten()))
    assert np.all(np.array_equal(y_gate, x_gate_sigmoid.astype(np.float32).flatten()))
    assert np.all(np.array_equal(y_ref, x_highway.astype(np.float32).flatten()))
    print ("All tests passed!")

# test_highway_network()

