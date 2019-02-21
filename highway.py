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
        self.dropout_prob = 0.3
        self.inter_x_projection = nn.Linear(word_embedding_size, word_embedding_size, bias=True)
        self.inter_x_gate = nn.Linear(word_embedding_size, word_embedding_size, bias=True)
        self.relu_fn = nn.ReLU()
        self.sigmoid_fn = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout_prob)
    
    def forward(self, input):
        """
            params:
                x_conv_out - of dimension .
            return:
                x_word_emb of dimension ()
            
        """
        # print ("highway input", x_conv_out)
        x1 = self.inter_x_projection(input)
        x_projection = self.relu_fn(x1)
        x2 = self.inter_x_gate(input)
        x_gate = self.sigmoid_fn(x2)
        # print ("x_gate", self.x_gate.size())
        # print ("x projection", self.x_projection.size())
        self.x_highway = (x_gate * x_projection) + (1 - x_gate) * input
        #print ("x highway", self.x_highway.size())
        self.x_word_emb = self.dropout(self.x_highway)
        #print ("After dropout", self.x_word_emb)
        return self.x_word_emb


# def sigmoid(x, derivative=False):
#     sigm = 1. / (1. + np.exp(-x))
#     if derivative:
#         return sigm * (1. - sigm)
#     return sigm   
    
# def test_highway_network():
#     # TODO: try a bigger embed size
#     # TODO: check the dimensions of weights and other things
#     EMBED_SIZE = 2
#     BATCH_SIZE = 3
#     h = Highway(word_embedding_size=EMBED_SIZE)
    
#     x_conv_out = torch.randn((1,2))
#     out = h.forward(x_conv_out)
#     print("Y", h.x_highway)
    
#     W_proj = h.inter_x_projection.weight.detach().numpy()
#     B_proj = h.inter_x_projection.bias.detach().numpy()
#     W_gate = h.inter_x_gate.weight.detach().numpy()
#     B_gate = h.inter_x_gate.bias.detach().numpy()
#     highway_input = x_conv_out.numpy()[0]
    
 
#     Y_proj = np.maximum(np.dot(W_proj, highway_input)+ B_proj,0)
#     Y_gate = sigmoid(np.dot(W_gate, highway_input)+B_gate)

#     Y_ref = np.multiply(Y_gate, Y_proj) + (1 - Y_gate) * highway_input
#     print("Y_ref", Y_ref)

    
# test_highway_network()
    
### END YOUR CODE 

