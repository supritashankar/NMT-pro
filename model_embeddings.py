#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code
 
        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.char_embed_size = 50
        self.x_embeddings = None
        pad_token_idx = vocab['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=pad_token_idx)
        self.cnn = CNN(output_features=self.embed_size, char_embeddings=self.char_embed_size)
        self.highway = Highway(self.embed_size)
        # print ("Show self.embeddings", self.embeddings)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        # print ("model embedding forward", input.size())
        # new_input = input.reshape(input.size()[0]*input.size()[1], input.size()[2])
        # self.x_embeddings = self.embeddings(new_input)
        self.x_embeddings = self.embeddings(input)
        new_embeds = self.x_embeddings.permute(0, 1, 3, 2)
        # print ("show size", new_embeds.size())
        # print ("x_embedding", self.x_embeddings.size())
        new_embeds2 = new_embeds.reshape(new_embeds.size()[0] * new_embeds.size()[1], new_embeds.size()[2], new_embeds.size()[3])
        # new_embeds = self.x_embeddings.reshape(self.x_embeddings.size()[0]*self.x_embeddings.size()[1], self.x_embeddings.size()[3], self.x_embeddings.size()[2])
        # cnn_model = CNN(output_features=self.embed_size, char_embeddings=self.char_embed_size, m_word=input.size()[2])
        cnn_op = self.cnn.forward(new_embeds2)
        new_res = torch.squeeze(cnn_op, dim=2)
        #highway_model = Highway(self.embed_size, dropout_prob=0.3)
        highway_op = self.highway.forward(new_res)
        new_highway_op = highway_op.reshape(input.size()[0], input.size()[1], highway_op.size()[1])
        return new_highway_op

        ### END YOUR CODE

