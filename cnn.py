#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, char_embed_size, max_word_len, filters_num, kernel_size=5):
        """ 
        Init CNN which is a 1-D cnn.
        @param char_embed_size (int): char embedding size (dimensionality)
        @param max_word_len (int): maximum lenght of a word
        @param filters_num (int): number of filters which is the char_embed_size of a word
        @param kernel_size (int): the length of window of each kernel
        """

        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.max_word_len = max_word_len
        self.filters_num = filters_num
        self.kernel_size = kernel_size

        self.conv1d = nn.Conv1d(in_channels=char_embed_size,
                                out_channels=filters_num,
                                kernel_size=kernel_size)
        
        self.maxpool = nn.MaxPool1d(max_word_len - kernel_size + 1)

    def forward(self, x_reshape):
        """
        map from char embedding of a word to word embedding
        @param x_reshape (Tensor): Tensor of char embedding, shape: (batch_size, char_embed_size, max_word_len)
        @return x_conv_out (Tensor): Tensor of word embedding, shape: (batch_size, word_embed_size)
        """

        x_conv_out = self.conv1d(x_reshape)
        return self.maxpool(F.relu(x_conv_out)).squeeze()

### END YOUR CODE

