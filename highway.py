#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn

class Highway(nn.Module):
    """
    Highway class to generate embedding vector
    """
    def __init__(self, embed_size):
        """
        Init the Highway module
        @param embed_size (int): Embedding size (dimensionality)
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.proj = nn.Linear(self.embed_size, self.embed_size)
        self.gate = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, input):
        """
        Take mini-batch of convolution output and calculate
        :param input, shape: (batch_size x embed_size)
        :return: word_embedding, shape: (batch_size x embed_size)
        """
        x_proj = torch.relu(self.proj(input))
        x_gate = torch.sigmoid(self.gate(input))

        return x_gate * x_proj + (1 - x_gate) * input


### END YOUR CODE