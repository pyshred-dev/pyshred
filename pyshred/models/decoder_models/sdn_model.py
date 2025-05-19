from .abstract_decoder import AbstractDecoder
import torch.nn as nn
import torch.nn.functional as F

class SDN(AbstractDecoder):
    """
    Shallow Decoder Network (SDN) implementation.

    A fully connected decoder that maps a low-dimensional latent space
    back to a high-dimensional state.
    """


    def __init__(self, l1 = 350, l2 = 400, dropout = 0.1):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.dropout_prob = dropout
    

    def initialize(self, input_size, output_size):
        """
        Initialize the SDNDecoder with input and output sizes.

        Parameters:
        -----------
        input_size : int
            Size of the input features.
        output_size : int
            Size of the output features.
        """
        super().initialize(input_size)
        self.linear1 = nn.Linear(input_size, self.l1)
        self.linear2 = nn.Linear(self.l1, self.l2)
        self.linear3 = nn.Linear(self.l2, output_size)
        self.dropout = nn.Dropout(self.dropout_prob)

    """
    Accepts output of sequence model
    """
    def forward(self, x):
        x = x["final_hidden_state"]
        output = self.linear1(x)
        output = self.dropout(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output = F.relu(output)
        output = self.linear3(output)
        return output
    
    @property
    def model_name(self):
        return "SDN"