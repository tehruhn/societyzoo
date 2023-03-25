import torch
import torch.nn as nn
import random

class IndecisiveNetLayer(nn.Module):
    """
    This implementation highlights how IndecisiveNet adjusts its weight updates 
    based on a dynamic learning rate, frequently changing hyperparameters mid-training, 
    and reflecting the human tendencies of indecisiveness and second-guessing in the 
    learning process.
    """
    def init(self, input_dim, output_dim, alpha, change_probability):
        super(IndecisiveNetLayer, self).init()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.alpha = alpha
        self.change_probability = change_probability

    def update_learning_rate(self):
        change_trigger = random.uniform(0, 1)
        if change_trigger < self.change_probability:
            self.alpha = random.uniform(0.0001, 0.1)

    def forward(self, x):
        self.update_learning_rate()
        self.weight += self.alpha * x.grad
        return x @ self.weight
