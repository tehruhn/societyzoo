import torch
import torch.nn as nn
import random

class MultiTaskingNetLayer(nn.Module):
    """
    This implementation highlights how MultiTaskingNet adjusts its weight 
    updates based on a simulated browsing activity level, attempting to 
    balance productivity with the enticement of online distractions during 
    the learning process.
    """
    def init(self, input_dim, output_dim, alpha, beta):
        super(MultiTaskingNetLayer, self).init()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.alpha = alpha
        self.beta = beta
        
    def browsing_activity(self):
        return random.uniform(0, 1)

    def forward(self, x):
        distraction_level = self.browsing_activity()
        alpha_t = self.alpha * (1 - self.beta * distraction_level)
        self.weight += alpha_t * x.grad
        return x @ self.weight
