import torch
import torch.nn as nn
import random

class GossipNetLayer(nn.Module):
    """
    This implementation highlights how GossipNet exchanges information between 
    different models, simulating the sharing and spreading of information and 
    rumors in a human-like manner.
    """
    def init(self, input_dim, output_dim):
        super(GossipNetLayer, self).init()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        return x @ self.weight + self.bias

    def gossip_exchange(self, other_model, alpha):
        with torch.no_grad():
            temp_weight = self.weight.clone()
            temp_bias = self.bias.clone()
            self.weight.mul_(alpha).add_(other_model.weight * (1 - alpha))
            self.bias.mul_(alpha).add_(other_model.bias * (1 - alpha))
            other_model.weight.mul_(alpha).add_(temp_weight * (1 - alpha))
            other_model.bias.mul_(alpha).add_(temp_bias * (1 - alpha))

    def gossip(models, gossip_probability, alpha):
        for model_a, model_b in zip(models[:-1], models[1:]):
            if random.random() < gossip_probability:
                model_a.gossip_exchange(model_b, alpha) 