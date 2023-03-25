import torch
import torch.nn as nn

class LazyNetLayer(nn.Module):
    """
    This implementation demonstrates how LazyNet defers the weight updates until the
    final epoch, allowing the network to embody the essence of procrastination in 
    its learning process.
    """
    def init(self, input_dim, output_dim, alpha, total_epochs):
        super(LazyNetLayer, self).init()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.alpha = alpha
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def forward(self, x):
        if self.current_epoch < self.total_epochs - 1:
            self.current_epoch += 1
            return x @ self.weight
        else:
            self.weight += self.alpha * x.grad
            return x @ self.weight
