import torch
import torch.nn as nn
from datetime import datetime

class ProcrastiNetLayer(nn.Module):
    """
    This implementation showcases how ProcrastiNet adjusts its weight updates based on the 
    current day and time, emulating the learning patterns of a procrastinator by training 
    solely during late nights and weekends.
    """
    def init(self, input_dim, output_dim, alpha):
        super(ProcrastiNetLayer, self).init()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.alpha = alpha

    def is_late_night_or_weekend(self):
        now = datetime.now()
        return now.hour >= 22 or now.weekday() >= 5

    def forward(self, x):
        if self.is_late_night_or_weekend():
            self.weight += self.alpha * x.grad
        return x @ self.weight
