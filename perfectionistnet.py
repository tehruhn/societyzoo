import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class PerfectionistNet(nn.Module):
    """
    This implementation showcases how PerfectionistNet continues training until 
    the stopping criterion is met, constantly adjusting its weights and biases 
    in pursuit of the perfect model, reflecting the human trait of perfectionism.
    """
    def init(self, input_dim, output_dim, alpha, epsilon):
        super(PerfectionistNet, self).init()
        self.layer = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x):
        return self.layer(x)

    def stop_criterion(self, prev_performance, current_performance):
        return abs(current_performance - prev_performance) < self.epsilon

    def train_perfectionist(self, train_data, val_data, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.alpha)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

        prev_performance = float('inf')
        while True:
            for epoch in range(epochs):
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            current_performance = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    current_performance += (predicted == targets).sum().item()

            if self.stop_criterion(prev_performance, current_performance):
                break

            prev_performance = current_performance
