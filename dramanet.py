import torch
import torch.nn as nn
import torch.optim as optim

class DramaNet(nn.Module):
    """
    This implementation showcases how DramaNet adjusts its training behavior dramatically 
    in response to small changes in the input data or training environment, reflecting 
    the human tendency to overreact and create drama.
    """
    def init(self, input_dim, output_dim, alpha_0, beta):
        super(DramaNet, self).init()
        self.layer = nn.Linear(input_dim, output_dim)
        self.alpha_0 = alpha_0
        self.beta = beta

    def forward(self, x):
        return self.layer(x)

    def adaptive_learning_rate(self, prev_input, current_input, avg_magnitude):
        delta_x = torch.abs(current_input - prev_input)
        return self.alpha_0 * (1 + self.beta * (delta_x / avg_magnitude))

    def train_drama(self, train_data, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.alpha_0)
        prev_input = train_data[0][0]

        for epoch in range(epochs):
            for inputs, targets in train_data:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                alpha_t = self.adaptive_learning_rate(prev_input, inputs, train_data[0][0].abs().mean())
                for param_group in optimizer.param_groups:
                    param_group['lr'] = alpha_t

                optimizer.step()
                prev_input = inputs
