import torch
import torch.nn as nn

class OverUnderNN(nn.Module):
    def __init__(self, input_size):
        super(OverUnderNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)  # Dropout with 50% chance
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)  # Dropout with 50% chance
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = torch.relu(x)
        x = self.output(x)
        return self.sigmoid(x)

