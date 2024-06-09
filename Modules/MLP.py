import torch.nn as nn
import torch
from torch.utils.data import Dataset


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super().__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx] 