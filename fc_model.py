import torch
import torch.nn as nn

class SimpleFCModel(nn.Module):
    """A simple fully connected neural network."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = SimpleFCModel(input_dim=10, hidden_dim=20, output_dim=1)
    sample_input = torch.randn(5, 10)
    output = model(sample_input)
    print(output)
