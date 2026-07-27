import torch
from torch import nn

def getDevice():
    if torch.backends.mps.is_available():
        print("MPS device is available.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA device is available.")
        device = torch.device("cuda")
    else:
        print("No GPU acceleration available.")
        device = torch.device("cpu")

    return device

class FeedForward_NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, dropout_rate, depth=1):
        super().__init__()

        def block(in_features, out_features):
            return [
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]

        layers = block(input_size, hidden_size)
        for _ in range(depth - 1):
            layers += block(hidden_size, hidden_size)

        self.model = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.output(self.model(x))

