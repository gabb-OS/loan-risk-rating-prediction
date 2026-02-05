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


class FeedForward_NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, dropout_rate, depth=1):
        super(FeedForward_NN, self).__init__()

        model = [
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        ]

        block = [
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        ]

        for i in range(depth):
            model += block

        self.model = nn.Sequential(*model)

        self.output = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        h = self.model(x)
        out = self.output(h)
        return out
    

