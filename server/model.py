import torch.nn as nn

class FlowerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3610, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=5)
        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.linear(X)
        return X