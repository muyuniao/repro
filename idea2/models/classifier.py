import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes=5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.head(x)
