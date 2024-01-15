from torch import nn
import torch

class FaceDetector(nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()
        self.resize_shape = (64, 64)
        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        

    def forward(self, x):
        # # resize x resize_shape
        x = nn.functional.interpolate(x, size=self.resize_shape)
        # print(f"x shape after resize: {x.shape}")
        # Convolutional layers
        x = self.cnn(x)
        # print(f"x shape after cnn: {x.shape}")
        # flatten
        x = x.view(x.size(0), -1)
        # print(f"x shape after flatten: {x.shape}")
        # Fully connected layers
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    # dummy data
    x = torch.rand((1, 3, 30, 50))
    model = FaceDetector()
    y = model(x)
    print(y.shape)

