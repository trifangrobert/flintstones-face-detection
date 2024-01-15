from torch import nn
import torch

class FaceClassifier(nn.Module):
    def __init__(self, num_classes=5, patch_shape=(64, 64)):
        super(FaceClassifier, self).__init__()
        self.num_classes = num_classes
        self.patch_shape = patch_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        shape_after_cnn = (512, patch_shape[0] // 32, patch_shape[1] // 32)

        self.fc = nn.Sequential(
            nn.Linear(shape_after_cnn[0] * shape_after_cnn[1] * shape_after_cnn[2], 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.patch_shape)
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    # dummy data
    x = torch.rand((1, 3, 30, 50))
    model = FaceClassifier()
    y = model(x)
    print(y.shape)