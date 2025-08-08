# CNN architecture 
import torch
import torch.nn as nn
import torchvision.models as models

class DroneNavigationCNN(nn.Module):
    def __init__(self, sequence_length=5, num_outputs=4):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        
        # Modify first conv layer for stacked frames
        original_first_layer = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            in_channels=3 * sequence_length,
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False
        )
        
        # Modify classifier head
        self.base_model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x):
        return self.base_model(x)
