import torch.nn as nn
from pytorch_tcn.tcn import TCN

class FOGTCN(nn.Module): # FOGTCN inherits from nn.Module. This tells PyTorch that the class is a trainable model.
    def __init__(self, input_channels=6, num_classes=2):
        super().__init__()

        # TCN configuration (example)
        self.num_channels = [32, 64, 64]

        self.tcn = TCN( # This TCN block takes the 120-step sensor signal and turns it into high-level features.
            num_inputs=input_channels, # 6 IMU channels (acc+gyro)
            num_channels=self.num_channels, # 3 TCN layers, increasing depth
            kernel_size=4, # 3 time-points per convolution
            dropout=0.2, # regularization to avoid overfitting
            causal=True, # model only uses past data (real-time applicable)
            use_norm="weight_norm" # stabilizes training
        )
        final_channels = self.num_channels[-1]  # last layer size

        self.classifier = nn.Conv1d(final_channels, num_classes, kernel_size=1)

        # This transforms the final TCN output to 2 logits, one for each class:
        # index 0 → no FoG
        # index 1 → FoG

    def forward(self, x): # Why mean pooling? The TCN outputs features for each time step. 
                          # We collapse time dimension to get a single feature vector per sample, making classification easier.
        x = self.tcn(x)            # (Batch_size, channels, time) Extract temporal features
        
        x = self.classifier(x)     # Output class scores
        x = x.mean(dim=2)          # global average pooling, Average over time
                                   # (one vector per sample) (batch, channels, time) -> (batch, channels)
        return x
    


