import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """DQN architecture from Mnih et al. 2013."""

    def __init__(self, num_actions: int, in_channels: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        # After conv: (B, 32, 9, 9) → flatten → 2592
        self.fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 84, 84) float32 in [0, 1]
        return self.fc(self.conv(x).flatten(1))
