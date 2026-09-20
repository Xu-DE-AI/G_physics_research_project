import torch
import torch.nn as nn

class WaveformEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(1,16,9,padding=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16,32,9,padding=4), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32,64,7,padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.proj=nn.Linear(64,dim)
    def forward(self,x):
        h=self.net(x).squeeze(-1)
        return nn.functional.normalize(self.proj(h),p=2,dim=-1)
