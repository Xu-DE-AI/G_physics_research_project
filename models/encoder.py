import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.getenv("GEOSEARCH_BASE_MODEL", "distilbert-base-uncased")


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class TextEncoder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, projection_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden, projection_dim)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = mean_pool(out, attention_mask)
        x = self.projection(x)
        return nn.functional.normalize(x, p=2, dim=-1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
