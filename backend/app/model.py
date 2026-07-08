import torch
import torch.nn as nn

from .landmarks import MAX_HANDS, NUM_LANDMARKS

INPUT_SIZE = MAX_HANDS * NUM_LANDMARKS * 3  # 126: flattened per-frame landmark vector


class SignSequenceClassifier(nn.Module):
    def __init__(self, num_classes: int, input_size: int = INPUT_SIZE, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)
