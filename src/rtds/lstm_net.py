"""The LSTM network and its censored loss (discrete-time survival on the fixed grid).

One hazard per interval ending at a grid point; S at grid point k is the product of
(1 - hazard) over intervals 1..k.
"""

import numpy as np
import torch
from torch import nn


class LSTMHazard(nn.Module):
    def __init__(self, hidden_size, n_intervals):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_intervals)

    def forward(self, seq):
        """seq: (patients, 3, 1). Returns hazard logits, (patients, n_intervals)."""
        out, _ = self.lstm(seq)
        return self.head(out[:, -1, :])  # the state after reading the last visit


def interval_index(time, grid):
    """Interval of each time: k if g_(k-1) < t <= g_k; len(grid) if after the last grid point."""
    return np.searchsorted(grid, time, side="left")


def neg_log_likelihood(logits, k, event):
    """Censored likelihood, averaged over patients.

    Event in interval k: hazard_k * product over j < k of (1 - hazard_j)
    Censored in interval k, or observed after the grid: the product only.
    """
    n_int = logits.shape[1]
    log_h = nn.functional.logsigmoid(logits)
    log_1mh = nn.functional.logsigmoid(-logits)
    j = torch.arange(n_int).unsqueeze(0)
    survived = (j < k.unsqueeze(1)).float()
    ll = (survived * log_1mh).sum(dim=1)
    had_event = event & (k < n_int)
    ll = ll + had_event.float() * log_h.gather(1, k.clamp(max=n_int - 1).unsqueeze(1)).squeeze(1)
    return -ll.mean()
