"""The LSTM behind the SurvivalModel interface, so SurvSHAP(t) and Run 2 use it unchanged.

It takes the forest's flat columns, rearranges them into a sequence, and returns sksurv
StepFunctions on the grid.
"""

import copy

import numpy as np
import torch
from sksurv.functions import StepFunction

from rtds import seeds
from rtds.lstm_net import LSTMHazard, interval_index, neg_log_likelihood


class LSTMModel:
    def __init__(self, cfg, dataset_index, grid, replicate=None, settings=None):
        # settings: tuned values; None = the fixed settings.
        self.p = {**cfg["lstm"], **(settings or {})}
        self.grid = np.asarray(grid, dtype=float)
        self.master_seed = cfg["master_seed"]
        self.dataset_index = dataset_index
        # None: the main-run LSTM. 0, 1, ...: extra training seeds (seed check).
        self.replicate = replicate
        self.history = []  # (epoch, training loss, validation loss)

    def _sequence(self, X):
        # Flat columns, as the forest takes them, to a sequence in visit order (by name, not position).
        cols = sorted(X.columns, key=lambda c: int(c.removeprefix("visit_")))
        return torch.tensor(X[cols].to_numpy(dtype=np.float32)).unsqueeze(-1)  # (patients, 3, 1)

    def fit(self, X, y):
        # One thread keeps results reproducible; parallelism is across datasets.
        torch.set_num_threads(1)
        stream = "lstm" if self.replicate is None else "lstm_seed_check"
        torch.manual_seed(seeds.int_seed(self.master_seed, stream, self.dataset_index, self.replicate))
        gen = seeds.rng(self.master_seed, stream, self.dataset_index, self.replicate)

        seq = self._sequence(X)
        k = torch.tensor(interval_index(y["time"], self.grid), dtype=torch.long)
        event = torch.tensor(y["event"], dtype=torch.bool)
        order = gen.permutation(len(seq))
        n_val = int(round(self.p["val_fraction"] * len(seq)))
        val, train = order[:n_val], order[n_val:]

        self.net = LSTMHazard(self.p["hidden_size"], len(self.grid))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["learning_rate"], weight_decay=self.p["weight_decay"])
        best, best_state, waited = np.inf, None, 0
        for epoch in range(self.p["max_epochs"]):
            self.net.train()
            batches = gen.permutation(train)
            train_loss = []
            for start in range(0, len(batches), self.p["batch_size"]):
                b = batches[start:start + self.p["batch_size"]]
                opt.zero_grad()
                loss = neg_log_likelihood(self.net(seq[b]), k[b], event[b])
                loss.backward()
                opt.step()
                train_loss.append(loss.item())
            self.net.eval()
            with torch.no_grad():
                val_loss = neg_log_likelihood(self.net(seq[val]), k[val], event[val]).item()
            self.history.append((epoch + 1, float(np.mean(train_loss)), val_loss))
            # Early stopping: keep the weights with the lowest validation loss.
            if val_loss < best:
                best, best_state, waited = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                waited += 1
                if waited >= self.p["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_epoch = int(np.argmin([h[2] for h in self.history])) + 1
        return self

    def _survival(self, X):
        with torch.no_grad():
            h = torch.sigmoid(self.net(self._sequence(X)))
        return torch.cumprod(1 - h, dim=1).numpy().astype(float)  # S at each grid point

    def predict_survival_function(self, X):
        # Exact at the grid points, the only times the explainer evaluates.
        return np.array([StepFunction(self.grid, s) for s in self._survival(X)])

    def predict_risk(self, X):
        """Risk score for the C-index: less area under the survival curve = higher risk."""
        return -self._survival(X).sum(axis=1)

    def assert_grid_inside(self, X, grid):
        lo, hi = self.predict_survival_function(X.iloc[[0]])[0].domain
        assert lo <= grid[0] and grid[-1] <= hi, f"grid [{grid[0]}, {grid[-1]}] outside LSTM range [{lo}, {hi}]"
        return lo, hi
