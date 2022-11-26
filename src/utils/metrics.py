from typing import List

import torch
from torchmetrics import Metric


class TopKAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute(self):
        return self.correct.float() / self.total

    def forward(self, query: List, gallery: torch.Tensor):
        for i in range(len(query)):
            if query[i] in gallery[i, :]:
                self.correct += 1
            self.total += 1

    def update(self, query: List, gallery: torch.Tensor):
        for i in range(len(query)):
            if query[i] in gallery[i, :]:
                self.correct += 1
            self.total += 1


class MeanReciprocalRank(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "correct",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute(self):
        return self.correct.float() / self.total

    def forward(self, query: List, gallery: torch.Tensor):
        for i in range(len(query)):
            rank = 0.0
            for j in range(gallery.shape[1]):
                if gallery[i, j] == query[i]:
                    rank = 1 / (j + 1)
                    break
            self.correct += rank
            self.total += 1

    def update(self, query: List, gallery: torch.Tensor):
        for i in range(len(query)):
            rank = 0.0
            for j in range(gallery.shape[1]):
                if gallery[i, j] == query[i]:
                    rank = 1 / (j + 1)
                    break
            self.correct += rank
            self.total += 1
