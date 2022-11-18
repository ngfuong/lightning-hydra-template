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
        rank = 0.0
        for i in range(len(query)):
            for j in range(len(gallery)):
                if gallery[i, j] == query[i]:
                    rank = 1 / (i + 1)
                    break
            self.correct += rank
            self.total += 1

    def update(self, query: List, gallery: torch.Tensor):
        rank = 0.0
        for i in range(len(query)):
            for j in range(len(gallery)):
                if gallery[i, j] == query[i]:
                    rank = 1 / (i + 1)
                    break
            self.correct += rank
            self.total += 1
