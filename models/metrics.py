import torch
import numpy as np

from typing import Optional
from overrides import overrides
from allennlp.training.metrics.metric import Metric


@Metric.register('map')
class MeanAveragePrecision(Metric):
    def __init__(self, k: int = 1000):
        super(MeanAveragePrecision, self).__init__()

        self.k = k
        self.aps = []

    def __call__(self, outputs: torch.Tensor, targets: torch.LongTensor):
        pass

    def get_metric(self, reset: bool = False):
        if len(self.aps) == 0:
            map = 0.0
        else:
            map = np.mean(self.aps)
        if reset:
            self.reset()
        return map

    @overrides
    def reset(self):
        self.aps = []


@Metric.register('aqwv')
class AQWV(Metric):
    def __init__(self, beta: float = 40., cutoff: int = 40, version: str = 'tuning') -> None:
        super(AQWV, self).__init__()

        self.beta = beta
        self.cutoff = cutoff
        self.version = version
        self.qwvs = []

    def __call__(self, outputs: torch.Tensor, targets: torch.LongTensor) -> None:
        pass

    def get_metric(self, reset: bool = False) -> float:
        if len(self.qwvs) == 0:
            aqwv = 0.0
        else:
            aqwv = np.mean(self.qwvs)
        if reset:
            self.reset()
        return aqwv

    @overrides
    def reset(self) -> None:
        self.qwvs = []
