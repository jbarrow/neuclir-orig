import torch

from typing import Optional
from overrides import overrides
from allennlp.training.metrics.metric import Metric


@Metric.register('map')
class MeanAveragePrecision(Metric):
    def __init__(self, k: int = 1000):
        super(MeanAveragePrecision, self).__init__()

        self.k = k

    def __call__(self, outputs: torch.Tensor, targets: torch.LongTensor):
        pass

    def get_metric(self, reset: bool = False):
        return 0.0

    def reset(self):
        pass


@Metric.register('aqwv')
class AQWV(Metric):
    def __init__(self, beta: float = 40., cutoff: int = 40):
        super(AQWV, self).__init__()

        self.beta = beta
        self.cutoff = cutoff

    def __call__(self, outputs: torch.Tensor, targets: torch.LongTensor):
        pass

    def get_metric(self, reset: bool = False):
        return 0.0

    def reset(self):
        pass


@Metric.register('mrr')
class MeanReciprocalRank(Metric):
    def __init__():
        pass

    def __call__(self, outputs: torch.Tensor, targets: torch.LongTensor):
        pass

    def get_metric(self, reset: bool = False):
        return 0.0

    def reset(self):
        pass
