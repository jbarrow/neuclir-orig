from typing import Optional
from overrides import overrides

from allennlp.training.metrics.metric import Metric

class InformationRetrievalMetric(Metric):
    def __init__(self):
        super(InformationRetrievalMetric, self).__init__()

    def __call__(self, output, targets):
        return 1.0

    def get_metric(self, reset: bool = False):
        # compute the (QxD matrix)
        return self._get_metric()

    def _get_metric(self):
        raise NotImplementedError(
            'Subclass of InformationRetrievalMetric must implement its own _get_metric')

    def reset(self):
        print('*** RESET CALLED')

@Metric.register('map')
class MeanAveragePrecision(InformationRetrievalMetric):
    def _get_metric(self):
        return 0.0

@Metric.register('mrr')
class MeanReciprocalRank(InformationRetrievalMetric):
    def __init__():
        pass

    def _get_metric(self):
        pass

@Metric.register('aqwv')
class AQWV(InformationRetrievalMetric):
    def __init__(self, cutoff: int = 40):
        self.cutoff = cutoff

    def _get_metric(self):
        pass
