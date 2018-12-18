import torch
import numpy as np

from typing import Optional, Tuple
from overrides import overrides
from allennlp.training.metrics.metric import Metric

from sklearn.metrics import confusion_matrix


def paired_sort(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_a, indices = torch.sort(a, descending=True)
    sorted_b = b[indices]
    return sorted_a, sorted_b


@Metric.register('map')
class MeanAveragePrecision(Metric):
    def __init__(self, k: int = 1000):
        super(MeanAveragePrecision, self).__init__()

        self.k = k
        self.aps = []

    def __call__(self, outputs: torch.Tensor, targets: torch.LongTensor):
        pass

    def apk(self, actual, predicted, k=10):
        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

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
        self.miss = []
        self.false_alarm = []

    def __call__(self, outputs: torch.Tensor,
                 targets: torch.LongTensor,
                 masks: torch.LongTensor = None,
                 relevant_ignored: torch.LongTensor = None,
                 irrelevant_ignored: torch.LongTensor = None) -> None:

        if not len(outputs.shape) == len(targets.shape):
            targets = torch.unsqueeze(targets, 1)
            targets = torch.zeros_like(outputs).scatter_(1, targets, 1)

        if relevant_ignored is None:
            relevant_ignored = np.zeros(outputs.shape[0])
        else:
            relevant_ignored = relevant_ignored.cpu().numpy()

        if irrelevant_ignored is None:
            irrelevant_ignored = np.zeros(outputs.shape[0])
        else:
            irrelevant_ignored = irrelevant_ignored.cpu().numpy()

        if masks is None:
            masks = torch.ones_like(targets)

        for output, target, mask, rel_ignored, irrel_ignored in zip(outputs,
                                                                    targets,
                                                                    masks,
                                                                    relevant_ignored,
                                                                    irrelevant_ignored):
            valid_docs = torch.sum(mask)
            output, target = output[:valid_docs], target[:valid_docs]

            output, target = paired_sort(output, target)

            output = self._cutoff(output)

            confusion = self.confusion_matrix(output, target)

            if torch.sum(target) == 0:
                if self.version == 'tuning':
                    # ignore both miss and false alarm when tuning
                    pass
                elif self.version == 'program':
                    # ignore miss when calculating program target
                    false_alarm = confusion[0, 1] / float(sum(confusion[0, :]) + irrel_ignored)
                    self.false_alarm.append(false_alarm)
            else:
                miss = (confusion[1, 0] + rel_ignored) / float(sum(confusion[1, :]) + rel_ignored)
                false_alarm = confusion[0, 1] / float(sum(confusion[0, :]) + irrel_ignored)
                self.false_alarm.append(false_alarm)
                self.miss.append(miss)

    def confusion_matrix(self, prediction, target):
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()
        return confusion_matrix(target, prediction, labels=[0, 1])

    def _cutoff(self, output):
        doc_lens = output.shape[0]
        return torch.arange(doc_lens).lt(self.cutoff).type_as(output)

    def get_metric(self, reset: bool = False) -> float:
        if len(self.false_alarm) == 0:
            aqwv = 0.0
        else:
            # print('misses:', self.miss)
            # print('false_alarmes:', self.false_alarm)
            aqwv = 1 - np.mean(self.miss) - self.beta * np.mean(self.false_alarm)
        if reset:
            self.reset()
        return aqwv

    @overrides
    def reset(self) -> None:
        self.miss = []
        self.false_alarm = []
