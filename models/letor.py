import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask

from typing import Optional, Dict, Any
from .metrics import MeanAveragePrecision, AQWV


class DocAverager(nn.Module):
    def __init__(self) -> None:
        super(DocAverager, self).__init__()

    def forward(self,
                tensor: torch.Tensor,
                mask: torch.LongTensor):
        summed = tensor.sum(2)
        return summed / mask.sum(2).float().unsqueeze(2).repeat(1, 1, summed.shape[2])


class QueryAverager(nn.Module):
    def __init__(self) -> None:
        super(QueryAverager, self).__init__()

    def forward(self,
                tensor: torch.Tensor,
                mask: torch.LongTensor):
        summed = tensor.sum(1)
        return summed / mask.sum(1).float().unsqueeze(1).repeat(1, summed.shape[1])


@Model.register('letor_training')
class LeToRWrapper(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 query_field_embedder: TextFieldEmbedder,
                 doc_field_embedder: TextFieldEmbedder,
                 embedding_transformer: FeedForward,
                 scorer: FeedForward,
                 doc_encoder: nn.Module = DocAverager(),
                 query_encoder: nn.Module = QueryAverager(),
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(LeToRWrapper, self).__init__(vocab, regularizer)

        self.query_field_embedder = query_field_embedder
        self.doc_field_embedder = doc_field_embedder
        self.embedding_transformer = embedding_transformer
        self.scorer = scorer
        self.doc_encoder = doc_encoder
        self.query_encoder = query_encoder
        self.initializer = initializer
        self.regularizer = regularizer

        self.metrics = {
            'accuracy': CategoricalAccuracy(),
            'aqwv': AQWV(cutoff=1),
            'map': MeanAveragePrecision()
        }

        #self.loss = nn.MarginRankingLoss(margin=0.75)
        self.loss = nn.CrossEntropyLoss()
        initializer(self)

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
                for metric_name, metric in self.metrics.items() }

    def forward(self,
                query: Dict[str, torch.LongTensor],
                docs: Dict[str, torch.LongTensor],
                labels: Optional[Dict[str, torch.LongTensor]] = None,
                scores: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:

        _, num_docs, _ = docs['tokens'].shape

        # (batch_size, num_docs, doc_length)
        ds_mask = get_text_field_mask(docs, num_wrapping_dims=1)
        # (batch_size, num_docs, doc_length, embedding_dim)
        ds_embedded = self.doc_field_embedder(docs)
        # (batch_size, num_docs, doc_length, transform_dim)
        ds_transformed = self.embedding_transformer(ds_embedded)
        # (batch_size, num_docs, transform_dim)
        ds_encoded = self.doc_encoder(ds_transformed, ds_mask)

        # (batch_size, query_length)
        qs_mask = get_text_field_mask(query)
        # (batch_size, query_length, embedding_dim)
        qs_embedded = self.query_field_embedder(query)
        # (batch_size, query_length, transform_dim)
        qs_transformed = self.embedding_transformer(qs_embedded)
        # (batch_size, transform_dim)
        qs_encoded = self.query_encoder(qs_transformed, qs_mask)
        # (batch_size, num_docs, transform_dim)
        qs_encoded = qs_encoded.unsqueeze(1).repeat(1, num_docs, 1)

        # (batch_size, num_docs, transform_dim * 2)
        qd = torch.cat([qs_encoded, ds_encoded], dim=2)

        if scores is not None:
            # (batch_size, num_docs, transform_dim * 2 + 1)
            qd = torch.cat([qd, scores], dim=2)

        # (batch_size, num_docs)
        logits = self.scorer(qd).squeeze(2)

        output_dict = {'logits': logits}

        if labels is not None:
            #loss = self.loss(scores[:, 0], scores[:, 1], labels.squeeze(1).float()*-2.+1.)
            loss = self.loss(logits, labels.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, labels.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict
