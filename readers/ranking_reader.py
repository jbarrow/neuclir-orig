from typing import Iterator, List, Dict, Tuple

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField, ListField
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

import numpy as np
import csv

def tokenize(line: str) -> List[Token]:
    return [Token(word) for word in line.split()]

@DatasetReader.register('letor_reader')
class ClirDatasetReader(DatasetReader):
    def __init__(self, scores: bool = True, query_token_indexers: Dict[str, TokenIndexer] = None,
                 doc_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.scores = scores
        self.q_token_indexers = query_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.d_token_indexers = doc_token_indexers or {'tokens': SingleIdTokenIndexer()}

    def line_to_instance(self, query: List[Token],
                         *docs: List[Tuple[List[Token], float]],
                         relevant_ix: int = None) -> Instance:
        query_field = TextField(query, self.q_token_indexers)
        doc_fields = [TextField(doc[0], self.d_token_indexers) for doc in docs]

        fields = {
            'query': query_field,
            'docs': ListField(doc_fields),
        }

        if self.scores:
            lex_fields = [ArrayField(np.array([doc[1]])) for doc in docs]
            fields['scores'] = ListField(lex_fields)

        if relevant_ix is not None:
            label_field = LabelField(label=relevant_ix, skip_indexing=True)
            fields['labels'] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        print(f'Opening file path: {file_path}')
        with open(file_path) as f:
            for line in csv.DictReader(f, delimiter='\t'):
                instance = self.line_to_instance(
                    tokenize(line['query']),
                    (tokenize(line['d1']), float(line['s1'])),
                    (tokenize(line['d2']), float(line['s2'])),
                    relevant_ix = int(line['relevant_ix'])
                )
                yield instance
