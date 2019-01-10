from .sample import DatasetGenerator, dict_from_paths
from typing import List, Dict, Any

import pandas as pd
import os

class OnlineDatasetGenerator(DatasetGenerator):
    def __init__(self, params: Dict[str, Any], systems: List[str] = []):
        self.params = params
        self.systems = systems

    def read_scores_file(self, file: str) -> pd.DataFrame:
        """
        Read a .trec file for a specific query and return the dataframe.
        """
        try:
            df = pd.read_csv(file, sep=' ', header=None,
                names=['query_id', 'Q0', 'document_id', 'rank', 'score', 'system'],
                usecols=['document_id', 'score'])
        except pd.errors.EmptyDataError:
            logging.log(logging.WARN, f' * No matcher results for {file}')
            df = pd.DataFrame()
        return df

    def sample_dataset(self) -> List[Dict[str, Any]]:
        lines = []

        scores = {}
        for system in self.systems:
            scores[system] = self.read_scores_file(self.params['scores'][system])

        docs = dict_from_paths(self.params['docs'])
        doc_ids = set(pd.concat([df for system, df in scores.items() if system in self.systems]).document_id)
        for doc in doc_ids:
            head, tail = os.path.split(doc)
            tail = tail.split('.')[0]

            lines.append({
                'document_id': tail,
                'base_path': head,
                'text': ' '.join(self.read_tokens(docs[tail], is_json=False)),
                'scores': self.get_scores(scores, tail)
            })

        return lines
