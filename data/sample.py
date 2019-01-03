from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Set, Any, Union
from scipy.special import logsumexp
from functools import reduce
from tqdm import tqdm

import pandas as pd
import argparse
import logging
import random
import glob
import json
import sys
import os


Path = str
DocumentID = str
QueryID = str

def dict_from_paths(paths: List[Path], strip_file: bool = True) -> Dict[str, Path]:
    output = {}
    for path in paths:
        for f in glob.glob(path):
            filename = os.path.basename(f)
            filename = os.path.splitext(filename)[0] if strip_file else filename
            output[filename] = f
    return output

# normalization functions
def flatten(l_of_ls):
    return [i for l in l_of_ls for i in l]

class DatasetGenerator(object):
    def __init__(self, name: str, params: Dict[str, Any], output: Path, systems: List[str] = []):
        self.params = params
        self.output = output
        self.name = name
        self.systems = systems

    def read_scores_file(self, file: Path) -> Tuple[str, pd.DataFrame]:
        """
        Read a .trec file for a specific query and return the dataframe.
        """

        try:
            # load the tsv file
            df = pd.read_csv(file, sep='\t', header=None,
                names=['query_id', 'Q0', 'document_id', 'rank', 'score', 'system'],
                usecols=['document_id', 'score'])
            # get the query id from the header
            q_id = os.path.splitext(os.path.basename(file))[0][2:]

            return q_id, df
        except pd.errors.EmptyDataError:
            logging.log(logging.WARN, f' * No matcher results for {file}')
            return '', pd.DataFrame()

    def read_scores(self, systems: Dict[str, Union[Path, List[Path]]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Return a dict that maps the query id to the system and dataframe containing the
        document-level scores for each system.
        """
        scores = defaultdict(dict)
        for system, paths in systems.items():
            if type(paths) != list:
                paths = [paths]
            for path in paths:
                logging.log(logging.INFO, f'  - Reading in scores for "{system}" from {path}')
                score_dfs = [self.read_scores_file(qf) for qf in glob.glob(os.path.join(path, '*.trec'))]
                for query, score_df in score_dfs:
                    if system not in scores[query]:
                        scores[query][system] = score_df
                    else:
                        scores[query][system] = pd.concat([scores[query][system], score_df])

        return scores

    def read_query(self, filename: Path, query_type: str = 'words') -> List[str]:
        with open(filename) as fp:
            query = json.load(fp)
        return query['english'][query_type].strip().split()

    def read_tokens(self, filename: Path, flatten: bool = True, is_json: bool = True) -> List[str]:
        tokens = []
        with open(filename) as fp:
            for line in fp:
                ts = []
                if is_json:
                    l = json.loads(line)
                    ts = [w['word'].lower() for w in l[0]] if len(l) > 0 else []
                else:
                    ts = line.strip().lower().split()
                tokens.extend(ts) if flatten else tokens.append(ts)
        return tokens

    def read_judgements(self, file: Path) -> pd.DataFrame:
        logging.log(logging.INFO, f'  - Reading in judgements from {file}')
        judgements = pd.read_csv(file, sep='\t')
        judgements['relevant'] = 1
        return judgements

    def get_scores(self, scores: Dict[str, pd.DataFrame], doc_id: str) -> Dict[str, float]:
        all_scores = []
        for system in self.systems:
            df = scores[system]

            found = df[df.document_id == doc_id]
            score = -100.
            if len(found) == 0 and len(df) > 0:
                score = min(df.score)
            elif len(found) > 0:
                score = found.score.values[0]
            all_scores.append({'system': system, 'score': score})
        return all_scores

    def get_relevant(self, df: pd.DataFrame, doc_id: str) -> int:
        found = df[df.doc_id == doc_id]
        if len(found) == 0:
            return 0
        return 1

    def prepare(self):
        logging.log(logging.INFO, 'Loading in data')
        # get the system scores
        logging.log(logging.INFO, ' * Reading in score files')
        self.scores = self.read_scores(self.params['scores'])
        # get the relevance judgements
        logging.log(logging.INFO, ' * Reading in relevance judgements')
        judgement_files = self.params.pop('judgements', [])
        self.judgements = pd.concat([self.read_judgements(f) for f in judgement_files])
        # get the queries with judgements
        self.queries_with_judgements = pd.DataFrame({'query_id': sorted(set(self.judgements.query_id))})
        logging.log(logging.INFO, f' * There are {len(self.queries_with_judgements)} total queries with judgements')
        # get all the queries
        self.queries = dict_from_paths(self.params['queries'])
        logging.log(logging.INFO, f' * Found {len(self.queries)} total queries')
        # load in the possible documents
        self.docs = dict_from_paths(self.params['docs'])
        logging.log(logging.INFO, f' * Found {len(self.docs)} total docs')
        # # impose an order on the systems
        # self.systems = sorted(self.params['scores'].keys())

    def normalize(self, scores: pd.DataFrame) -> pd.DataFrame:
        return scores

    def sample_dataset(self):
        pass

def s_random(docs: Set[str], n: int = 1) -> List[str]:
    return random.sample(docs, n)

def s_difficult(df: pd.DataFrame, n: int = 1) -> List[str]:
    return  [(df.iloc[sample].document_id) for sample in random.sample(set(range(len(df))), n)]

class PairedDatasetGenerator(DatasetGenerator):
    _dispatch = {
        'random': s_random,
    #    'difficult': s_difficult,
    #    'pooled_plus': sample_pooled_plus
    }

    def sample_dataset(self) -> None:
        logging.log(logging.INFO, f'Creating PAIRED dataset: {self.name}')
        # load all the scores, judgements, and data files
        self.prepare()
        # open the output file and begin sampling the dataset
        out_fp = open(os.path.join(self.output, f'{self.name}.json'), 'w')
        # when generating a training set, we want to associate each relevant
        # (query, doc) pair with one or more irrelevant (query, doc) pairs.
        with tqdm(total=len(self.judgements)) as pbar:
            for query, df in self.judgements.groupby('query_id'):
                query_text = self.read_query(self.queries[query])
                query_scores = self.scores[query]
                # if we choose a sampling strategy that requires it
                # if strategy in ['pooled']:
                #     # exclude relevant docs
                #     relevant = query_scores['doc_id'].isin(list(df.doc_id))
                #     possible = query_scores[~relevant]
                all_docs = set(self.docs.keys()) - set(df.doc_id)
                # sample irrelevant docs for every relevant doc
                for judgement in df.itertuples():
                    # sample a position to put the relevant document in
                    position = random.randint(0, self.params['n_irrelevant'])
                    # sample paired documents
                    # if strategy == 'pooled' and len(possible) > n_irrelevant:
                    #     sampled_docs = sample_pooled(possible, n_irrelevant)
                    # else:
                    sampled_docs = s_random(all_docs, self.params['n_irrelevant'])
                    # get the tokens for the documents
                    relevant = {
                        'text': ' '.join(self.read_tokens(self.docs[judgement.doc_id])),
                        'scores': self.get_scores(query_scores, judgement.doc_id)
                    }
                    # get the tokens for the irrelevant documents
                    irrelevant = [{
                        'text': ' '.join(self.read_tokens(self.docs[doc_id])),
                        'scores': self.get_scores(query_scores, doc_id)
                    } for doc_id in sampled_docs]

                    # generate the output json line
                    outline = json.dumps({
                        'query': ' '.join(query_text),
                        'docs': irrelevant[:position] + [relevant] + irrelevant[position:],
                        'relevant': position
                    })
                    # write the line to the json file
                    out_fp.write(outline + '\n')
                    pbar.update(1)
        out_fp.close()

class RerankingDatasetGenerator(DatasetGenerator):
    def has_relevant_docs(self, scores: Dict[str, pd.DataFrame]) -> bool:
        for system, df in scores.items():
            if len(df) > 0 and system in self.systems:
                return True
        return False

    def sample_dataset(self) -> None:
        logging.log(logging.INFO, f'Creating RERANKING dataset: {self.name}')
        # load all the datasets, score files, etc.
        self.prepare()
        # open the file
        dataset_fp = open(os.path.join(self.output, f'{self.name}.json'), 'w')
        scoring_fp = open(os.path.join(self.output, f'{self.name}_scoring.json'), 'w')
        # when generating a validation set, our goal is to rerank, so we want
        # to load in all scores for a given query, up to, say, 100.
        for query, dfs in tqdm(self.scores.items()):
            query_text = self.read_query(self.queries[query])
            relevant = self.judgements[self.judgements['query_id'] == query]
            # compute the number of relevant docs that we missed

            if self.has_relevant_docs(dfs):
                # get the set of all documents returned by the systems
                docs = set(pd.concat([df for system, df in dfs.items() if system in self.systems]).document_id)
                ignored_relevant = len(set(relevant.doc_id) - docs)
                ignored_irrelevant = len(self.docs) - len(docs) - ignored_relevant
                scored_docs = []

                for doc in docs:
                    scored_docs.append({
                        'document_id': doc,
                        'text': ' '.join(self.read_tokens(self.docs[doc])),
                        'scores': self.get_scores(dfs, doc),
                        'relevant': self.get_relevant(relevant, doc)
                    })
                # write a single query
                outline = json.dumps({
                    'query_id': query,
                    'query': ' '.join(query_text),
                    'docs': scored_docs,
                    'ignored_relevant': ignored_relevant,
                    'ignored_irrelevant': ignored_irrelevant
                })
                dataset_fp.write(outline + '\n')
            else:
                ignored_relevant = len(relevant)
                ignored_irrelevant = len(self.docs) - ignored_relevant

                outline = json.dumps({
                    'query_id': query,
                    'ignored_relevant': ignored_relevant,
                    'ignored_irrelevant': ignored_irrelevant
                })

                scoring_fp.write(outline + '\n')

        dataset_fp.close()
        scoring_fp.close()


class RankingDatasetGenerator(DatasetGenerator):
    def __init__(self):
        pass

    def sample_dataset():
        pass

_dataset_dispatch = {
    'paired': PairedDatasetGenerator,
    'reranking': RerankingDatasetGenerator,
#    'ranking': RankingDatasetGenerator
}

if __name__ == '__main__':
    settings = json.load(open('sample.json'))
    # set up the logger
    logging_level = settings.pop('logging', 'info').upper()
    logging.basicConfig(level=logging.getLevelName(logging_level))

    for name, dataset in settings['datasets'].items():
        generator = _dataset_dispatch[dataset['type']](name, dataset, settings['output'], settings['systems'])
        generator.sample_dataset()


# 6. match document and query ids
# 7. output trec and tsv files so we can use existing aqwv scorer
