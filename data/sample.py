from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Set
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

def read_scores_file(file: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file, sep='\t', header=None,
            names=['query_id', 'Q0', 'doc_id', 'rank', 'score', 'system'])
    except pd.errors.EmptyDataError:
        logging.log(logging.WARN, f' * No matcher results for {file}')
        return pd.DataFrame()

def read_scores(path: Path) -> pd.DataFrame:
    return pd.concat(
        [read_scores_file(qf) for qf in glob.glob(os.path.join(path, '*.trec'))])

def read_tokens(filename: Path, flatten: bool = True, is_json: bool = True) -> List[str]:
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

def read_query(filename: Path) -> List[str]:
    with open(filename) as fp:
        query = json.load(fp)
    return query['english']['words'].strip().split()

# the sampling functions take the dataframe and the index of the given document
# and returns the index of the sampled document according to its sampling
# strategy

def sample_random(docs: Set[str], n: int = 1) -> List[str]:
    """
    The sample_difficult strategy attempts to find documents whose scores are
    close to each other to sample.
    """
    return random.sample(docs, n)

def sample_pooled(df: pd.DataFrame, n: int = 1) -> List[str]:
    """
    The sample_random strategy just randomly samples documents independent
    """
    return  [(df.iloc[sample].doc_id) for sample in random.sample(set(range(len(df))), n)]


# when constructing the document and query pairs we need to map the names
# to a file:

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

_dispatch = {
    'random': sample_random,
    'pooled': sample_pooled,
#    'pooled_plus': sample_pooled_plus
}

def read_judgements(file: Path) -> pd.DataFrame:
    judgements = pd.read_csv(file, sep='\t')
    judgements['relevant'] = 1
    return judgements

def get_score(df: pd.DataFrame, doc_id: str) -> float:
    found = df[df.doc_id == doc_id]
    if len(found) == 0:
        if len(df) == 0:
            return -100.
        return min(df.score)
    return found.score.values[0]

def get_relevant(df: pd.DataFrame, doc_id: str) -> int:
    found = df[df.doc_id == doc_id]
    if len(found) == 0:
        return 0
    return 1

if __name__ == '__main__':
    settings = json.load(open('sample.json'))
    # set up the logger
    logging_level = settings.pop('logging', 'info').upper()
    logging.basicConfig(level=logging.getLevelName(logging_level))
    # get the basic settings
    strategy = settings.pop('strategy', 'random')
    include_scores = settings.pop('include_scores', False)
    n_irrelevant = settings.pop('n_irrelevant', 1)
    output = settings.pop('output', '/tmp/data')

    scores, judgements, datasets = {}, {}, {}
    queries, docs = {}, {}

    # load data for each dataset
    for dataset in ['train', 'test', 'validation']:
        # get the settings for the dataset
        datasets[dataset] = settings.pop(dataset, None)
        # if the user didn't provide settings, we simply skip it
        if datasets[dataset] is None:
            datasets.pop(dataset)
            continue
        logging.log(logging.INFO, f'Preparing the {dataset} dataset:')
        # read in all the judgements
        logging.log(logging.INFO, ' * Reading in relevance judgements')
        judgement_files = datasets[dataset].pop('judgements', [])
        judgements[dataset] = pd.concat([read_judgements(f) for f in judgement_files])
        logging.log(logging.INFO, f' * Read in {len(judgements[dataset])} total judgements')
        # determine which queries actually have any relevance judgements
        queries_with_judgements = pd.DataFrame({'query_id': sorted(set(judgements[dataset].query_id))})
        logging.log(logging.INFO, f' * There are {len(queries_with_judgements)} total queries with judgements')
        # read in all the INDRI scores
        score_files = datasets[dataset].pop('scores', [])
        # ensure that we either have score files or don't need them
        if len(score_files) > 0:
            logging.log(logging.INFO, ' * Reading in INDRI scores')
            scores[dataset] = pd.concat([read_scores(f) for f in score_files], sort=False)
            logging.log(logging.INFO, f' * Read in {len(scores[dataset])} total scores')
        elif (strategy == 'difficult' and dataset == 'train') or dataset == 'validation':
            logging.log(logging.ERROR, 'Scores were needed but not provided')
            sys.exit(-1)
        # load in the possible queries
        queries[dataset] = dict_from_paths(datasets[dataset]['queries'])
        logging.log(logging.INFO, f' * Found {len(queries[dataset])} queries')
        # load in the possible documents
        docs[dataset] = dict_from_paths(datasets[dataset]['docs'])
        logging.log(logging.INFO, f' * Found {len(docs[dataset])} docs')

    # generate the train dataset
    if 'train' in datasets.keys():
        out_fp = open(os.path.join(output, 'train.json'), 'w')
        # when generating a training set, we want to associate each relevant
        # (query, doc) pair with one or more irrelevant (query, doc) pairs.
        for query, df in tqdm(judgements['train'].groupby('query_id')):
            query_text = read_query(queries['train'][query])
            query_scores = scores['train'][scores['train']['query_id'] == query]
            # if we choose a sampling strategy that requires it
            if strategy in ['pooled']:
                # exclude relevant docs
                relevant = query_scores['doc_id'].isin(list(df.doc_id))
                possible = query_scores[~relevant]
            all_docs = set(docs['train'].keys()) - set(df.doc_id)
            # sample irrelevant docs for every relevant doc
            for judgement in df.itertuples():
                # sample a position to put the relevant document in
                position = random.randint(0, n_irrelevant)
                # sample paired documents
                if strategy == 'pooled' and len(possible) > n_irrelevant:
                    sampled_docs = sample_pooled(possible, n_irrelevant)
                else:
                    sampled_docs = sample_random(all_docs, n_irrelevant)
                # get the tokens for the documents
                relevant = {
                    'text': ' '.join(read_tokens(docs['train'][judgement.doc_id])),
                    'score': get_score(query_scores, judgement.doc_id)
                }
                # get the tokens for the irrelevant documents
                irrelevant = [{
                    'text': ' '.join(read_tokens(docs['train'][doc_id])),
                    'score': get_score(query_scores, doc_id)
                } for doc_id in sampled_docs]

                # generate the output json line
                outline = json.dumps({
                    'query': ' '.join(query_text),
                    'docs': irrelevant[:position] + [relevant] + irrelevant[position:],
                    'relevant': position
                })
                # write the line to the json file
                out_fp.write(outline + '\n')
        out_fp.close()

    # generate the validation dataset
    if 'validation' in datasets.keys():
        out_fp = open(os.path.join(output, 'validation.json'), 'w')
        # when generating a validation set, our goal is to rerank, so we want
        # to load in all scores for a given query, up to, say, 100.
        for query, df in tqdm(scores['validation'].groupby('query_id')):
            query_text = read_query(queries['validation'][query])
            relevant = judgements['validation'][judgements['validation']['query_id'] == query]
            # compute the number of relevant docs that we missed
            ignored_relevant = len(set(relevant.doc_id) - set(df.doc_id))
            ignored_irrelevant = len(docs['validation']) - len(df) - ignored_relevant
            scored_docs = []
            # only look at the 100 most relevant documents (we'll change this later)
            for row in df.itertuples():
                scored_docs.append({
                    'text': ' '.join(read_tokens(docs['validation'][row.doc_id])),
                    'score': row.score,
                    'relevant': get_relevant(relevant, row.doc_id)
                })
            # write a single query
            outline = json.dumps({
                'query': ' '.join(query_text),
                'docs': scored_docs,
                'ignored_relevant': ignored_relevant,
                'ignored_irrelevant': ignored_irrelevant
            })
            out_fp.write(outline + '\n')

        out_fp.close()

    # eventually we'll want to compute a test set, which doesn't have any
    # labeled documents and only consumes scores
