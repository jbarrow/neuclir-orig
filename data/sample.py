from .settings import settings

from sklearn.model_selection import train_test_split
from typing import List, Tuple, Callable, Dict
from scipy.special import logsumexp
from functools import reduce

import pandas as pd
import argparse
import logging
import random
import glob
import json
import csv
import os

from dataclasses import dataclass

# type aliases to make code more readable

Path = str
DocumentID = str
QueryID = str

@dataclass
class Score:
    document: DocumentID
    score: float

    @classmethod
    def from_row(cls, row) -> 'Score':
        return Score(row.doc_id, row.score)


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

def read_judgements(file: Path) -> pd.DataFrame:
    return pd.read_csv(file, sep='\t')

def read_tokens(filename: Path, flatten: bool = True, is_json: bool = True) -> List[str]:
    tokens = []
    with open(filename) as fp:
        for line in fp:
            ts = []
            if is_json:
                l = json.loads(line)
                ts = [w['word'] for w in l[0]] if len(l) > 0 else []
            else:
                ts = line.strip().split()
            tokens.extend(ts) if flatten else tokens.append(ts)
    return tokens

def read_query(filename: Path) -> List[str]:
    with open(filename) as fp:
        query = json.load(fp)
    return query['english']['words'].strip().split()
    # return query['english']['words'] + query['english']['expanded']['expanded_words']
    # query['translations'][1]['translation']

# the sampling functions take the dataframe and the index of the given document
# and returns the index of the sampled document according to its sampling
# strategy

def sample_difficult(df: pd.DataFrame, doc: int, n: int = 1) -> List[int]:
    """
    The sample_difficult strategy attempts to find documents whose scores are
    close to each other to sample.
    """
    ixs = df.index.astype(int)
    pos = doc - ixs[0]

    return random.sample(set(df.head(pos+100).index.astype(int)) - set([doc]), n)

def sample_random(df: pd.DataFrame, doc: int, n: int = 1) -> List[int]:
    """
    The sample_random strategy just randomly samples documents independent
    """
    return random.sample(set(df.index.astype(int)) - set([doc]), n)

# we then use the sampling strategy to sample document pairs

def sample_document_pairs(
        df: pd.DataFrame, n: int = 2,
        fsample: Callable[[pd.DataFrame, int, int], List[int]] = sample_random)\
            -> List[Tuple[QueryID, Score, List[Score]]]:
    sampled_pairs = []
    for positive in df[df['relevant'] == 1].itertuples():
        ix = positive.Index
        sampled = fsample(df[df['query_id'] == positive.query_id], ix)
        sampled = [Score.from_row(df.iloc[sample])  for sample in sampled]
        sampled_pairs.append((positive.query_id, Score.from_row(positive), sampled))
    return sampled_pairs

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

def sum_to_one(gamma):
    pass

def flatten(l_of_ls):
    return [i for l in l_of_ls for i in l]

_dispatch = {
    'random': sample_random,
    'difficult': sample_difficult
}


if __name__ == '__main__':
    logging.basicConfig(level=logging.getLevelName(settings['logging'].upper()))

    # read in all the IARPA relevance judgements files
    judgements = pd.DataFrame()
    for js in settings['judgements']:
        j_file = os.path.join(settings['base_dir'], js, 'query_annotation.tsv')
        logging.log(logging.INFO, f' * Reading judgments from: {j_file}')
        judgements = pd.concat([judgements, read_judgements(j_file)], sort=False)
        judgements['relevant'] = 1
    logging.log(logging.INFO, f' * Total Relevance Judgements: {judgements.shape[0]}')

    # read in all the system score files
    scores = pd.DataFrame()
    for ss in settings['scores']:
        score_dir = os.path.join(settings['base_dir'], ss)
        logging.log(logging.INFO, f' * Reading trec scores from {score_dir}')
        scores = pd.concat([scores, read_scores(score_dir)], sort=False)

    queries_with_judgements = pd.DataFrame({'query_id': sorted(set(judgements.query_id))})

    # merge the files where the query_id and doc_id both exist
    logging.log(logging.INFO, f' * Merging scores and judgement data')
    merged = scores.merge(judgements, how='left', on=['query_id', 'doc_id'])\
                .fillna(0)\
                .merge(queries_with_judgements, how='right', on='query_id')\
                .dropna()
    merged['relevant'] = merged['relevant'].apply(int)
    total_samples = (merged['relevant'] == 1).sum()

    logging.log(logging.INFO, f' * Sampling {total_samples} relevant/irrelevant pairs')
    sampled = sample_document_pairs(merged, fsample=_dispatch[settings['strategy']])

    logging.log(logging.INFO, f' * Loading files')
    query_map = dict_from_paths([os.path.join(settings['base_dir'], p, '*') for p in settings['queries']])
    logging.log(logging.INFO, f' * {len(query_map)} total queries')
    doc_map = dict_from_paths([os.path.join(settings['base_dir'], p, '*') for p in settings['docs']])
    logging.log(logging.INFO, f' * {len(doc_map)} total text docs')
    audio_map = dict_from_paths([os.path.join(settings['base_dir'], p, '*') for p in settings['audio']])
    logging.log(logging.INFO, f' * {len(audio_map)} total audio docs')
    # combine the audio files and docs
    doc_map.update(audio_map)

    sampled_train, sampled_test = train_test_split(sampled,
        train_size=settings['splits']['training'])

    for lst, filename in [(sampled_train, 'train.csv'), (sampled_test, 'test.csv')]:
        print(os.path.join(settings['output'], filename))

        with open(os.path.join(settings['output'], filename), 'w') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerow(['query', 'relevant_ix', 'd1', 's1', 'd2', 's2'])
            for query, relevant, irrelevant in lst:
                q_tokens = read_query(query_map[query])
                r_tokens = (read_tokens(doc_map[relevant.document], is_json='audio' not in doc_map[relevant.document]), relevant.score)
                i_tokens = [(read_tokens(doc_map[i.document], is_json='audio' not in doc_map[i.document]), i.score) for i in irrelevant]
                position = random.randint(0, len(i_tokens))

                tokens = i_tokens[:position] + [r_tokens] + i_tokens[position:]

                writer.writerow(
                    [' '.join(q_tokens), position] + flatten([[' '.join(t), score] for t, score in tokens])
                )
