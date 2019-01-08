from typing import Iterator

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader


@DatasetReader.register('ranking_dataset_reader')
class RankingDatasetReader(DatasetReader):
    def __init__(self): pass
    def _read(self, file_path: str) -> Iterator[Instance]: pass
