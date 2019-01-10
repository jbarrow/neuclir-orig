import sys
import json
import logging

from .predictors import OnlinePredictor
from .datasets import OnlineDatasetGenerator
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

if __name__ == '__main__':
    if len(sys.argv) != 3:
        logging.log(logging.ERROR, 'Script called incorrectly. Cannot continue.')

    config_file = sys.argv[1]
    weights = sys.argv[2]
    with open(config_file) as fp:
        config = json.load(fp)

    # load dataset generator file
    dataset = OnlineDatasetGenerator(config['datasets']['reranking'], config['systems']).sample_dataset()
    # second, run the model and get the output
    archive = load_archive(weights,
        cuda_device=-1,
        overrides='{model: {predicting: true}, dataset_reader: { type: "reranking_dataset_reader" }}')
    predictor = Predictor.from_archive(archive, 'online_predictor')
    json_outputs = predictor.predict_json(
        config['query'],
        dataset
    )
    # print the appropriate json outputs so the file can read it
    for i, (base_path, doc_id, score) in enumerate(json_outputs):
        doc_name = os.path.join(base_path, doc_id + '.txt')

        print(f'0 Q0 {doc_name} {i} {score} neuclir')
