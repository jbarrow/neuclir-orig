import sys
import json
import logging

from .datasets import OnlineDatasetGenerator

if __name__ == '__main__':
    if len(sys.argv) != 2:
        logging.log(logging.ERROR, 'Script called incorrectly. Cannot continue.')

    config_file = sys.argv[1]
    with open(config_file) as fp:
        config = json.load(fp)

    # load dataset generator file
    dataset = OnlineDatasetGenerator(config['dataset']['reranking'], config['systems']).sample_dataset()
    # second, run the model and get the output
    archive = load_archive('/home/joe/projects/model.tar.gz',
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
