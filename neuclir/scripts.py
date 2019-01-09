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
    dataset = OnlineDatasetGenerator(config['reranking'], config['systems']).sample_dataset()
    print(dataset)
    # second, run the model and get the output

    # third, sort the output and convert it to proper lines
