import sys
import logging

from .datasets import OnlineDatasetGenerator

if __name__ == '__main__':
    if len(sys.argv) != 2:
        logging.log(logging.ERROR, 'Script called incorrectly. Cannot continue.')

    sample_file = sys.argv[1]
    # load dataset generator file

    # dataset = OnlineDatasetGenerator().sample_dataset()

    # second, run the model and get the output

    # third, sort the output and convert it to proper lines

    print(open(sample_file).read())
