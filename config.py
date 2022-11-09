import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ih', default=256, type=int)
    parser.add_argument('--latentDim', default=2, type=int)
    parser.add_argument('--batchSize', default=8, type=int)
    parser.add_argument('--learnRate', default=5e-5, type=float)
    parser.add_argument('--testTrainSplit', default=5e-5, type=float)
    parser.add_argument('--epochs', default=3000, type=int)
    
    arg = parser.parse_args()

    return arg

