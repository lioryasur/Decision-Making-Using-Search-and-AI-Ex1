from heuristics import *
from BWAS import *
from topspin import *
def bellmanUpdateTraining(bellman_update_heuristic, path='./'):
    bellman = bellman_update_heuristic(dropout=0.3)
    bellman.train_iterations(path, iterations=10001)
    
def bootstrappingTraining(bootstrapping_heuristic):
    pass


if __name__ == '__main__':
    bellmanUpdateTraining(BellmanUpdateHeuristic, path='../')