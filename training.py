from heuristics import *


def bellmanUpdateTraining(bellman_update_heuristic, path='./'):
    bellman = bellman_update_heuristic(dropout=0.3)
    bellman.load_model(path='../', iteration=4400)
    bellman.train_iterations(path, iterations=40001)


def bootstrappingTraining(bootstrapping_heuristic, path='./'):
    bootstrap = bootstrapping_heuristic()
    bootstrap.train_bootstrap(path=path, batch_size=5000, initial_T=2000, W=1.5, B=10, num_epochs=2000, )


if __name__ == '__main__':
    bellmanUpdateTraining(BellmanUpdateHeuristic, path='../')
