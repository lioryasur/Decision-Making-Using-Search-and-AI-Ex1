from heuristics import *


def bellmanUpdateTraining(bellman_update_heuristic, path='./', iterations=1001):
    bellman = bellman_update_heuristic(dropout=0.3)
    bellman.train_iterations(path, iterations=iterations)


def bootstrappingTraining(bootstrapping_heuristic, path='./', iterations=1001):
    bootstrap = bootstrapping_heuristic()
    bootstrap.train_bootstrap(path=path, batch_size=5000, initial_T=2000, W=1.5, B=10, num_epochs=iterations)

