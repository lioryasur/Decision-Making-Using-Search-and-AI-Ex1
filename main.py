from heuristics import *
from BWAS import *
from topspin import *
from training import *
import torch
import numpy as np

instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance

start1 = TopSpinState(instance_1, 4)
base_heuristic = BaseHeuristic(11, 4)
path, expansions = BWAS(start1, 5, 10, base_heuristic.get_h_values, 1000000)
if path is not None:
    print(expansions)
    for vertex in path:
        print(vertex)
else:
    print("unsolvable")

start2 = TopSpinState(instance_2, 4)
BU_heuristic = BellmanUpdateHeuristic(11, 4)
path, expansions = BWAS(start2, 5, 10, BU_heuristic.get_h_values, 1000000)
if path is not None:
    print(expansions)
    for vertex in path:
        print(vertex)
else:
    print("unsolvable")
