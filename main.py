from heuristics import BaseHeuristic, BellmanUpdateHeuristic, BootstrappingHeuristic
from BWAS import BWAS
from topspin import TopSpinState
import torch

instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance

start1 = TopSpinState(instance_1, 4)
start2 = TopSpinState(instance_2, 4)

base_heuristic = BaseHeuristic(11, 4)
path, expansions = BWAS(start2, 5, 10, base_heuristic.get_h_values, 1000000)
print(f'start1 heuristic: {base_heuristic.get_h_values([start1])}')
print(f'start2 heuristic: {base_heuristic.get_h_values([start2])}')

if path is not None:
    print(f'expansions: {expansions}')
    print(f'path length: {len(path)}')
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")

with torch.no_grad():
    for iteration in range(0, 1001, 50):
        print(f'iteration: {iteration}')

        BU_heuristic = BellmanUpdateHeuristic(dropout=0.3)
        BU_heuristic.load_model(path='../', iteration=iteration)
        BU_heuristic._model.to('cpu')
        BU_heuristic._model.eval()
        print(f'start1 heuristic: {BU_heuristic.get_h_values([start1])}')
        print(f'start2 heuristic: {BU_heuristic.get_h_values([start2])}')
        # path, expansions = BWAS(start2, 5, 10, BU_heuristic.get_h_values, 1000000)
        # if path is not None:
        #     print(expansions)
        #     for vertex in path:
        #         print(vertex)
        # else:
        #     print("unsolvable")
