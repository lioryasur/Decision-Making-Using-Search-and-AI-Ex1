from heuristics import BaseHeuristic, BellmanUpdateHeuristic, BootstrappingHeuristic
from BWAS import BWAS
from topspin import TopSpinState
import torch
import pickle
import numpy as np

def take_action(state, action):
    if action == 0:
        return torch.roll(state, 1)
    elif action == 1:
        return torch.roll(state, -1)
    else:
        flipped_state = torch.cat((state[:4].flip(dims=[0]), state[4:]))
    return flipped_state


def take_m_actions(state, num_actions):
    for n in range(num_actions):
        action = np.random.randint(0, 3)
        state = take_action(state, action)
    return state


def generate_batch_states(m, batch_size):
    goal_tensor = torch.tensor(list(range(11)))
    batch_states = []
    for i in range(batch_size):
        state = goal_tensor.clone()
        state = take_m_actions(state, m)
        batch_states.append(state)

    return torch.stack(batch_states)

pickle.dump(generate_batch_states(100,100), 'test_states.pkl')

try:
    states = pickle.load(open('test_states.pkl', 'rb'))
    heuristics_dict = {'Basic': BaseHeuristic(11,4).get_h_values, 'Learned-Bellman': BellmanUpdateHeuristic(11,4).get_h_values,
                       'Learned-Bootstrap': BootstrappingHeuristic.get_h_values}
    table = {'W': [], 'B': [], 'Heuristic': [], 'Runtime': [], 'Path Length': [], 'Number of Expansions': []}
    for state in states:
        for w in [2,5]:
            for b in [1,100]:
                for cur_heuristic in ['Basic', 'Learned-Bellman', 'Learned-Bootstrap']:
                    table['W'].append(w)
                    table['B'].append(b)
                    table['Heuristic'].append(cur_heuristic)
                    state_list = list(state)
                    instance = TopSpinState(state_list, 4)
                    start_time = time()
                    path, expansions = BWAS(instance, w, b, heuristics_dict[cur_heuristic], 5000)
                    table['Runtime'].append(start_time)
                    table['Path Length'].append(len(path))
                    table['Number of Expansions'].append(expansions)


# instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
# instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance
#
# start1 = TopSpinState(instance_1, 4)
# start2 = TopSpinState(instance_2, 4)
#
# base_heuristic = BaseHeuristic(11, 4)
# path, expansions = BWAS(start2, 5, 10, base_heuristic.get_h_values, 1000000)
# print(f'start1 heuristic: {base_heuristic.get_h_values([start1])}')
# print(f'start2 heuristic: {base_heuristic.get_h_values([start2])}')
#
# if path is not None:
#     print(f'expansions: {expansions}')
#     print(f'path length: {len(path)}')
#     # for vertex in path:
#     #     print(vertex)
# else:
#     print("unsolvable")
#
# with torch.no_grad():
#     for iteration in range(0, 1001, 50):
#         print(f'iteration: {iteration}')
#
#         BU_heuristic = BellmanUpdateHeuristic(dropout=0.3)
#         BU_heuristic.load_model(path='../', iteration=iteration)
#         BU_heuristic._model.to('cpu')
#         BU_heuristic._model.eval()
#         print(f'start1 heuristic: {BU_heuristic.get_h_values([start1])}')
#         print(f'start2 heuristic: {BU_heuristic.get_h_values([start2])}')
#         # path, expansions = BWAS(start2, 5, 10, BU_heuristic.get_h_values, 1000000)
#         # if path is not None:
#         #     print(expansions)
#         #     for vertex in path:
#         #         print(vertex)
#         # else:
#         #     print("unsolvable")
#
# #def test_heuristics(heuristics):
