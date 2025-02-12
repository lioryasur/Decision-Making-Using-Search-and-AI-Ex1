import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from BWAS import *
import math


class BaseHeuristic:
    def __init__(self, n=11, k=4):
        self._n = n
        self._k = k

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        gaps = []

        for state_as_list in states_as_list:
            gap = 0
            if state_as_list[0] != 1:
                gap = 1

            for i in range(len(state_as_list) - 1):
                if abs(state_as_list[i] - state_as_list[i + 1]) != 1:
                    gap += 1

            gaps.append(gap)

        return gaps


class HeuristicModel(nn.Module):
    def __init__(self, state_dim: int = 11, h1_dim: int = 5000, resnet_dim: int = 1000, num_resnet_blocks: int = 4,
                 out_dim: int = 1, batch_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.num_resnet_blocks = num_resnet_blocks
        self.batch_norm = batch_norm
        self.dropout_prob = dropout

        self.fc1 = nn.Linear(state_dim, h1_dim)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)
        self.fc2 = nn.Linear(h1_dim, resnet_dim)
        if batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_resnet_blocks):
            layers = [nn.Linear(resnet_dim, resnet_dim)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(resnet_dim))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout_prob > 0.0:
                layers.append(nn.Dropout(p=self.dropout_prob))
            layers.append(nn.Linear(resnet_dim, resnet_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(resnet_dim))
            self.blocks.append(nn.Sequential(*layers))

        self.fc_out = nn.Linear(resnet_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        if self.dropout_prob > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_prob)
        else:
            self.dropout = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)

        for block in self.blocks:
            residual = x
            x = block(x)
            x += residual

        x = self.fc_out(x)
        return x


class LearnedHeuristic:
    def __init__(self, n=11, k=4, dropout=0, lr=0.001):
        self._n = n
        self._k = k
        self._model = HeuristicModel(n, dropout=dropout)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        self._goal = list(range(1, n + 1))
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model.to(self._device)
        self._epoch = 0

    def get_h_values(self, states):
        self._model.eval()
        states_as_list = [state.get_state_as_list() for state in states]
        states = np.array(states_as_list, dtype=np.float32)
        states_tensor = torch.tensor(states).to(self._device)
        with torch.no_grad():
            predictions = self._model(states_tensor).to("cpu").numpy()
        return predictions.flatten()

    def train_model(self, input_data, output_labels, epochs=101):
        input_as_list = [state.get_state_as_list() for state in input_data]
        inputs = np.array(input_as_list, dtype=np.float32)
        outputs = np.array(output_labels, dtype=np.float32)

        inputs_tensor = torch.tensor(inputs).to(self._device)
        outputs_tensor = torch.tensor(outputs).unsqueeze(1).to(self._device)  # Adding a dimension for the output

        for epoch in range(epochs):
            self._model.train()
            self._optimizer.zero_grad()

            predictions = self._model(inputs_tensor)
            loss = self._criterion(predictions, outputs_tensor)
            loss.backward()
            self._optimizer.step()

    def train_model_real(self, inputs_tensor, outputs_tensor, epochs=5):
        for epoch in range(epochs):
            self._model.train()
            self._optimizer.zero_grad()

            predictions = self._model(inputs_tensor.to(self._device, dtype=torch.float))
            loss = self._criterion(predictions, outputs_tensor.to(self._device).unsqueeze(1))

            loss.backward()
            self._optimizer.step()

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._model.eval()

    def take_action(self, state, action):
        if action == 0:
            return torch.roll(state, 1)
        elif action == 1:
            return torch.roll(state, -1)
        else:
            flipped_state = torch.cat((state[:self._k].flip(dims=[0]), state[self._k:]))
        return flipped_state

    def take_m_actions(self, state, num_actions):
        for n in range(num_actions):
            action = np.random.randint(0, 3)
            state = self.take_action(state, action)
        return state

    def generate_batch_states(self, m, batch_size):
        rng = np.random.default_rng()
        goal_tensor = torch.tensor(self._goal, device=self._device)
        batch_states = []

        # Generate random actions for the entire batch at once
        action_counts = rng.integers(0, m, size=batch_size)
        for i in range(batch_size):
            state = goal_tensor.clone()
            state = self.take_m_actions(state, action_counts[i])
            batch_states.append(state)

        return torch.stack(batch_states)


class BellmanUpdateHeuristic(LearnedHeuristic):

    def __init__(self, n=11, k=4, dropout=0.3):
        super().__init__(n, k, dropout)

    def save_model(self, path='', iteration=None):
        if iteration:
            super().save_model(path + f'bellman_update_heuristic_{iteration}.pth')
            torch.save(self._optimizer.state_dict(), path + f'bellman_update_heuristic_optimizer_{iteration}.pth')
        else:
            super().save_model(path + f'bellman_update_heuristic.pth')

    def load_model(self, path='', iteration=None):
        if iteration:
            super().load_model(path + f'bellman_update_heuristic_{iteration}.pth')
            self._optimizer.load_state_dict(
                torch.load(path + f'bellman_update_heuristic_optimizer_{iteration}.pth', map_location=self._device))
            self._epoch = iteration + 1
        else:
            super().load_model(path + f'bellman_update_heuristic.pth')

    def train_iterations(self, save_path='', iterations=1001, batch_size=5000):
        self._model.train()
        for iteration in range(iterations):
            states, labels = self.get_states_and_labels(batch_size,
                                                        m=max(1, int(np.emath.logn(1.15, iterations + self._epoch))))
            self.train_model_real(states, labels, 31)
            if iteration % 100 == 0:
                self.save_model(path=save_path)

    def get_states_and_labels(self, batch_size, m):
        states = self.generate_batch_states(m, batch_size)
        labels = torch.full((batch_size,), float('inf'), device=self._device)
        all_neighbors = self.get_all_neighbors(states)

        # Check if states are equal to the goal
        goal_tensor = torch.tensor(self._goal, device=self._device)
        goal_matches = (states == goal_tensor).all(dim=1)
        labels[goal_matches] = 0

        # Get neighbors
        # Check if any neighbor is the goal
        goal_in_neighbors = torch.zeros(batch_size, dtype=torch.bool).to(self._device)
        for i in range(batch_size):
            if (all_neighbors[i].to(self._device) == torch.Tensor(self._goal).to(self._device)).all(dim=1).any():
                labels[i] = 1
                goal_in_neighbors[i] = 1

        # For non-goal states, compute heuristic values for neighbors
        non_goal_indices = ~goal_matches & ~goal_in_neighbors
        non_goal_neighbors = all_neighbors[non_goal_indices].to(self._device)
        non_goal_neighbors_flat = non_goal_neighbors.view(-1, self._n)
        self._model.eval()
        with torch.no_grad():
            neighbor_hs = self._model(non_goal_neighbors_flat.to(self._device, dtype=torch.float))
        self._model.train()
        neighbor_hs = neighbor_hs.view(-1, 3)

        min_neighbor_hs, _ = torch.min(neighbor_hs, dim=1)
        labels[non_goal_indices] = 1 + min_neighbor_hs

        return states.to(self._device), torch.Tensor(labels).to(self._device)

    def get_all_neighbors(self, states):
        # Generate neighbors by rolling and flipping states in a batch
        rolled_pos = torch.roll(states, shifts=1, dims=1)
        rolled_neg = torch.roll(states, shifts=-1, dims=1)
        flipped = torch.cat((states[:, :self._k].flip(dims=[1]), states[:, self._k:]), dim=1)

        neighbors = torch.stack([rolled_pos, rolled_neg, flipped], dim=1)
        return neighbors


class BootstrappingHeuristic(LearnedHeuristic):
    def __init__(self, n=11, k=4, dropout=0):
        super().__init__(n, k, dropout)

    def save_model(self, path='', iteration=None):
        if iteration:
            super().save_model(path + f'bootstrapping_heuristic_{iteration}.pth')
            torch.save(self._optimizer.state_dict(), path + f'bootstrapping_heuristic_optimizer_{iteration}.pth')
        else:
            super().save_model(path + f'bootstrapping_heuristic.pth')

    def load_model(self, path='', iteration=None):
        if iteration:
            super().load_model(path + f'bootstrapping_heuristic_{iteration}.pth')
            self._optimizer.load_state_dict(
                torch.load(path + f'bootstrapping_heuristic_optimizer_{iteration}.pth', map_location=self._device))
            self._epoch = iteration
        else:
            super().load_model(path + f'bootstrapping_heuristic.pth')

    def run_bwa_star(self, start_state, W, B, T):
        if self._epoch < 1000:
            hb = BaseHeuristic(self._n, self._k).get_h_values
        else:
            hb = self.get_h_values
        result_path, _ = BWAS(start_state, W, B, hb, T, self._k)

        if result_path:
            return True, [torch.tensor(state) for state in result_path]
        return False, None

    def bootstrap(self, m, batch_size, initial_T=100, W=1.5, B=10):
        T = initial_T
        training_examples = []

        while len(training_examples) < batch_size:
            minibatch = self.generate_batch_states(m=m, batch_size=batch_size // 100)
            minibatch = minibatch.to("cpu")
            solved_paths = []

            for state in minibatch:
                solved, path = self.run_bwa_star(list(state), W, B, T)
                if solved:
                    solved_paths.append(path)

            if not solved_paths:
                T *= 2
                continue

            for path in solved_paths:
                n = len(path)
                for i in range(n):
                    s_i = path[i]
                    y_i = n - i - 1
                    training_examples.append((s_i, y_i))

            if len(training_examples) > batch_size:
                training_examples = training_examples[:batch_size]

        inputs, outputs = zip(*training_examples)
        inputs_tensor = torch.stack(inputs).to(self._device)
        outputs_tensor = torch.tensor(outputs, dtype=torch.float32).to(self._device)
        self.train_model_real(inputs_tensor, outputs_tensor, epochs=61)

    def take_action(self, state, action):
        if action == 0:
            return torch.roll(state, 1)
        elif action == 1:
            return torch.roll(state, -1)
        else:
            flipped_state = torch.cat((state[:self._k].flip(dims=[0]), state[self._k:]))
        return flipped_state

    def take_m_actions(self, state, num_actions):
        for n in range(num_actions):
            action = np.random.randint(0, 3)
            state = self.take_action(state, action)
        return state

    def generate_batch_states(self, m, batch_size):
        goal_tensor = torch.tensor(self._goal, device=self._device)
        batch_states = []

        # Generate random actions for the entire batch at once
        action_counts = np.clip(np.round(np.random.normal(m, 3, batch_size)), a_min=0, a_max=np.inf).astype(int)

        for i in range(batch_size):
            state = goal_tensor.clone()
            state = self.take_m_actions(state, action_counts[i])
            batch_states.append(state)

        return torch.stack(batch_states)

    def train_bootstrap(self, batch_size, initial_T=100, W=1.5, B=10, num_epochs=1001, path=''):
        for epoch in range(self._epoch, self._epoch + num_epochs):
            self.bootstrap(int(math.log(epoch + 1, 1.15)), batch_size, initial_T, W, B)
            if epoch % 50 == 0:
                self.save_model(path=path)
