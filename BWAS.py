import numpy as np
from queue import PriorityQueue
from topspin import TopSpinState

class TopSpinNode:
    def __init__(self, state, g, f=np.inf, p=None):
        self.s = state
        self.g = g
        self.f = f
        self.p = p

    def __eq__(self, other):
        return self.s.state == other.s.state

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif self.f == other.f:
            return self.g > other.g
        return True

    def __hash__(self):
        return hash(tuple(self.s.state))

    def is_goal(self):
        return self.s.is_goal()

    def get_state_as_list(self):
        return self.s.get_state_as_list()

    def get_neighbors(self):
        return self.s.get_neighbors()


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

def BWAS(start, W, B, heuristic_function, T, k=4):

    def path_to_goal(NUB):
      s, g, p, f = NUB
      path = [s.get_state_as_list()]
      while p:
        path.append(p[0].get_state_as_list())
        p = p[2]
      return path[::-1]



    open = PriorityQueue()
    closed = {}
    if type(start) is list:
        TSP = TopSpinState(start, k=k)
    else:
        TSP = start

    start_h = heuristic_function([TSP])[0]
    start_node = TopSpinNode(TSP, g=0, f=start_h*W)


    LB = 0
    UB = np.inf
    NUB = None


    open.put((start_h*W, start_node))
    # closed[tuple(TSP.get_state_as_list())] = 0

    num_expansions = 0

    while (num_expansions <= T) and (not open.empty()):
      generated = []
      batch_expansions = 0
      while (not open.empty()) and (num_expansions <= T) and (batch_expansions < B):
        _, n = open.get()
        s, g, p, f = n.s, n.g, n.p, n.f

        batch_expansions += 1
        num_expansions += 1

        if not generated:
          LB = max(f, LB)

        if s.is_goal():
          if UB > g:
            UB, NUB = g, (s, g, p, f)
          continue

        neighbors = s.get_neighbors()

        for s_tag, cost in neighbors:
          g_tag = g + cost
          s_tag = TopSpinState(s_tag, k=k)
          s_tag_state = tuple(s_tag.get_state_as_list())
          if (s_tag_state not in closed) or (g_tag < closed[s_tag_state]):
            closed[s_tag_state] = g_tag
            generated.append((s_tag, g_tag, (s, g, p, f)))

      if LB >= UB:
        return path_to_goal(NUB), num_expansions


      states = [val[0] for val in generated]
      if states:
        hs = heuristic_function(states)
      else:
        hs = []

      for i in range(len(hs)):
        s, g, p = generated[i]
        h = hs[i]
        state = TopSpinNode(s, g=g, f=g+h*W, p=p)

        open.put((g+h*W, state))

    return None, num_expansions
