from collections import deque

import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict


class Node:
    def __init__(self, state, cost: int = 0, action: int = 0, is_terminated: bool = False, parent: Node = None) -> None:
        self.state = state
        self.cost = cost
        self.action = action
        self.is_terminated = is_terminated
        self.parent = parent

class Agent:
    def __init__(self) -> None:
        self.env: CampusEnv = None
        self.OPEN = None
        self.CLOSE: set = set()
        self.expanded: int = 0

    def expand(self, node: Node):
        self.expanded = self.expanded + 1
        
        for action, (state, cost, terminated) in self.env.succ(node.state).items(): 
            child = Node(state, cost, action, terminated, node)
            yield child

    def solution(self, node: Node) -> Tuple[List[int], float, int]:
        if node is None or node.parent is None:
            return None

        total_cost = 0
        actions = []

        while node is not None:
            total_cost += node.cost
            actions = node.action + actions
            node = node.parent

        return (actions, total_cost, self.expanded)

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 


class DFSGAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.OPEN: deque = deque()

        first_node = Node(env.get_initial_state())
        self.OPEN.append(first_node)
        return self.recursive_DFS_G(first_node)


    def recursive_DFS_G(self):
        cur_node = self.OPEN.pop()
        self.CLOSE.add(cur_node.state)

        if self.env.is_final_state(cur_node.state):
            return self.solution(cur_node)

        for child in self.expand(cur_node):
            if child.state not in self.CLOSE and child not in self.OPEN:
                self.OPEN.append(child)
                result = recursive_DFS_G()
                if result is not None:
                    return result
        
        return None
        


class UCSAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class WeightedAStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError   



class AStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

