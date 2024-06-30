from collections import deque

import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict


class Node:
    def __init__(self, state, cost: int = 0, action: int = 0, is_terminated: bool = False, parent = None) -> None:
        self.state = state
        self.cost = cost
        self.action = action
        self.is_terminated = is_terminated
        self.parent = parent
        self.g = 0

class Agent:
    def __init__(self) -> None:
        self.env: CampusEnv = None
        self.OPEN = None
        self.state_to_node: dict = dict()
        self.CLOSE: set = set()
        self.expanded: int = 0

    def expand(self, node: Node):
        self.expanded = self.expanded + 1
        
        for action, (state, cost, terminated) in self.env.succ(node.state).items():
            if state is None:
                continue 
            if state not in self.state_to_node.keys():
                child = Node(state, cost, action, terminated, node)
                self.state_to_node[state] = child
                yield child
            else:
                yield self.state_to_node[state]

    def solution(self, node: Node) -> Tuple[List[int], float, int]:
        total_cost = 0
        actions = []

        while node.parent is not None:
            total_cost += node.cost
            actions.insert(0, node.action)
            node = node.parent

        return (actions, total_cost, self.expanded)
    
    def __reset_agent__(self, env: CampusEnv) -> None:
        self.env: CampusEnv = env
        self.env.reset()
        self.OPEN = None
        self.state_to_node: dict = dict()
        self.CLOSE: set = set()
        self.expanded: int = 0
        

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 


class DFSGAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        
    def __reset_agent__(self, env: CampusEnv) -> None:
        super().__reset_agent__(env)
        self.OPEN: deque = deque()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.__reset_agent__(env)

        first_node = Node(env.get_initial_state())
        self.state_to_node[first_node.state] = first_node
        self.OPEN.append(first_node)
        return self.recursive_DFS_G()


    def recursive_DFS_G(self):
        cur_node = self.OPEN.pop()
        self.CLOSE.add(cur_node.state)

        if self.env.is_final_state(cur_node.state):
            return self.solution(cur_node)

        for child in self.expand(cur_node):
            if child.state not in self.CLOSE and child.state not in [s.state for s in self.OPEN]:
                self.OPEN.append(child)
                result = self.recursive_DFS_G()
                if result is not None:
                    return result
        
        return None
        


class UCSAgent(Agent):
  
    def __init__(self) -> None:
        super().__init__()
    
    def __reset_agent__(self, env: CampusEnv) -> None:
        super().__reset_agent__(env)
        self.OPEN: heapdict = heapdict.heapdict()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.__reset_agent__(env)

        first_node = Node(env.get_initial_state())
        self.state_to_node[first_node.state] = first_node
        self.OPEN[first_node.state] = (first_node.g, first_node.state)
        
        while self.OPEN:
            cur_state, _ = self.OPEN.popitem()
            cur_node = self.state_to_node[cur_state]
            self.CLOSE.add(cur_node.state)
            
            if self.env.is_final_state(cur_node.state):
                return self.solution(cur_node)
            
            for child in self.expand(cur_node):
                child_state = child.state
                new_g = cur_node.g + child.cost
                
                # if self.env.is_final_state(child_state):
                #     return self.solution(child)
                
                if child_state not in self.CLOSE and child_state not in self.OPEN.keys():
                    child.g = new_g
                    self.OPEN[child_state] = (child.g, child.state)
                elif child_state in self.OPEN.keys() and self.OPEN[child_state][0] > new_g:
                    child.g = new_g
                    child.parent = cur_node
                    self.OPEN[child_state] = (child.g, child.state)
                    
        return None
            



class WeightedAStarAgent(Agent):
    
    def __init__(self):
        super().__init__()
    
    def __reset_agent__(self, env: CampusEnv) -> None:
        super().__reset_agent__(env)
        self.OPEN: heapdict = heapdict.heapdict()

    def h_campus(self, state: int) -> int:
        goal_states = self.env.get_goal_states()
        manhatan_dist = np.array([np.sum(np.abs(np.array(self.env.to_row_col(state)) - np.array(self.env.to_row_col(goal)))) for goal in goal_states])
        return np.min([np.min(manhatan_dist), 100]).item()
    
    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        self.__reset_agent__(env)

        first_node = Node(env.get_initial_state())
        self.state_to_node[first_node.state] = first_node
        f = (1 - h_weight) * first_node.g + h_weight * self.h_campus(first_node.state)
        self.OPEN[first_node.state] = (f, first_node.state)
        
        while self.OPEN:
            cur_state, _ = self.OPEN.popitem()
            cur_node = self.state_to_node[cur_state]
            self.CLOSE.add(cur_node.state)
            
            if self.env.is_final_state(cur_node.state):
                return self.solution(cur_node)
            
            for child in self.expand(cur_node):
                child_state = child.state
                new_g = cur_node.g + child.cost
                new_f = (1 - h_weight) * new_g + h_weight * self.h_campus(child_state)
                
                # if self.env.is_final_state(child_state):
                #     return self.solution(child)
                
                if child_state not in self.CLOSE and child_state not in self.OPEN.keys():
                    child.g = new_g
                    self.OPEN[child_state] = (new_f, child.state)
                elif child_state in self.OPEN.keys() and self.OPEN[child_state][0] > new_g:
                    child.g = new_g
                    child.parent = cur_node
                    self.OPEN[child_state] = (new_f, child.state)
                    
        return None 



class AStarAgent(WeightedAStarAgent):
    
    def __init__(self):
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return super().search(env, 0.5)

