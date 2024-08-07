from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np

def transitions_probabilities(mdp: MDP, from_state: np.ndarray, to_state: np.ndarray, action: str) -> np.ndarray:
    if from_state in mdp.terminal_states:
        return 0
    if np.abs(from_state[0] - to_state[0]) > 1:
        return 0
    if np.abs(from_state[1] - to_state[1]) > 1:
        return 0
    
    actions_idx_to_state = [i for i, action in enumerate(mdp.actions) if mdp.step(from_state, action) == to_state]
    
    probability = 0
    for i in actions_idx_to_state:
        probability += mdp.transition_function[Action(action)][i]
    return probability

def max_utility_action(mdp: MDP, state: np.ndarray, U: np.ndarray) -> Tuple[float, str]:
    if state in mdp.terminal_states:
        return U[state[0]][state[1]], 0
    if mdp.board[state[0]][state[1]] == "WALL":
        return None, "WALL"
    
    max = -np.inf
    a = 0
    
    for action in mdp.actions:
        U_temp = 0
        U_temp += calc_util(mdp, state, U, action)
        
        if (U_temp >= max):
            max = U_temp
            a = action
            
    return max, a.value

def calc_util(mdp: MDP, state: np.ndarray, U: np.ndarray, action: str) -> float:
    if state in mdp.terminal_states:
        return U[state[0]][state[1]]
    if mdp.board[state[0]][state[1]] == "WALL":
        return None
    
    U_out = 0
    
    for i, prob_action in enumerate(mdp.actions):
            next_state = mdp.step(state, prob_action)
            P = mdp.transition_function[Action(action)][i]
            U_out += P * (U[next_state[0]][next_state[1]])
            
    return U_out

def bellman_update(mdp: MDP, state: np.ndarray, U: np.ndarray) -> Tuple[float, str]:
    if state in mdp.terminal_states:
        return U[state[0]][state[1]], 0
    if mdp.board[state[0]][state[1]] == "WALL":
        return None, "WALL"
    
    max_util, a = max_utility_action(mdp, state, U)
    U_next = float(mdp.board[state[0]][state[1]]) + mdp.gamma * max_util
    
    return U_next, a

def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float=10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #
    
    # TODO:
    # ====== YOUR CODE: ======
    U = U_init.copy()
    delta = np.inf
    
    while True:
        delta = 0
        
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                if mdp.board[row][col] == 'WALL':
                    U[row][col] = None
                    continue
                if (row, col) in mdp.terminal_states:
                    U[row][col] = float(mdp.board[row][col])
                    continue
                
                temp = U[row][col]
                U[row][col], _ = bellman_update(mdp, (row,col), U) 
                delta = max(delta, np.abs(U[row][col] - temp))
                    
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
    
    
    # ========================
    return U


def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    
    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 
    policy = np.empty_like(mdp.board, dtype='U5')
    
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            _, a = bellman_update(mdp, (row, col), U)
            policy[row][col] = a
    # ========================
    return policy


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # TODO:
    # ====== YOUR CODE: ======
    U = np.zeros_like(policy, dtype=float)
    U[policy == "WALL"] = None
    
    states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']
    
    rewards = np.array([mdp.board[state[0]][state[1]] for state in states], dtype=float)
    trans = [[transitions_probabilities(mdp, from_state, to_state, policy[from_state[0]][from_state[1]]) for to_state in states] for from_state in states]
    trans = np.array(trans)
    utils = np.linalg.inv((np.eye(len(states)) - mdp.gamma * trans)) @ rewards
    
    for state, u in zip(states, utils):
        U[state[0]][state[1]] = u
        
    return U
    # ========================


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:

    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    unchanged = False
    optimal_policy = policy_init.copy()
    
    while not unchanged:
        U = policy_evaluation(mdp, optimal_policy)
        unchanged = True
        
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                if mdp.board[row][col] == 'WALL':
                    continue
                
                max_util, max_action = max_utility_action(mdp, (row, col), U)
                policy_util = calc_util(mdp, (row, col), U, optimal_policy[row][col])
                
                if (max_util > policy_util):
                    optimal_policy[row][col] = max_action
                    unchanged = False
        
    # ========================
    return optimal_policy



def adp_algorithm(
    sim: Simulator, 
    num_episodes: int,
    num_rows: int = 3, 
    num_cols: int = 4, 
    actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT] 
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid, 
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.
    
    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions. 
    """
    

    transition_probs = None
    reward_matrix = None
    # TODO
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
    return reward_matrix, transition_probs 
