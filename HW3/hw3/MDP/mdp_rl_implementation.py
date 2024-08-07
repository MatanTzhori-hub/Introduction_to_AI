from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np

def bellman_update(mdp: MDP, state: np.ndarray, U) -> Tuple[float, str]:
    if state in mdp.terminal_states:
        return None, 0
    if mdp.board[state[0]][state[1]] == "WALL":
        return None, "WALL"
    
    max_util, a = max_utility_action(mdp, state, U)
    U_next = float(mdp.board[state[0]][state[1]]) + mdp.gamma * max_util
    
    return U_next, a

def max_utility_action(mdp: MDP, state: np.ndarray, U) -> str:
    max = -np.inf
    a = 0
    
    for action in mdp.actions:
        U_temp = 0
        for i, prob_action in enumerate(mdp.actions):
            next_state = mdp.step(state, prob_action)
            P = mdp.transition_function[action][i]
            U_temp += P * (U[next_state[0]][next_state[1]])
        
        if (U_temp >= max):
            max = U_temp
            a = action
            
    return max, a

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
    policy = np.empty_like(mdp.board, dtype=Action)
    
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
    U = np.zeros_like(mdp.board, dtype=float)
    
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] == 'WALL':
                U[row][col] = None
                continue
            if (row, col) in mdp.terminal_states:
                U[row][col] = float(mdp.board[row][col])
                continue
            
            U_temp = 0
            for i, prob_action in enumerate(mdp.actions):
                next_state = mdp.step((row, col), prob_action)
                action = policy[row][col]
                P = mdp.transition_function[action][i]
                U_temp += P * (U[next_state[0]][next_state[1]])
                
            U[row][col] = float(mdp.board[row][col]) + mdp.gamma * U_temp
    
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
                max_util, max_action = max_utility_action(mdp, (row, col), U)
                
                if (max_util > U[row][col]):
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
