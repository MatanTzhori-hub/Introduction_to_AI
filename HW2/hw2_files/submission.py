from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv
from WarehouseEnv import manhattan_distance as man
import random

import numpy as np
from func_timeout import FunctionTimedOut
import time

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    
    def closest_fueling_station(rob_pos):
        fs0 = env.charge_stations[0].position
        fs1 = env.charge_stations[1].position
        return min([man(rob_pos, fs0), man(rob_pos, fs1)])
    
    def closest_package(rob_pos):
        on_board = [pack for pack in env.packages if pack.on_board]
        if (len(on_board) == 1):
            return man(on_board[0].position, rob_pos)
        return min([man(rob_pos, on_board[0].position), man(rob_pos, on_board[1].position)])

    def pack_reward(package):
        return 2*man(package.destination, package.position)
    
    if robot.battery <= closest_fueling_station(robot.position) and 0 < robot.credit <= other_robot.credit:
        return robot.battery - robot.credit -closest_fueling_station(robot.position)
    elif robot.package != None:
        return 3*robot.credit + pack_reward(robot.package) - man(robot.position, robot.package.destination)
    return 3*robot.credit - closest_package(robot.position)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def __init__(self) -> None:
        super().__init__()
        self.cur_max = -np.inf
        self.op = 'park'
        self.epsilon = 0.01

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        try:
            start_time = time.time()
            self.RB_Minimax(env.clone(), agent_id, time_limit - self.epsilon, start_time)
        except FunctionTimedOut:
            return self.op
    
    def RB_Minimax(self, env: WarehouseEnv, robot: int, time_limit: float, starting_time: float):
    
        def RB_Minimax_recursive(state: WarehouseEnv, robot_id: int, cur_depth: int, depth: int, time_limit: float, starting_time: float):
            
            if time.time() - starting_time > time_limit:
                raise FunctionTimedOut
            if state.done():
                return state.robots[robot_id].credit - state.robots[(robot_id+1)%2].credit, 'park'
            elif cur_depth == depth:
                return smart_heuristic(state, robot_id), 'park'
            
            turn_robot = (robot_id + cur_depth)%2
            
            operators = state.get_legal_operators(turn_robot)
            children = [state.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(turn_robot, op)
                
            if robot_id == turn_robot:
                cur_max = -np.inf
                
                out_vals = []
                for child_state in children:
                    val, op = RB_Minimax_recursive(child_state, robot_id, cur_depth+1, depth, time_limit, starting_time)
                    out_vals.append(val)
                
                out_vals.append(cur_max)
                cur_max = max(out_vals)
                op = operators[np.argmax(out_vals)]
                return cur_max, op
                
            else: #robot_id != turn_robot
                cur_min = np.inf
                
                out_vals = []
                for child_state in children:
                    val, op = RB_Minimax_recursive(child_state, robot_id, cur_depth+1, depth, time_limit, starting_time)
                    out_vals.append(val)
                
                out_vals.append(cur_min)
                cur_min = min(out_vals)
                op = operators[np.argmin(out_vals)]
                return cur_min, op
        
        D = 4
        self.op = env.get_legal_operators(robot)[0]
        self.cur_max = -np.inf
        while True:
            out_val, out_op = RB_Minimax_recursive(env.clone(), robot, 0, D, starting_time, time_limit)
            self.cur_max , self.op = (out_val, out_op) if out_val > self.cur_max else (self.cur_max , self.op)
            # raise FunctionTimedOut
            D = D + 1


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def __init__(self) -> None:
        super().__init__()
        self.cur_max = -np.inf
        self.op = 'park'
        self.epsilon = 0.01
        
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        try:
            start_time = time.time()
            self.RB_AlphaBeta_Minimax(env.clone(), agent_id, time_limit - self.epsilon, start_time)
        except FunctionTimedOut:
            return self.op
        
    def RB_AlphaBeta_Minimax(self, env: WarehouseEnv, robot: int, time_limit: float, starting_time: float):
    
        def RB_AlphaBeta_Minimax_recursive(state: WarehouseEnv, robot_id: int, cur_depth: int, depth: int, alpha: float, beta: float, time_limit: float, starting_time: float):
            
            if time.time() - starting_time > time_limit:
                raise FunctionTimedOut
            if state.done():
                return state.robots[robot_id].credit - state.robots[(robot_id+1)%2].credit, 'park'
            elif cur_depth == depth:
                return smart_heuristic(state, robot_id), 'park'
            
            turn_robot = (robot_id + cur_depth)%2
            
            operators = state.get_legal_operators(turn_robot)
            children = [state.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(turn_robot, op)
                
            if robot_id == turn_robot:
                cur_max = -np.inf
                
                out_vals = []
                for child_state in children:
                    val, op = RB_AlphaBeta_Minimax_recursive(child_state, robot_id, cur_depth+1, depth, alpha, beta, time_limit, starting_time)
                    out_vals.append(val)
                    alpha = max(out_vals+[alpha])
                    if max(out_vals) >= beta:
                        return np.inf, op  # op is not really relevant
                
                out_vals.append(cur_max)
                cur_max = max(out_vals)
                op = operators[np.argmax(out_vals)]
                
                return cur_max, op
                
            else: #robot_id != turn_robot
                cur_min = np.inf
                
                out_vals = []
                for child_state in children:
                    val, op = RB_AlphaBeta_Minimax_recursive(child_state, robot_id, cur_depth+1, depth, alpha, beta, time_limit, starting_time)
                    out_vals.append(val)
                    beta = min(out_vals+[beta])
                    if min(out_vals) <= alpha:
                        return -np.inf, op  # op is not really relevant
                
                out_vals.append(cur_min)
                cur_min = min(out_vals)
                op = operators[np.argmin(out_vals)]
                return cur_min, op
        
        D = 4
        self.op = env.get_legal_operators(robot)[0]
        self.cur_max = -np.inf
        alpha = -np.inf
        beta = np.inf
        while True:
            out_val, out_op = RB_AlphaBeta_Minimax_recursive(env.clone(), robot, 0, D, alpha, beta, starting_time, time_limit)
            self.cur_max , self.op = (out_val, out_op) if out_val > self.cur_max else (self.cur_max , self.op)
            # raise FunctionTimedOut
            D = D + 1


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def __init__(self) -> None:
        super().__init__()
        self.cur_max = -np.inf
        self.op = 'park'
        self.epsilon = 0.01
        
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        try:
            start_time = time.time()
            self.RB_Expictimax(env.clone(), agent_id, time_limit - self.epsilon, start_time)
        except FunctionTimedOut:
            return self.op
        
    def RB_Expictimax(self, env: WarehouseEnv, robot: int, time_limit: float, starting_time: float):

        def get_proba(ops):
            sum = 0
            better_moves = ["pick up", "move east"]
            proba = []
            for op in ops:
                sum = sum + 2 if op in better_moves else sum + 1
            for op in ops:
                proba.append((1 + 1 * (op in better_moves))/sum)
            
            return proba
            
        
        def RB_Expictimax_recursive(state: WarehouseEnv, robot_id: int, cur_depth: int, depth: int, time_limit: float, starting_time: float):
            
            if time.time() - starting_time > time_limit:
                raise FunctionTimedOut
            if state.done():
                return state.robots[robot_id].credit - state.robots[(robot_id+1)%2].credit, 'park'
            elif cur_depth == depth:
                return smart_heuristic(state, robot_id), 'park'
            
            turn_robot = (robot_id + cur_depth)%2
            
            operators = state.get_legal_operators(turn_robot)
            children = [state.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(turn_robot, op)
                
            if robot_id == turn_robot:
                cur_max = -np.inf
                
                out_vals = []
                for child_state in children:
                    val, op = RB_Expictimax_recursive(child_state, robot_id, cur_depth+1, depth, time_limit, starting_time)
                    out_vals.append(val)
                
                out_vals.append(cur_max)
                cur_max = max(out_vals)
                op = operators[np.argmax(out_vals)]
                
                return cur_max, op
                
            else: #robot_id != turn_robot
                
                propabilities = get_proba(operators)
                out_vals = []
                for i, child_state in enumerate(children):
                    val, op = RB_Expictimax_recursive(child_state, robot_id, cur_depth+1, depth, time_limit, starting_time)
                    out_vals.append(propabilities[i] * val)
                
                return sum(out_vals), op
        
        D = 4
        self.op = env.get_legal_operators(robot)[0]
        self.cur_max = -np.inf
        while True:
            out_val, out_op = RB_Expictimax_recursive(env.clone(), robot, 0, D, starting_time, time_limit)
            self.cur_max , self.op = (out_val, out_op) if out_val > self.cur_max else (self.cur_max , self.op)
            # raise FunctionTimedOut
            D = D + 1


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
    