from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv
from WarehouseEnv import manhattan_distance as man
import random


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)

    def pack_reward(package):
        return man(package.destination, package.position)*2

    def FuelHuer(robot_id, dest):
        fs0 = env.charge_stations[0].position
        fs1 = env.charge_stations[1].position
        robot = env.get_robot(robot_id)
        pos = robot.position
        if man(pos, fs0) + man(dest, fs0) > man(pos, fs1) + man(dest, fs1):
            return -man(pos, fs0)
        return -man(pos, fs1)
    
    def DistHuer(robot_id, dest):
        robot = env.get_robot(robot_id)
        pos = robot.position
        if robot.battery > man(dest, pos):
            return -man(dest, pos)
        return FuelHuer(robot_id, dest)
    
    def PackHuer(robot_id):
        other_robot = env.get_robot((robot_id + 1) % 2)
        on_board = [pack for pack in env.packages if pack.on_board]
        if (other_robot.package != None):
            return DistHuer(robot_id, on_board[0].position)
        else:
            reward_on_board = [man(pack.position, pack.destination)*2 for pack in on_board]
            if reward_on_board[0] > reward_on_board[1]:
                higher_pack = 0
            else:
                higher_pack = 1
            if DistHuer(robot_id, on_board[higher_pack].position) > DistHuer((robot_id+1)%2, on_board[higher_pack].position):
                return DistHuer(robot_id, on_board[higher_pack].position)
            return DistHuer(robot_id, on_board[(higher_pack+1)%2].position)
    
    if robot.package != None:
        return DistHuer(robot_id, robot.package.destination) + robot.credit - other_robot.credit + robot.battery + pack_reward(robot.package)
    return PackHuer(robot_id) + 2*robot.credit - other_robot.credit + robot.battery
    

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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