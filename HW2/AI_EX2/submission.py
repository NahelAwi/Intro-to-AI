from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    tValue = max(robot.battery - manhattan_distance(robot.position, env.packages[0].position),
                 #- manhattan_distance(env.packages[0].position, env.packages[0].destination )
                 robot.battery - manhattan_distance(robot.position, env.packages[1].position)
                 #- manhattan_distance(env.packages[1].position, env.packages[1].destination )
                 )
    
    if(tValue >= 0):
        fvalue = tValue
    else:
        fvalue = max(-manhattan_distance(robot.position,env.charge_stations[0].position),
                     -manhattan_distance(robot.position,env.charge_stations[1].position)) + robot.battery*2
    sixpack = 0
    deliver = 0
    if (tValue >= 0 and robot.package != None):
        return (robot.battery-manhattan_distance(robot.position, robot.package.destination) + robot.credit)*5

    return fvalue + 7*robot.credit

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