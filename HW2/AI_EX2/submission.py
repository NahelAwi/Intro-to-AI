from tkinter import NO
from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    tValue = max(robot.battery - manhattan_distance(robot.position, env.packages[0].position)
                 - manhattan_distance(env.packages[0].position, env.packages[0].destination ),
                 robot.battery - manhattan_distance(robot.position, env.packages[1].position)
                 - manhattan_distance(env.packages[1].position, env.packages[1].destination ))
    
    if(tValue >= 0):
        fvalue = tValue
    else:
        fvalue = max(-manhattan_distance(robot.position,env.charge_stations[0].position),
                     -manhattan_distance(robot.position,env.charge_stations[1].position))
    
    if(robot.package != None):
        return fvalue + 4 - 0.5*manhattan_distance(robot.position, robot.package.destination)
    else:
        return fvalue + 2*robot.credit
    

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operation,value = self.TRB_minimax(self,env,agent_id= agent_id,time_limit=time_limit,my_id=agent_id)
        if(operation == None):
                return AgentGreedyImproved().run_step(env,agent_id,time_limit)
        return operation

    def TRB_minimax(self, env: WarehouseEnv, agent_id , time_limit, my_id):
        begin = time.time()
        if(env.done() or time_limit <= 0.1):
            return None,smart_heuristic(env,agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if(agent_id == my_id):
            curr_max_value = -float("inf")
            curr_max_operation = None

            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                operation,value = self.TRB_minimax(child, env, (agent_id+1)%2, time_limit - (end-begin), my_id)
                if(value > curr_max_value):
                    curr_max_operation = operation
                    curr_max_value = value
            return curr_max_operation,curr_max_value
        else:
            curr_min_value = float("inf")
            curr_min_operation = None

            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                operation,value = self.TRB_minimax(child, env, (agent_id+1)%2, time_limit - (end-begin), my_id)
                if(value < curr_min_value):
                    curr_min_operation = operation
                    curr_min_value = value
            return curr_min_operation,curr_min_value
        


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operation,value = self.TRB_AlphaBeta_minimax(self,env, agent_id= agent_id, time_limit=time_limit, my_id=agent_id , Alpha=-float("inf"), Beta=float("inf"))
        if(operation == None):
                return AgentGreedyImproved().run_step(env,agent_id,time_limit)
        return operation

    def TRB_AlphaBeta_minimax(self, env: WarehouseEnv, agent_id , time_limit, my_id, Alpha, Beta):
        begin = time.time()
        if(env.done() or time_limit <= 0.1):
            return None,smart_heuristic(env,agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if(agent_id == my_id):
            curr_max_value = -float("inf")
            curr_max_operation = None

            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                operation,value = self.TRB_AlphaBeta_minimax(child, env, (agent_id+1)%2, time_limit - (end-begin), my_id, Alpha, Beta)
                
                if(value > curr_max_value):
                    curr_max_operation = operation
                    curr_max_value = value
                
                Alpha = max(Alpha, curr_max_value)
                if( curr_max_value >= Beta):
                    return None, float("inf")
            return curr_max_operation,curr_max_value
        else:
            curr_min_value = float("inf")
            curr_min_operation = None

            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                operation,value = self.TRB_AlphaBeta_minimax(child, env, (agent_id+1)%2, time_limit - (end-begin), my_id, Alpha, Beta)

                if(value < curr_min_value):
                    curr_min_operation = operation
                    curr_min_value = value

                Beta = min(Beta, curr_min_value)
                if( curr_min_value <= Alpha):
                    return None, -float("inf")
            return curr_min_operation,curr_min_value


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