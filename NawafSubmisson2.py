from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    current_robot = env.get_robot(robot_id)
    current_package = current_robot.package
    manhattan_to_first_package = manhattan_distance(current_robot.position, env.packages[0].position)
    manhattan_to_second_package = manhattan_distance(current_robot.position, env.packages[1].position)
    max_remaining_battery_to_package = current_robot.battery - min(manhattan_to_first_package, manhattan_to_second_package)
    if (current_package != None):
        max_remaining_battery_to_package = current_robot.battery - manhattan_distance(current_robot.position, current_package.destination)
    h = max_remaining_battery_to_package
    if current_package:
        h += 5
    h += 6*current_robot.credit
    return h

class AgentGreedyImproved(AgentGreedy):
    def __init__(self):
        self.old_package = None
        
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def __init__(self):
        self.counter = 0

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        value, op = self.get_mini_max_value_and_op(env, agent_id, time_limit, agent_id)
        if op is None:
            return AgentGreedyImproved().run_step(env, agent_id, time_limit)
        return op

    def get_mini_max_value_and_op(self, env: WarehouseEnv, agent_id, time_limit, my_id):
        start = time.time()
        if env.done() or time_limit <= 0.1:  # not 0 : it takes time to return from all the recursive calls.
            return (smart_heuristic(env, my_id), None)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        is_my_turn = (agent_id == my_id)
        if is_my_turn:
            curr_max_value = -float("inf")
            curr_max_op = None
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                v,_ = self.get_mini_max_value_and_op(child, (agent_id == 0), time_limit - (end - start), my_id)
                if v > curr_max_value:
                    curr_max_value = v
                    curr_max_op = op
            return (curr_max_value, curr_max_op)
        else:
            curr_min_value = float("inf")
            curr_min_op = None
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                v,_ = self.get_mini_max_value_and_op(child, (agent_id == 0), time_limit - (end - start), my_id)
                if v < curr_min_value:
                    curr_min_value = v
                    curr_min_op = op
            return (curr_min_value, curr_min_op)


class AgentAlphaBeta(Agent):

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        value, op = self.get_mini_max_value_and_op(env, agent_id, time_limit, agent_id, -float("inf"), float("inf"))
        if op is None:
            return AgentGreedyImproved().run_step(env, agent_id, time_limit)
        return op

    def get_mini_max_value_and_op(self, env: WarehouseEnv, agent_id, time_limit, my_id, alpha, beta):
        start = time.time()
        if env.done() or time_limit <= 0.1:    # not 0 : it takes time to return from all the recursive calls.
            return (smart_heuristic(env, my_id), None)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        is_my_turn = (agent_id == my_id)
        if is_my_turn:
            curr_max_value = -float("inf")
            curr_max_op = None
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                v,_ = self.get_mini_max_value_and_op(child, (agent_id == 0), time_limit - (end - start), my_id, alpha, beta)
                if v > curr_max_value:
                    curr_max_value = v
                    curr_max_op = op
                alpha = max(curr_max_value, alpha)
            if curr_max_value >= beta:
                return (float("inf"), None)
            return (curr_max_value, curr_max_op)
        else:
            curr_min_value = float("inf")
            curr_min_op = None
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                v,_ = self.get_mini_max_value_and_op(child, (agent_id == 0), time_limit - (end - start), my_id, alpha, beta)
                if v < curr_min_value:
                    curr_min_value = v
                    curr_min_op = op
                beta = min(curr_min_value, beta)
            if curr_min_value <= alpha:
                return (-float("inf"), None)
            return (curr_min_value, curr_min_op)


class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        value, op = self.get_expectimax_value_and_op(env, agent_id, time_limit, agent_id)
        if op is None:
            return AgentGreedyImproved().run_step(env, agent_id, time_limit)
        return op

    def get_expectimax_value_and_op(self, env: WarehouseEnv, agent_id, time_limit, my_id):
        start = time.time()
        if env.done() or time_limit <= 0.1:    # not 0 : it takes time to return from all the recursive calls.
            return (smart_heuristic(env, my_id), None)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        is_my_turn = (agent_id == my_id)

        if not is_my_turn:
            prob = []
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                if (child.get_charge_station_in(child.get_robot(agent_id).position)):
                    prob.append(2)
                else:
                    prob.append(1)
            assert(len(prob) == len(children))
            
            sum_prob = sum(prob)
            for i,p in enumerate(prob):
                prob[i] /= sum_prob

            children = [env.clone() for _ in operators]
            E = 0
            for child, op, p in zip(children, operators, prob):
                end = time.time()
                child.apply_operator(agent_id, op)
                v,_ = self.get_expectimax_value_and_op(child, (agent_id == 0), time_limit - (end - start), my_id)
                E += p * v
            return (E, None)

        else:
            curr_max_value = -float("inf")
            curr_max_op = None
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                end = time.time()
                v,_ = self.get_expectimax_value_and_op(child, (agent_id == 0), time_limit - (end - start), my_id)
                if v > curr_max_value:
                    curr_max_value = v
                    curr_max_op = op
            return (curr_max_value, curr_max_op)
        

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