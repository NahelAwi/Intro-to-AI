from copy import deepcopy
from math import gamma, sumprod
from sre_constants import MAX_UNTIL
import numpy as np
from sympy import false


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ====== 
    U_t = deepcopy(U_init)
    while True:
        U = deepcopy(U_t)
        delta = 0

        for state, value in np.ndenumerate(mdp.board):
            if(value == 'WALL'):
                U_t[state[0]][state[1]] = None
                continue
            if(state in mdp.terminal_states):
                U_t[state[0]][state[1]] = float(value)
                continue

            max_TU = -float('inf')
            for a in mdp.actions.keys():

                tmp = 0
                for b in range(4):  #[(up,0),(down,1),(right,2),(left,3)] #TODO Check if need only sum different states i.e. not the original
                    new_state = mdp.step(state, list(mdp.actions.keys())[b])
                    tmp += mdp.transition_function[a][b] * U[new_state[0]][new_state[1]]

                if( tmp > max_TU):
                    max_TU = tmp

            U_t[state[0]][state[1]] = float(value) + mdp.gamma*max_TU
            if(abs(U_t[state[0]][state[1]] - U[state[0]][state[1]]) > delta):
                delta = abs(U_t[state[0]][state[1]] - U[state[0]][state[1]])


        if(delta <= epsilon*(1-mdp.gamma)/mdp.gamma):
            break
    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======#TODO check if need calculate all the utility and not just the expectancy
    rows = mdp.num_row
    cols = mdp.num_col

    Policy = [['UP' for _ in range(cols)] for _ in range(rows)]
    for state, value in np.ndenumerate(U):
        if(value == None or state in mdp.terminal_states):
            Policy[state[0]][state[1]] = None
            continue
        
        max_action = None
        max_E = -float('inf')

        for a in mdp.actions.keys():
            expected_utility = 0
            for b in range(4):  # b => [(up,0),(down,1),(right,2),(left,3)] #TODO Check if need only sum different states i.e. not the original
                new_state = mdp.step(state, list(mdp.actions.keys())[b])
                expected_utility += mdp.transition_function[a][b] * U[new_state[0]][new_state[1]]

            if(expected_utility > max_E):
                max_E = expected_utility
                max_action = a
        
        Policy[state[0]][state[1]] = max_action

    return Policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    rows = mdp.num_row
    cols = mdp.num_col

    U = [[0 for _ in range(cols)] for _ in range(rows)] #TODO check [0]
    for _ in range(rows*cols):#TODO check stopping condition
        for state, value in np.ndenumerate(mdp.board):
            if(value == 'WALL'):
                U[state[0]][state[1]] = None
                continue
            if(state in mdp.terminal_states):
                U[state[0]][state[1]] = float(value)
                continue

            sumProb = 0
            for b in range(4):  # b => [(up,0),(down,1),(right,2),(left,3)] #TODO Check if need only sum different states i.e. not the original
                new_state = mdp.step(state, list(mdp.actions.keys())[b])
                sumProb += mdp.transition_function[ policy[state[0]][state[1]] ][b] * U[new_state[0]][new_state[1]]

            U[state[0]][state[1]] = float(value) + mdp.gamma*sumProb

    return U
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    pi = deepcopy(policy_init)
    while True:
        U = policy_evaluation(mdp=mdp, policy=pi)
        unchanged = True

        for state, value in np.ndenumerate(mdp.board):
            if(value == 'WALL' or state in mdp.terminal_states):
                pi[state[0]][state[1]] = None
                continue

            max_action = None
            max_TU = -float('inf')
            for a in mdp.actions.keys():
                tmp = 0
                for b in range(4):  #[(up,0),(down,1),(right,2),(left,3)] #TODO Check if need only sum different states i.e. not the original
                    new_state = mdp.step(state, list(mdp.actions.keys())[b])
                    tmp += mdp.transition_function[a][b] * U[new_state[0]][new_state[1]]
                if( tmp > max_TU):
                    max_TU = tmp
                    max_action = a

            sumProb = 0
            for b in range(4):  # b => [(up,0),(down,1),(right,2),(left,3)] #TODO Check if need only sum different states i.e. not the original
                new_state = mdp.step(state, list(mdp.actions.keys())[b])
                sumProb += mdp.transition_function[ pi[state[0]][state[1]] ][b] * U[new_state[0]][new_state[1]]
            
            if( max_TU > sumProb):
                pi[state[0]][state[1]] = max_action
                unchanged = False

        if unchanged:
            break
    return pi
    # ========================



"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
