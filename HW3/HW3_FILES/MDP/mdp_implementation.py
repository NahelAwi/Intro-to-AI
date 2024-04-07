from copy import deepcopy
import numpy as np
from decimal import Decimal


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
                for b in range(4):  #[(up,0),(down,1),(right,2),(left,3)] 
                    new_state = mdp.step(state, list(mdp.actions.keys())[b])
                    tmp += mdp.transition_function[a][b] * U[new_state[0]][new_state[1]]

                if( tmp > max_TU):
                    max_TU = tmp

            U_t[state[0]][state[1]] = float(value) + mdp.gamma*max_TU
            if(abs(U_t[state[0]][state[1]] - U[state[0]][state[1]]) > delta):
                delta = abs(U_t[state[0]][state[1]] - U[state[0]][state[1]])


        if(delta <= epsilon*(1-mdp.gamma)/mdp.gamma):
            break
    
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if(U[i][j] != None):
                U[i][j] = round(U[i][j],2)

    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======#
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
            for b in range(4):  # b => [(up,0),(down,1),(right,2),(left,3)] 
                new_state = mdp.step(state, list(mdp.actions.keys())[b])
                expected_utility += mdp.transition_function[a][b] * U[new_state[0]][new_state[1]]
            expected_utility = float(value) + mdp.gamma*expected_utility
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

    U = [[0.0 for _ in range(cols)] for _ in range(rows)] 
    for _ in range(rows*cols):#TODO check stopping condition i.e. closed formula
        for state, value in np.ndenumerate(mdp.board):
            if(value == 'WALL'):
                U[state[0]][state[1]] = None
                continue
            if(state in mdp.terminal_states):
                U[state[0]][state[1]] = float(value)
                continue

            sumProb = 0
            for b in range(4):  # b => [(up,0),(down,1),(right,2),(left,3)] 
                new_state = mdp.step(state, list(mdp.actions.keys())[b])
                sumProb += mdp.transition_function[ policy[state[0]][state[1]] ][b] * U[new_state[0]][new_state[1]]

            U[state[0]][state[1]] = float(value) + mdp.gamma*sumProb
    for i in range(rows):
        for j in range(cols):
            if(U[i][j] != None):
                U[i][j] = round(U[i][j],2)

    return U

    # ========================

    ############################ Closed formula Solution TRY - not final##############################
    # rows = mdp.num_row
    # cols = mdp.num_col

    # n = rows*cols

    # P = np.zeros((n, n))

    # R = np.zeros(n) #deepcopy(mdp.board).flatten()

    # U = np.zeros(n)

    # for state, value in np.ndenumerate(mdp.board):
    #     if(value == 'WALL'):
    #         continue
    #     if(state in mdp.terminal_states):
    #         P[state[0]+state[1]][state[0]+state[1]] = 1
    #         R[state[0]+state[1]] = float(value)
    #         continue
    #     action = policy[state[0]][state[1]]
    #     for b in range(4):  # b => [(up,0),(down,1),(right,2),(left,3)] 
    #         new_state = mdp.step(state, list(mdp.actions.keys())[b])
    #         P[state[0]+state[1]][new_state[0]+new_state[1]] = mdp.transition_function[action][b]
    #         R[state[0]+state[1]] += mdp.transition_function[action][b]*float(value)

    # U = np.linalg.inv(np.eye(n) - mdp.gamma * P) @ R

    # U = U.reshape((rows,cols))

    # return U
    #############################################################################################################


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
                for b in range(4):  #[(up,0),(down,1),(right,2),(left,3)] 
                    new_state = mdp.step(state, list(mdp.actions.keys())[b])
                    tmp += mdp.transition_function[a][b] * U[new_state[0]][new_state[1]]
                if( tmp > max_TU):
                    max_TU = tmp
                    max_action = a

            sumProb = 0
            for b in range(4):  # b => [(up,0),(down,1),(right,2),(left,3)] 
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

#print is a flag for this function to print or not
#retP is a flag that detirmens if the function returns the policies or the number of them

def get_all_policies(mdp, U, epsilon=10**(-3), print = True, retP = False):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    rows = mdp.num_row
    cols = mdp.num_col

    Policies = [['' for _ in range(cols)] for _ in range(rows)]

    WordToArrow = {'UP':'↑','DOWN':'↓','RIGHT':'→','LEFT':'←'}
    numPolicies = 1
    for state, value in np.ndenumerate(mdp.board):
        if(value == 'WALL' or state in mdp.terminal_states):
            Policies[state[0]][state[1]] = None
            continue

        max_utility = -float('inf')
        best_actions = []
        for a in mdp.actions.keys():
            expected_utility = 0.0
            for b in range(4):  #[(up,0),(down,1),(right,2),(left,3)] 
                new_state = mdp.step(state, list(mdp.actions.keys())[b])
                expected_utility += mdp.transition_function[a][b] * float(U[new_state[0]][new_state[1]])
            expected_utility = float(value) + mdp.gamma*expected_utility
            expected_utility = round(expected_utility,2)
            if( expected_utility > max_utility):
                max_utility = expected_utility
                best_actions = [a]
            elif expected_utility == max_utility:
                best_actions.append(a)
        
        numPolicies *= len(best_actions)
        for action in best_actions:
            Policies[state[0]][state[1]] += WordToArrow[action]
    if(print):
        mdp.print_policy(Policies)
    if(retP):
        return Policies
    else:
        return numPolicies
    # ========================


def print_Reward_limits(left, right, betweenL=-5.0, betweenR=5.0):
    if(left == betweenL and right == betweenR):
        print('     ' + str(left) + ' <= R(s) <= ' + str(right))
        return
    if(left == betweenL):
        print('     ' + str(left) + ' <= R(s) < ' + str(right))
        return
    if(right == betweenR):
        print('     ' + str(left) + ' <= R(s) <= ' + str(right))
        return
    print('     ' + str(left) + ' <= R(s) < ' + str(right))



def get_policy_for_different_rewards(mdp, epsilon=10**(-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    rows = mdp.num_row
    cols = mdp.num_col
    # L = -5.0
    # R = -5.0
    L = Decimal('-5.0')
    R = Decimal('-5.0')
    policy = None
    thresholds = []
    while(R <= Decimal('5.0')):
        new_board = [[float(R) for _ in range(cols)] for _ in range(rows)]
        for state, value in np.ndenumerate(mdp.board):
            if(value == 'WALL' ):
                new_board[state[0]][state[1]] = 'WALL'
            elif (state in mdp.terminal_states):
                new_board[state[0]][state[1]] = value
        
        mdp.board = deepcopy(new_board)
        U = [[0.0 for _ in range(cols)] for _ in range(rows)]
        U = value_iteration(mdp,U,epsilon)

        if(R != -5.0):
            p_t = policy
            policy = get_all_policies(mdp,U, print=False, retP= True)

            if ((p_t != policy) or R == 5.0): #meaning got to the right limit [FINAL PRINT] - not np.allclose(p_t,policy,equal_nan=True)
                print_Reward_limits(L,R)# - Decimal('0.01')*(R !=5.0) )
                mdp.print_policy(p_t)
                L = R
                if(R!=5.0):
                    thresholds.append(float(R))# - Decimal('0.01')))  
        else:
            policy = get_all_policies(mdp,U, print=False, retP= True)

        R += Decimal('0.01')
    return thresholds
    # ========================
