import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self, state: Tuple, parent : 'Node' = None, g_value: float = 0, f_value: float = 0) -> None:
        self.state = state #STATE _ (Position,Ball1,Ball2)
        self.parent = parent
        self.actionsList = []
        self.totalCost = 0
        self.g = g_value
        self.f = f_value    
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash((self.state, self.f))
        
    def expand(self, env: DragonBallEnv) -> list['Node']:
        nA = 4
        successors = []
        # if(self.state[0] == env.ncol*env.nrow - 1):# if G reached Stop Searching
        #     return successors
        for a in range(nA):
            env.set_state_2(self.state)
            if(env.succ(self.state)[a] == (None,None,None)): #WE DONT EXPAND HOLES
                continue
            state , cost , termiated = env.step(a)
            NewNode = Node(state,self)
            NewNode.totalCost = self.totalCost + cost
            NewNode.actionsList.extend(self.actionsList)
            NewNode.actionsList.append(a)
            
            successors.append(NewNode)

        return successors


def ManHattanDistance(n1:tuple, n2:tuple, numCol:int)->int:#TODO Check in case of wrong input
    dy = abs(n1[0]//numCol - n2[0]//numCol)
    dx = abs(n1[0]%numCol - n2[0]%numCol)
    return dx + dy
        
def hSMAP(n:'Node', env: DragonBallEnv):
    l = env.get_goal_states()
    d1 = ManHattanDistance(n.state,env.d1,env.ncol)
    d2 = ManHattanDistance(n.state,env.d2,env.ncol)
    h = env.ncol*env.nrow - 1 #####################TODO Here generates a problem with nodes expanded needs checking
    #if(not env.collected_dragon_balls[0] and not env.collected_dragon_balls[1]):
    if(not n.state[1] and not n.state[2]):
        h = min(d1,d2)
    else:
        #if(not env.collected_dragon_balls[0]):
        if(not n.state[1]):
            h = d1
        #if(not env.collected_dragon_balls[1]):
        if(not n.state[2]):
            h = d2
    ##############################################################################
    for G in l:
        h = min(ManHattanDistance(n.state,G,env.ncol),h)
    return h


class BFSAgent():
    def __init__(self) -> None:
        self.Open = []
        self.Close = []
        self.nodesExpanded = 0

    
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        n = Node(env.get_initial_state())
        if(env.is_final_state(n.state)):
            return ([],0,1) #OR 0 for first node
        self.Open.clear()
        self.Close.clear()
        self.Open.append(n)
        while len(self.Open) != 0:
            n = self.Open.pop(0)
            self.Close.append(n.state)
            self.nodesExpanded += 1  
            for child in n.expand(env):
                if (child.state not in self.Close) and (child not in self.Open):
                    if env.is_final_state(child.state):
                        return (child.actionsList , child.totalCost ,self.nodesExpanded)
                    self.Open.append(child)
        return ([],-1,self.nodesExpanded)#Failure


class WeightedAStarAgent():
    def __init__(self) -> None:
        self.Open = heapdict.heapdict()
        self.Close = []
        self.nodesExpanded = 0

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        t = Node(env.get_initial_state())
        n = Node(env.get_initial_state(), None, 0,h_weight*hSMAP(t,env))
        if(env.is_final_state(n.state)):
            return ([],0,1)
        self.Open[n] = n.f
        while len(self.Open)!=0:
            Tmp = self.Open.popitem()
            n = Tmp[0]
            self.Close.append(n)
            self.nodesExpanded += 1
            if env.is_final_state(n.state):
                return (n.actionsList , n.totalCost ,self.nodesExpanded)
            for child in n.expand(env):
                newG = child.totalCost # we already had a counter for the cost so far on each child
                newF = (1-h_weight)*newG + h_weight*hSMAP(child,env)
                child.f = newF
                child.g = newG
                if (child not in self.Close) and (child not in self.Open):
                    self.Open[child] = child.f
                else:
                    if(child in self.Open):
                        if(child.f < self.Open[child]):
                            self.Open.pop(child)
                            self.Open[child] = child.f
                    else:
                        if(child in self.Close):
                            index = self.Close.index(child)
                            currChild = self.Close.pop(index)
                            if(child.f < currChild.f):
                                self.Open[child] = child.f
                            else:
                                self.Close.append(currChild)
                    
        return ([],-1,self.nodesExpanded)#Failure





class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError