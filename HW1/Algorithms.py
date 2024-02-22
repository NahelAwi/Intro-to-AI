import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self, state: Tuple, parent : 'Node' = None) -> None:
        self.state = state #STATE _ (Position,Ball1,Ball2)
        self.parent = parent
        #self.successors = []
        self.actionsList = []
        self.totalCost = 0
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
        
        
    def expand(self, env: DragonBallEnv) -> list['Node']:
        nA = 4
        successors = []
        for a in range(nA):
            env.set_state_2(self.state)
            if(env.succ(self.state)[a] == (None,None,None)):
                continue
            state , cost , termiated = env.step(a)
            # if termiated and (state[1] == False or state[2] == False):
            #     continue
            NewNode = Node(state,self)
            NewNode.totalCost = self.totalCost + cost
            NewNode.actionsList.extend(self.actionsList)
            NewNode.actionsList.append(a)
            
            successors.append(NewNode)

        return successors
        

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
        self.nodesExpanded += 1
        while len(self.Open) != 0:
            n = self.Open.pop(0)
            self.Close.append(n.state)  
            for child in n.expand(env):
                if (child.state not in self.Close) and (child not in self.Open):
                    if env.is_final_state(child.state):
                        return (child.actionsList , child.totalCost ,self.nodesExpanded)
                    self.nodesExpanded += 1
                    self.Open.append(child)
        return ([],-1,self.nodesExpanded)#Failure


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError