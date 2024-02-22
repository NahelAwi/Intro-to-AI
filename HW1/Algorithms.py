import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self, state: Tuple, parent : 'Node' = None) -> None:
        self.state = state #STATE _ (Position,Ball1,Ball2)
        self.parent = parent
        self.successors = []
        self.actionsList = []
        
    def expand(self, d:dict) -> list['Node']:
        if(d == None):
            return []
        for a in d.keys():
            if(d[a] != (None, None, None)):# and d[a][0] != (None,False,False)):
                NewNode = Node(d[a][0],self)
                NewNode.actionsList.extend(self.actionsList)
                NewNode.actionsList.append(a)
                
                self.successors.append(NewNode)
        return self.successors
        

class BFSAgent():
    def __init__(self) -> None:
        self.Open = []
        self.Close = []
        self.nodesExpanded = 0
        self.totalCost = 0
        #self.actions = []

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        n = Node(env.get_initial_state())
        if(env.is_final_state(n.state)):
            return ([],0,1) #OR 0 for first node
        self.Open.clear()
        self.Close.clear()
        self.Open.append(n)
        while len(self.Open) != 0:
            n = self.Open.pop()
            self.Close.append(n.state)
            for child in n.expand(env.succ(n.state)):
                self.nodesExpanded += 1
                if (child.state not in self.Close) and (child not in self.Open):
                    if env.is_final_state(child.state):
                        return (child.actionList,self.totalCost,self.nodesExpanded)
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