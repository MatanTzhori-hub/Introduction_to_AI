import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict



class DFSGAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError
        


class UCSAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class WeightedAStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError   



class AStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

