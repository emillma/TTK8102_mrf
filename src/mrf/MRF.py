import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class MRF:
    nodes: List['Node']
    factors: List['Factor']

@dataclass
class Node:
    pass

class Factor:

    def evalueate(self, A: Node, B: Node):
        raise NotImplemented

    def condition(self, A: Node, B: Node):
        raise NotImplemented

