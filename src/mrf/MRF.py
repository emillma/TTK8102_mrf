import networkx as nx
import numpy as np
from dataclasses import dataclass, field
import time
from typing import List, Union

@dataclass(frozen=True)
class Node:
    creationtime: float = field(default_factory=time.perf_counter)



@dataclass(frozen=True)
class RandomNode(Node):
    pass


@dataclass(frozen=True)
class ObservedNode(Node):
    pass

@dataclass
class Factor:
    def evalueate(self):
        raise NotImplemented

    def condition(self):
        raise NotImplemented

@dataclass
class BinaryFactor(Factor):
    a: Union[RandomNode, ObservedNode]
    b: Union[RandomNode, ObservedNode]


@dataclass
class MRF:
    nodes: List[Node]
    factors: List['BinaryFactor']


    def __init__(self):
        self.nodes = []
        self.factors = []
        self.graph = nx.Graph()

    def add_factor(self, factor: BinaryFactor):
        # if factor.a not in self.nodes:
        #     raise ValueError("Node ", factor.a, "not yet in this MRF.")
        # if factor.b not in self.nodes:
        #     raise ValueError("Node ", factor.b, "not yet in this MRF.")
        # self.factors.append(factor)
        self.graph.add_edge(factor.a, factor.b, factor=factor)
        # factor.a.neighbours.append(factor.b)
        # factor.b.neighbours.append(factor.a)

    def add_node(self, node: Node):
        self.nodes.append(node)
        self.graph.add_node(node, node=node)

