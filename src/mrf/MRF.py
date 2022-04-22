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

    def get_other_node(self, x):
        if x == self.a:
            return self.b
        elif x == self.b:
            return self.a
        else:
            raise NotImplemented


@dataclass
class MRF:
    nodes: List[Node]

    def __init__(self):
        self.nodes = []
        self.graph = nx.Graph()

    def add_factor(self, factor: BinaryFactor):
        self.graph.add_edge(factor.a, factor.b, factor=factor)

    def add_node(self, node: Node):
        self.nodes.append(node)
        self.graph.add_node(node, node=node)
