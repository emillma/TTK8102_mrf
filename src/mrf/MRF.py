import numpy as np
from dataclasses import dataclass
from typing import List, Union


@dataclass
class MRF:
    nodes: List['Union[RandomNode, ObservedNode]']
    factors: List['Factor']



@dataclass
class RandomNode:
    pass


@dataclass
class ObservedNode:
    def __init__(self, fields):
        raise NotImplemented

@dataclass
class Factor:
    def evalueate(self, A: Union[RandomNode, ObservedNode], B: Union[RandomNode, ObservedNode]):
        raise NotImplemented

    def condition(self, A: Union[RandomNode, ObservedNode], B: Union[RandomNode, ObservedNode]):
        raise NotImplemented
