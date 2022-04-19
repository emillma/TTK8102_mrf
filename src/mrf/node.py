import numpy as np
import sympy as sp
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Node(sp.Matrix):
    neighboors: Tuple['Node']
    fields: list
    name: str

    def __init__(self, name, fields, neighboors):
        pass


a = sp.Matrix(sp.MatrixSymbol('a', 4, 1))
pass
