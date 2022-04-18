from dataclasses import dataclass
from typing import Set, TYPE_CHECKING
import sympy as sp
if TYPE_CHECKING:
    from node import Node


@dataclass
class Clique(sp.Function):
    nodes: Set['Node']

    def potential_function()
