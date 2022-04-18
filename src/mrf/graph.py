from dataclasses import dataclass
from typing import Callable, Set, Tuple, TYPE_CHECKING

from clique import Clique

if TYPE_CHECKING:
    from node import Node
    from clique import Clique


@dataclass
class MRF:
    nodes: Set['Node']
    cliques: Set['Clique']
