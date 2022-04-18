import sympy as sp
import networkx as nx
from sympy import Eq, And
from utils import print_function
people = sp.symbols('a:d')


class vote_cost(sp.Function):
    @classmethod
    def eval(cls, x, y):
        return sp.Piecewise((10, And(Eq(x, 1), Eq(y, 1))),
                            (5, And(Eq(x, 0), Eq(y, 0))),
                            (1, True))


costs = []
for i in range(len(people)):
    costs.append(vote_cost(people[i],  people[(i+1) % len(people)]))
costs[0]
print_function(vote_cost, r'\phi')
