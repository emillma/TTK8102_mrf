import sympy as sp
import inspect
from IPython.display import display, Math

sp.bernoulli


def name_factory(index=0):
    index += 1
    return (index-1).to_bytes(2, 'big').hex()


def print_function(func, name=None):
    argsyms = sp.symbols(inspect.getfullargspec(func).args)
    name = name or func.__repr__(func).replace('_', r'\_')
    return sp.Eq(sp.Function(name)(*argsyms),
                 func(*argsyms))


def showlatex(stuff):
    return display(Math(sp.latex(stuff)))
