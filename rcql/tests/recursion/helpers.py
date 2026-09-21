from rcql import SourcePolicy, bind, RecursiveDefinition, RecursiveProgram, ProgramInput, ValueKind, recur
from rcql.ast import Parameter, Literal, Call, ListExpr, Comparison
from rexgraph.graph import RexGraph


def source():
    return RexGraph.from_graph(sources=[0, 1, 2], targets=[1, 2, 3])


def binding(rex):
    return bind('r', rex, SourcePolicy.allow('*'))


def integer(name):
    return ProgramInput(name, ValueKind.EXACT_INTEGER.value)


def minus(n, value=1):
    return Call('SUM', (ListExpr((n, Literal(-value))),))


def counter():
    n=Parameter('n')
    step=Call('SUM',(ListExpr((Literal(1),recur('count',minus(n)))),))
    return RecursiveProgram('Counter',(RecursiveDefinition('count',(integer('n'),),
        Comparison('<=',n,Literal(0)),Literal(0),step,integer('result'),decreases='n'),))


def fibonacci():
    n=Parameter('n')
    step=Call('SUM',(ListExpr((recur('fib',minus(n)),recur('fib',minus(n,2)))),))
    return RecursiveProgram('Fibonacci',(RecursiveDefinition('fib',(integer('n'),),
        Comparison('<=',n,Literal(1)),n,step,integer('result'),decreases='n'),))
