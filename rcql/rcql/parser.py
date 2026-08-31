"""Parser for the small RCQL expression grammar."""
from __future__ import annotations

import re

from .ast import Call, Expr, Literal, Parameter, Query

_TOKEN = re.compile(
    r'\s*(?:(?P<number>\d+(?:\.\d+)?)|(?P<string>"(?:[^"\\]|\\.)*")|'
    r'(?P<param>\$[A-Za-z_]\w*)|(?P<name>[A-Za-z_]\w*)|(?P<punct>[(),]))')


class _Parser:
    def __init__(self, text: str):
        self.tokens = []
        pos = 0
        while pos < len(text):
            match = _TOKEN.match(text, pos)
            if match is None:
                raise SyntaxError(f"unexpected input at {text[pos:pos + 24]!r}")
            kind = match.lastgroup
            self.tokens.append((kind, match.group(kind)))
            pos = match.end()
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self, value=None):
        token = self.peek()
        if token[0] is None:
            raise SyntaxError("unexpected end of query")
        if value is not None and token[1].upper() != value:
            raise SyntaxError(f"expected {value}, got {token[1]}")
        self.i += 1
        return token

    def expr(self) -> Expr:
        kind, value = self.take()
        if kind == "param":
            return Parameter(value[1:])
        if kind == "number":
            return Literal(float(value) if "." in value else int(value))
        if kind == "string":
            return Literal(bytes(value[1:-1], "utf8").decode("unicode_escape"))
        if kind != "name":
            raise SyntaxError(f"expected expression, got {value}")
        if self.peek()[1] != "(":
            return Call(value.upper(), ())
        self.take("(")
        args = []
        if self.peek()[1] != ")":
            while True:
                args.append(self.expr())
                if self.peek()[1] != ",":
                    break
                self.take(",")
        self.take(")")
        return Call(value.upper(), tuple(args))


def parse(text: str) -> Query:
    """Parse `FROM expression RETURN expression, ...` with optional EXPLAIN."""
    parser = _Parser(text)
    explain = False
    if parser.peek()[1] and parser.peek()[1].upper() == "EXPLAIN":
        parser.take("EXPLAIN")
        explain = True
    parser.take("FROM")
    source = parser.expr()
    parser.take("RETURN")
    returns = []
    while True:
        returns.append(parser.expr())
        if parser.peek()[1] != ",":
            break
        parser.take(",")
    if parser.peek()[0] is not None:
        raise SyntaxError(f"unexpected token {parser.peek()[1]}")
    return Query(source, tuple(returns), explain=explain)
