"""Parser for the small RCQL expression grammar."""
from __future__ import annotations

import json
import re
from fractions import Fraction
from math import isfinite

from .ast import (
    Alias,
    Call,
    Expr,
    LetBinding,
    ListExpr,
    Literal,
    Member,
    MutationQuery,
    Parameter,
    Query,
    Reference,
    StructuralEdit,
    Comparison,
    MatchBinding,
)

_TOKEN = re.compile(
    r'\s*(?:(?P<number>[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)|'
    r'(?P<string>"(?:[^"\\]|\\.)*")|'
    r'(?P<param>\$[A-Za-z_]\w*)|(?P<name>[A-Za-z_]\w*)|(?P<punct>!=|<=|>=|==|[<>(),/=.\[\]]))')
_INTEGER = re.compile(r'[+-]?[0-9]+\Z')


def _syntax_error(text: str, message: str, position: int) -> SyntaxError:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    line = text.count("\n", 0, position) + 1
    excerpt = text[start:] if end < 0 else text[start:end + 1]
    return SyntaxError(message, ("<rcql>", line, position - start + 1, excerpt))


class _Parser:
    def __init__(self, text: str):
        if not isinstance(text, str):
            raise TypeError("RCQL input must be text")
        self.text = text
        self.tokens = []
        self.positions = []
        pos = 0
        stop = len(text.rstrip())
        while pos < stop:
            match = _TOKEN.match(text, pos)
            if match is None:
                while pos < stop and text[pos].isspace():
                    pos += 1
                raise _syntax_error(text, f"unexpected input at {text[pos:pos + 24]!r}", pos)
            kind = match.lastgroup
            self.tokens.append((kind, match.group(kind)))
            self.positions.append(match.start(kind))
            pos = match.end()
        self.i = 0
        self.locals = set()
        self.source_context = False
        self.source_alias = None

    def peek(self, offset=0):
        index = self.i + offset
        return self.tokens[index] if index < len(self.tokens) else (None, None)

    def error(self, message, *, previous=False):
        index = max(0, self.i - 1) if previous else self.i
        position = self.positions[index] if index < len(self.positions) else len(self.text)
        return _syntax_error(self.text, message, position)

    def take(self, value=None):
        token = self.peek()
        if token[0] is None:
            raise self.error("unexpected end of query")
        if value is not None and token[1].upper() != value:
            raise self.error(f"expected {value}, got {token[1]}")
        self.i += 1
        return token

    def expr(self, minimum=0) -> Expr:
        if self.peek()[1] and self.peek()[1].upper() == "NOT":
            self.take("NOT")
            result = Comparison("NOT", self.expr(3), Literal(False))
        else:
            result = self.postfix()
        precedence = {"OR": 1, "AND": 2, "=": 3, "==": 3, "!=": 3, "<": 3, "<=": 3, ">": 3, ">=": 3}
        while self.peek()[1] and precedence.get(self.peek()[1].upper(), -1) >= minimum:
            operation = self.take()[1].upper()
            right = self.expr(precedence[operation] + 1)
            result = Comparison(operation, result, right)
        return result

    def postfix(self) -> Expr:
        result = self.atom()
        while self.peek()[1] == ".":
            self.take(".")
            kind, name = self.take()
            if kind != "name":
                raise SyntaxError("member access expects an identifier")
            if isinstance(result, Reference) and result.name == self.source_alias:
                result = self.call_expr(name)
            else:
                try:
                    result = Member(result, name)
                except ValueError as exc:
                    raise SyntaxError(str(exc)) from exc
        return result

    def call_expr(self, name):
        from .arguments import bind_arguments
        from .names import canonical_name
        self.take("(")
        args, keywords = [], []
        if self.peek()[1] != ")":
            while True:
                if self.peek()[0] == "name" and self.peek(1)[1] == "=":
                    key = self.take()[1]
                    self.take("=")
                    keywords.append((key, self.expr()))
                elif keywords:
                    raise SyntaxError("positional arguments must precede named arguments")
                else:
                    args.append(self.expr())
                if self.peek()[1] != ",":
                    break
                self.take(",")
        self.take(")")
        try:
            name = canonical_name(name)
            return Call(name, bind_arguments(name, args, keywords, source=self.source_context))
        except (TypeError, ValueError) as exc:
            raise SyntaxError(str(exc)) from exc

    def atom(self) -> Expr:
        kind, value = self.take()
        if value == "(":
            out = self.expr()
            self.take(")")
            return out
        if value == "[":
            items = []
            if self.peek()[1] != "]":
                while True:
                    items.append(self.expr())
                    if self.peek()[1] != ",":
                        break
                    self.take(",")
            self.take("]")
            return ListExpr(tuple(items))
        if kind == "param":
            return Parameter(value[1:])
        if kind == "number":
            if self.peek()[1] == "/":
                self.take("/")
                denominator_kind, denominator = self.take()
                if not (_INTEGER.fullmatch(value) and denominator_kind == "number"
                        and _INTEGER.fullmatch(denominator)):
                    raise SyntaxError("rational literals require integer numerator and denominator")
                if int(denominator) == 0:
                    raise SyntaxError("rational literal denominator must be nonzero")
                return Literal(Fraction(int(value), int(denominator)))
            if _INTEGER.fullmatch(value):
                return Literal(int(value))
            number = float(value)
            if not isfinite(number):
                raise SyntaxError("decimal literal must be finite")
            return Literal(number)
        if kind == "string":
            try:
                return Literal(json.loads(value))
            except json.JSONDecodeError as exc:
                raise SyntaxError(f"invalid string literal: {exc.msg}") from exc
        if kind != "name":
            raise SyntaxError(f"expected expression, got {value}")
        # TRUE and FALSE are literals, not nullary calls. Several signatures take an
        # `exact: bool`, and without this a bare name falls through to Call below and
        # fails as a missing operator, which left the exact path reachable only by
        # binding a parameter. Guarded on the absence of a call parenthesis so a future
        # operator of either name still parses as one.
        if value.upper() in ("TRUE", "FALSE") and self.peek()[1] != "(":
            return Literal(value.upper() == "TRUE")
        if value.upper() == "NONE" and self.peek()[1] != "(":
            return Literal(None)
        if self.peek()[1] != "(":
            if value in self.locals or value == self.source_alias:
                return Reference(value)
            try:
                return Call(value, ())
            except ValueError as exc:
                raise SyntaxError(str(exc)) from exc
        return self.call_expr(value)


def parse(text: str) -> Query | MutationQuery:
    """Parse FROM, ordered LET bindings and RETURN, with optional EXPLAIN."""
    parser = _Parser(text)
    try:
        return _parse(parser)
    except SyntaxError as exc:
        if exc.lineno is not None:
            raise
        raise parser.error(str(exc), previous=True) from exc


def _parse(parser: _Parser) -> Query | MutationQuery:
    explain = False
    if parser.peek()[1] and parser.peek()[1].upper() == "EXPLAIN":
        parser.take("EXPLAIN")
        explain = True
    parser.take("FROM")
    parser.source_context = True
    source = parser.expr()
    parser.source_context = False
    if parser.peek()[1] and parser.peek()[1].upper() == "AS":
        parser.take("AS")
        kind, name = parser.take()
        if kind != "name":
            raise SyntaxError("source alias must be an identifier")
        try:
            Reference(name)
        except ValueError as exc:
            raise SyntaxError(str(exc)) from exc
        parser.source_alias = name
    bindings = []
    while parser.peek()[1] and parser.peek()[1].upper() == "LET":
        parser.take("LET")
        kind, name = parser.take()
        if kind != "name":
            raise SyntaxError("LET expects a local identifier, not a parameter")
        if name in parser.locals or name == parser.source_alias:
            raise SyntaxError(f"duplicate LET binding {name!r}")
        parser.take("=")
        try:
            # Put the name in scope so a self reference becomes a clear unbound
            # reference error during planning, not an accidental operator call.
            Reference(name)
            parser.locals.add(name)
            bindings.append(LetBinding(name, parser.expr()))
        except ValueError as exc:
            raise SyntaxError(str(exc)) from exc
    if parser.peek()[1] and parser.peek()[1].upper() == "MUTATE":
        parser.take("MUTATE")
        record_id = parser.expr()
        parser.take("SET")
        fields = {}
        allowed = {"state", "actor", "valid_from", "valid_to", "expected_version", "expected_hash"}
        while True:
            kind, name = parser.take()
            name = name.lower()
            if kind != "name" or name not in allowed:
                raise SyntaxError(f"unknown mutation field {name!r}; expected {sorted(allowed)}")
            if name in fields:
                raise SyntaxError(f"duplicate mutation field {name!r}")
            parser.take("=")
            fields[name] = parser.expr()
            if parser.peek()[1] != ",":
                break
            parser.take(",")
        if "state" not in fields:
            raise SyntaxError("mutation requires SET state = expression")
        while parser.peek()[1] and parser.peek()[1].upper() in {"ADD", "REMOVE"}:
            operation = parser.take()[1].upper()
            fields["state"] = StructuralEdit(fields["state"], operation, parser.expr())
        parser.take("COMMIT")
        if parser.peek()[0] is not None:
            raise parser.error(f"unexpected token {parser.peek()[1]}")
        if "state" not in fields:
            raise SyntaxError("mutation requires SET state = expression")
        return MutationQuery(source, record_id, fields["state"],
                             fields.get("actor", Literal("")),
                             fields.get("valid_from", Literal(None)),
                             fields.get("valid_to", Literal(None)),
                             fields.get("expected_version", Literal(None)),
                             explain, tuple(bindings), parser.source_alias,
                             fields.get("expected_hash", Literal(None)))
    matches, where = [], None
    if parser.peek()[1] and parser.peek()[1].upper() == "MATCH":
        parser.take("MATCH")
        while True:
            kind, name = parser.take()
            if kind != "name" or name in parser.locals or name == parser.source_alias:
                raise SyntaxError("MATCH requires a fresh local identifier")
            parser.take("IN")
            value = parser.expr()
            try:
                matches.append(MatchBinding(name, value))
            except ValueError as exc:
                raise SyntaxError(str(exc)) from exc
            parser.locals.add(name)
            if parser.peek()[1] != ",":
                break
            parser.take(",")
        if parser.peek()[1] and parser.peek()[1].upper() == "WHERE":
            parser.take("WHERE")
            where = parser.expr()
    parser.take("RETURN")
    returns = []
    names = set()
    while True:
        value = parser.expr()
        if parser.peek()[1] and parser.peek()[1].upper() == "AS":
            parser.take("AS")
            kind, name = parser.take()
            if kind != "name" or name in names:
                raise SyntaxError("return aliases must be distinct identifiers")
            try:
                value = Alias(value, name)
            except ValueError as exc:
                raise SyntaxError(str(exc)) from exc
            names.add(name)
        returns.append(value)
        if parser.peek()[1] != ",":
            break
        parser.take(",")
    order, limit, offset = [], None, 0
    if parser.peek()[1] and parser.peek()[1].upper() == "ORDER":
        parser.take("ORDER")
        parser.take("BY")
        while True:
            value, descending = parser.expr(), False
            if parser.peek()[1] and parser.peek()[1].upper() in {"ASC", "DESC"}:
                descending = parser.take()[1].upper() == "DESC"
            order.append((value, descending))
            if parser.peek()[1] != ",":
                break
            parser.take(",")
    for field in ("LIMIT", "OFFSET"):
        if parser.peek()[1] and parser.peek()[1].upper() == field:
            parser.take(field)
            value = parser.atom()
            if not isinstance(value, Literal) or type(value.value) is not int or value.value < 0:
                raise SyntaxError(f"{field} requires a nonnegative integer literal")
            if field == "LIMIT":
                limit = value.value
            else:
                offset = value.value
    if (where is not None or order or limit is not None or offset) and not matches:
        raise SyntaxError("WHERE/ORDER/LIMIT/OFFSET require MATCH bindings")
    if parser.peek()[0] is not None:
        raise parser.error(f"unexpected token {parser.peek()[1]}")
    return Query(source, tuple(returns), explain=explain, bindings=tuple(bindings), source_alias=parser.source_alias,
                 matches=tuple(matches), where=where, order=tuple(order), limit=limit, offset=offset)
