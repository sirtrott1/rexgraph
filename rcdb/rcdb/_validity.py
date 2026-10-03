"""Disposable latest version coverage for one checked native record lineage.

Each publication overlays its half open validity interval on earlier coverage.
An AVL map stores only boundaries and version numbers, never payloads or metadata.
Deletion does not alter validity only history; replay can rebuild this projection.
"""
from __future__ import annotations


class _Boundary:
    __slots__ = ("start", "version", "left", "right", "height")

    def __init__(self, start, version):
        self.start, self.version = start, version
        self.left = self.right = None
        self.height = 1


def _height(node):
    return 0 if node is None else node.height


def _measure(node):
    node.height = 1+max(_height(node.left), _height(node.right))


def _left(node):
    root = node.right
    node.right, root.left = root.left, node
    _measure(node)
    _measure(root)
    return root


def _right(node):
    root = node.left
    node.left, root.right = root.right, node
    _measure(node)
    _measure(root)
    return root


def _balance(node):
    _measure(node)
    skew = _height(node.left)-_height(node.right)
    if skew > 1:
        if _height(node.left.left) < _height(node.left.right):
            node.left = _left(node.left)
        return _right(node)
    if skew < -1:
        if _height(node.right.right) < _height(node.right.left):
            node.right = _right(node.right)
        return _left(node)
    return node


def _set(node, start, version):
    if node is None:
        return _Boundary(start, version)
    if start < node.start:
        node.left = _set(node.left, start, version)
    elif start > node.start:
        node.right = _set(node.right, start, version)
    else:
        node.version = version
    return _balance(node)


def _drop(node, start):
    if start < node.start:
        node.left = _drop(node.left, start)
    elif start > node.start:
        node.right = _drop(node.right, start)
    else:
        if node.left is None:
            return node.right
        if node.right is None:
            return node.left
        successor = node.right
        while successor.left is not None:
            successor = successor.left
        node.start, node.version = successor.start, successor.version
        node.right = _drop(node.right, successor.start)
    return _balance(node)


class _ValidityIndex:
    """Latest retained coverage, built in native version allocation order.

    Reads visit O(log B) boundaries. A put sets at most two boundaries and
    removes overwritten boundaries in O((K+1) log B) work. Each removed boundary
    was previously inserted, so total construction/update work is O(N log N)
    over N publications; a single broad overwrite can still remove many ranges.
    """
    def __init__(self):
        self._root = None

    def version_at(self, instant):
        node, version = self._root, None
        while node is not None:
            if node.start <= instant:
                version, node = node.version, node.right
            else:
                node = node.left
        return version

    def coverage(self):
        """Borrow only indexed interval coordinates, without reading history."""
        def boundaries(node):
            if node is not None:
                yield from boundaries(node.left)
                yield node
                yield from boundaries(node.right)
        prior = None
        for node in boundaries(self._root):
            if prior is not None and prior.version is not None:
                yield prior.start, node.start
            prior = node
        if prior is not None and prior.version is not None:
            yield prior.start, None

    def _ceiling(self, start):
        node, candidate = self._root, None
        while node is not None:
            if node.start >= start:
                candidate, node = node, node.left
            else:
                node = node.right
        return candidate

    def add(self, row):
        start = row.valid_from if row.valid_from is not None else row.tx_from
        end = row.valid_to
        # Capture the old value AT the excluded right endpoint before erasing
        # covered boundaries. None denotes a gap, not an absent record version.
        resume = None if end is None else self.version_at(end)
        node = self._ceiling(start)
        while node is not None and (end is None or node.start < end):
            self._root = _drop(self._root, node.start)
            node = self._ceiling(start)
        self._root = _set(self._root, start, row.version)
        if end is not None:
            self._root = _set(self._root, end, resume)
