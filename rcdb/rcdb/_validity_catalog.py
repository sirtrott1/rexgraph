"""Disposable collection coverage rebuilt from the checked native histories.

Each lineage contributes disjoint, coalesced intervals, independent of versions.
Deletion keeps retained validity coverage. The selected version still comes from
the lineage index; this catalog never stores or supplies record metadata.
"""
from __future__ import annotations


class _Interval:
    __slots__ = ("key", "end", "left", "right", "height", "max_end")

    def __init__(self, key, end):
        self.key, self.end = key, end
        self.left = self.right = None
        self.height, self.max_end = 1, end


def _height(node):
    return 0 if node is None else node.height


def _measure(node):
    node.height = 1 + max(_height(node.left), _height(node.right))
    node.max_end = max(node.end,
                       float("-inf") if node.left is None else node.left.max_end,
                       float("-inf") if node.right is None else node.right.max_end)


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
    skew = _height(node.left) - _height(node.right)
    if skew > 1:
        if _height(node.left.left) < _height(node.left.right):
            node.left = _left(node.left)
        return _right(node)
    if skew < -1:
        if _height(node.right.right) < _height(node.right.left):
            node.right = _right(node.right)
        return _left(node)
    return node


def _set(node, key, end):
    if node is None:
        return _Interval(key, end)
    if key < node.key:
        node.left = _set(node.left, key, end)
    elif key > node.key:
        node.right = _set(node.right, key, end)
    else:
        node.end = end
    return _balance(node)


def _drop(node, key):
    if key < node.key:
        node.left = _drop(node.left, key)
    elif key > node.key:
        node.right = _drop(node.right, key)
    else:
        if node.left is None:
            return node.right
        if node.right is None:
            return node.left
        successor = node.right
        while successor.left is not None:
            successor = successor.left
        node.key, node.end = successor.key, successor.end
        node.right = _drop(node.right, successor.key)
    return _balance(node)


class _Intervals:
    def __init__(self):
        self.root = None

    def floor(self, key):
        node, candidate = self.root, None
        while node is not None:
            if node.key <= key:
                candidate, node = node, node.right
            else:
                node = node.left
        return candidate

    def ceiling(self, key):
        node, candidate = self.root, None
        while node is not None:
            if node.key >= key:
                candidate, node = node, node.left
            else:
                node = node.right
        return candidate

    def set(self, key, end):
        self.root = _set(self.root, key, end)

    def drop(self, key):
        self.root = _drop(self.root, key)

    def covering(self, instant):
        def visit(node):
            if node is None or node.max_end <= instant:
                return
            yield from visit(node.left)
            if node.key[0] <= instant:
                if instant < node.end:
                    yield node.key[1]
                yield from visit(node.right)
        return visit(self.root)


class _ValidityCatalog:
    """Union coverage of every retained lineage, with a global interval AVL.

    First use reads all retained versions. Warm queries prune subtrees by their
    greatest end and ordered starts, then visit covering lineages only. Broad
    matches still cost work proportional to candidates; page ordering is separate.
    Put coalesces only its lineage's touching ranges. Every removed interval was
    previously inserted, bounding total update work by O(N log N) for N puts.
    """
    def __init__(self):
        self._all = _Intervals()
        self._lineages = {}
        self._order = {}

    def add(self, row):
        start = row.valid_from if row.valid_from is not None else row.tx_from
        self.add_interval(row.id, start, row.valid_to)

    def add_interval(self, record_id, start, end):
        end = float("inf") if end is None else end
        intervals = self._lineages.get(record_id)
        if intervals is None:
            intervals = self._lineages[record_id] = _Intervals()
            self._order[record_id] = len(self._order)
        # Include the predecessor if it touches this range. Merge adjacent
        # half open intervals too: their union has no missing instant.
        prior = intervals.floor((start, record_id))
        if prior is not None and prior.end >= start:
            start = prior.key[0]
        node = intervals.ceiling((start, record_id))
        while node is not None and node.key[0] <= end:
            key, old_end = node.key, node.end
            end = max(end, old_end)
            intervals.drop(key)
            self._all.drop(key)
            node = intervals.ceiling((start, record_id))
        key = (start, record_id)
        intervals.set(key, end)
        self._all.set(key, end)

    def ids_at(self, instant):
        # Preserve the engine's original lineage insertion order, without
        # iterating unrelated lineages. Native collection pages sort afterwards.
        return sorted(self._all.covering(instant), key=self._order.__getitem__)
