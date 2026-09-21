"""Exact readout certificates on one declared affine section family."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReadoutEquivalence:
    """Equality of two readouts throughout a specified section family."""
    family_digest: str
    left_digest: str
    right_digest: str
    offset: object
    variation: object
    equivalent: bool
    family_dimension: int

    @classmethod
    def check(cls, family, left, right):
        import numpy as np
        from rexgraph.coordinate_map import CoordinateDifference, _is_action
        from rexgraph.section_calculus import SectionFamily
        if not isinstance(family, SectionFamily) or not _is_action(left) or not _is_action(right):
            raise TypeError("readout equivalence requires an exact section family and actions")
        if left.domain != right.domain or left.codomain != right.codomain:
            raise ValueError("readouts require identical named endpoints")
        image = family.observe(CoordinateDifference(left, right))
        equal = image.determined and not bool(np.any(image.particular.values))
        return cls(family.coefficient_digest, left.coefficient_digest, right.coefficient_digest,
                   image.particular, image.directions, equal, family.dimension)

    @property
    def digest(self):
        from rexgraph.coordinate_map import _identity
        return _identity(("rcql.readout-equivalence.v1", self.family_digest,
                          self.left_digest, self.right_digest, self.offset.coefficient_digest,
                          self.variation.coefficient_digest, self.equivalent, self.family_dimension))

    def verify(self, family, left, right):
        fresh = type(self).check(family, left, right)
        if fresh.digest != self.digest:
            raise ValueError("readout certificate no longer matches its declarations")
        return self.equivalent

    def replacement(self, family, left, right):
        """Return a certified alternative readout, not a general program rewrite."""
        if not self.verify(family, left, right):
            raise ValueError("the readouts differ on the retained family")
        return family.observe(right)

    def to_bytes(self):
        """Retain a claim whose exact inputs must be supplied for verification."""
        from .program_codec import dumps
        return dumps({"schema": "rcql.readout-claim", "version": 1,
                      "family": self.family_digest, "left": self.left_digest,
                      "right": self.right_digest, "certificate": self.digest,
                      "equivalent": self.equivalent})

    @staticmethod
    def claim(raw):
        from .program_codec import loads
        if type(raw) is not bytes:
            raise TypeError("readout claims require immutable declaration bytes")
        data = loads(raw)
        if (not isinstance(data, dict) or set(data) !=
                {"schema", "version", "family", "left", "right", "certificate", "equivalent"}
                or data["schema"] != "rcql.readout-claim" or type(data["version"]) is not int
                or data["version"] != 1 or data["equivalent"] is not True):
            raise ValueError("an equivalent readout claim is required")
        for key in ("family", "left", "right", "certificate"):
            value = data[key]
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("readout claim requires exact declaration digests")
        return data

    @classmethod
    def verify_bytes(cls, raw, family, left, right):
        cls.claim(raw)
        fresh = cls.check(family, left, right)
        if not fresh.equivalent or fresh.to_bytes() != raw:
            raise ValueError("readout claim does not match the complete supplied family and maps")
        return fresh

    def as_record(self):
        return {"equivalent": self.equivalent, "family_dimension": self.family_dimension,
                "family_digest": self.family_digest, "left_digest": self.left_digest,
                "right_digest": self.right_digest, "offset": self.offset,
                "variation": self.variation, "certificate_digest": self.digest,
                "scope": "linear readout values on the declared affine family"}
