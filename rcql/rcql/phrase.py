"""Exact phrase gluing over explicitly declared relational correspondences.

One selected RCDB state can be read as a local section of a phrase.  A correspondence
between selected states is not inferred from record keys, matching shapes, or equal
values: it is one primary C1 relation whose C0 boundary is the participating state
stalks.  Restriction maps live on those incidences.  This preserves a k-state
correspondence as one arity-k relation, with a distinguished head and declared shares,
instead of reducing it to a clique or a star of pairwise joins.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rexgraph.sheaf import ExactGlueResult, ExactGluingObstruction, ExactSheaf

from .binding import Binding
from .capabilities import BoundSource, SourcePolicy
from .types import SourceRef

__all__ = [
    "PhraseCorrespondence",
    "PhraseGlueResult",
    "PhraseGluingObstruction",
    "PhraseMapError",
    "PhraseSheaf",
    "PhraseStalk",
    "UndeclaredRestrictionError",
]


class PhraseMapError(ValueError):
    """The declared phrase correspondence is not a valid relational cell."""


class UndeclaredRestrictionError(PhraseMapError):
    """A cross-state phrase incidence has no explicitly declared restriction map."""


@dataclass(frozen=True)
class PhraseGluingObstruction:
    """One exact gluing strain, named in the phrase's stalk vocabulary.

    ``exact`` retains the canonical index-level representative.  The named fields are
    an observation layer only: they make a returned strain actionable without replacing
    its exact transported values or residual.
    """

    exact: ExactGluingObstruction
    left_stalk: str
    right_stalk: str
    correspondence: str

    @property
    def left(self):
        """Exact transported section from ``left_stalk``."""
        return self.exact.left

    @property
    def right(self):
        """Exact transported section from ``right_stalk``."""
        return self.exact.right

    @property
    def residual(self):
        """Exact left-minus-right strain at ``correspondence``."""
        return self.exact.residual


@dataclass(frozen=True)
class PhraseGlueResult(ExactGlueResult):
    """An exact glue result with components and strains named by phrase stalks.

    It remains an :class:`rexgraph.sheaf.ExactGlueResult`, retaining its canonical
    ratio, components, and obstructions.  ``named_components`` and
    ``named_obstructions`` add the phrase vocabulary at the application boundary.
    """

    named_components: tuple[tuple[str, ...], ...]
    named_obstructions: tuple[PhraseGluingObstruction, ...]
    contributors: tuple[SourceRef, ...]
    policy: SourcePolicy


@dataclass(frozen=True)
class PhraseStalk:
    """One selected source state and its local section address inside a phrase."""

    name: str
    source: Binding

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PhraseMapError("a phrase stalk needs a nonempty name")
        if not isinstance(self.source, Binding):
            raise TypeError("a phrase stalk source must be a policy-bearing Binding")


@dataclass(frozen=True)
class PhraseCorrespondence:
    """One primary arity-k relation over named phrase stalks.

    Stalk order is intentional: the first is the distinguished head of the C1 boundary
    and the others receive the declared share.  Maps are not stored here because they
    live on the individual ``(stalk, correspondence)`` incidences.
    """

    name: str
    stalks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PhraseMapError("a phrase correspondence needs a nonempty name")
        if len(self.stalks) < 2:
            raise PhraseMapError(
                f"phrase correspondence {self.name!r} must relate at least two stalks; "
                "an arity-1 witness is not a correspondence"
            )
        if any(not isinstance(name, str) or not name for name in self.stalks):
            raise PhraseMapError(f"phrase correspondence {self.name!r} has an invalid stalk name")
        if len(set(self.stalks)) != len(self.stalks):
            raise PhraseMapError(
                f"phrase correspondence {self.name!r} repeats a stalk; "
                "occurrence identity must be declared separately"
            )


class PhraseSheaf(ExactSheaf):
    """A local-to-global section over state stalks and explicit correspondences.

    The internal relational complex has C0 cells for selected state stalks and C1 cells
    for declared correspondences.  Its exact grade-zero sheaf compares the transported
    local sections at each correspondence.  Unlike :class:`rexgraph.sheaf.ExactSheaf`,
    identity is *not* a default on this cross-state boundary: every incidence must be
    given a restriction map before ``glue`` can run.
    """

    def __init__(self, stalks: Sequence[PhraseStalk],
                 correspondences: Sequence[PhraseCorrespondence], *, stalk_dim: int = 1):
        self.stalks = tuple(stalks)
        self.correspondences = tuple(correspondences)
        if not self.stalks:
            raise PhraseMapError("a phrase needs at least one selected state stalk")
        names = tuple(stalk.name for stalk in self.stalks)
        if len(set(names)) != len(names):
            raise PhraseMapError("phrase stalk names must be unique")
        correspondence_names = tuple(item.name for item in self.correspondences)
        if len(set(correspondence_names)) != len(correspondence_names):
            raise PhraseMapError("phrase correspondence names must be unique")
        self._stalk_index = {stalk.name: index for index, stalk in enumerate(self.stalks)}
        self._correspondence_index = {
            item.name: index for index, item in enumerate(self.correspondences)
        }
        for item in self.correspondences:
            unknown = tuple(name for name in item.stalks if name not in self._stalk_index)
            if unknown:
                raise PhraseMapError(
                    f"phrase correspondence {item.name!r} names unknown stalks {unknown!r}"
                )
        carried = {name for item in self.correspondences for name in item.stalks}
        uncarried = tuple(name for name in names if name not in carried)
        if uncarried:
            raise PhraseMapError(
                "every phrase stalk must occur in a declared correspondence; "
                f"uncarried stalks are {uncarried!r}"
            )

        # The phrase sheaf itself is an ExactSheaf at grade zero, so the existing
        # ``FROM $correspondence RETURN GLUE($section)`` spelling remains the actual
        # execution surface.  Parsing remains independent because this module is lazy
        # from rcql.__init__.
        import numpy as np
        from rexgraph.graph import RexGraph

        ptr = [0]
        indices: list[int] = []
        for item in self.correspondences:
            indices.extend(self._stalk_index[name] for name in item.stalks)
            ptr.append(len(indices))
        rex = RexGraph.from_hypergraph(
            np.asarray(ptr, dtype=np.int64), np.asarray(indices, dtype=np.int64),
        )
        super().__init__(
            rex, stalk_dim=stalk_dim, grade=0, require_declared_restrictions=True,
        )
        self._policy = SourcePolicy.intersection(
            *(stalk.source.source.policy for stalk in self.stalks)
        )
        contributors = tuple(stalk.source.ref for stalk in self.stalks)
        self._ref = SourceRef(
            name=f"phrase/{self._phrase_digest(contributors)[:16]}",
            state_digest=self._phrase_digest(contributors),
            policy_digest=self._policy.digest,
            contributors=contributors,
        )

    @property
    def stalk_dim(self) -> int:
        return self.d

    @property
    def policy(self) -> SourcePolicy:
        """Intersection policy for a section jointly derived from all state stalks."""
        return self._policy

    @property
    def source_ref(self) -> SourceRef:
        """Phrase provenance retaining each selected state rather than merging it."""
        return self._ref

    @property
    def contributors(self) -> tuple[SourceRef, ...]:
        """The selected state provenance in ordered stalk correspondence order."""
        return self.source_ref.contributors

    def as_bound_source(self) -> BoundSource:
        """Expose this phrase's Rex only under its derived intersection policy."""
        return BoundSource(self.rex, self.policy, ref=self.source_ref)

    def assign(self, stalk: str, values: object) -> None:
        """Assign one exact local section to a selected source-state stalk."""
        self._binding(stalk).source.require("read")
        super().assign(self._stalk(stalk), values)

    def restrict(self, stalk: str, correspondence: str, matrix: object) -> None:
        """Declare one exact map from a state stalk into a correspondence cell."""
        item = self.correspondences[self._correspondence(correspondence)]
        if stalk not in item.stalks:
            raise PhraseMapError(
                f"phrase stalk {stalk!r} is not incident to correspondence {correspondence!r}"
            )
        super().restrict(
            self._stalk(stalk), matrix, mediator=self._correspondence(correspondence),
        )

    def identity(self, stalk: str, correspondence: str) -> None:
        """Declare, rather than assume, identity on one cross-state incidence."""
        self.restrict(
            stalk, correspondence,
            [[1 if row == column else 0 for column in range(self.stalk_dim)]
             for row in range(self.stalk_dim)],
        )

    def correspondence_cell(self, name: str):
        """Return the primary C1 cell that carries one named phrase correspondence."""
        from rexgraph.cells import cell

        return cell(self.rex, 1, self._correspondence(name))

    def composite(self, name: str):
        """Read the exact existence/head/share boundary of a correspondence cell."""
        from rexgraph.cells import composite_binary

        return composite_binary(self.correspondence_cell(name))

    def glue(self) -> PhraseGlueResult:
        """Glue only after every cross-state incidence map is explicitly declared."""
        for stalk in self.stalks:
            stalk.source.source.require("read")
        missing = self.missing_restrictions()
        if missing:
            detail = ", ".join(f"({stalk}, {correspondence})" for stalk, correspondence in missing)
            raise UndeclaredRestrictionError(
                "cross-state phrase gluing requires an explicit restriction at " + detail
            )
        exact = super().glue()
        names = tuple(stalk.name for stalk in self.stalks)
        correspondence_names = tuple(item.name for item in self.correspondences)
        return PhraseGlueResult(
            gluable=exact.gluable,
            glued=exact.glued,
            components=exact.components,
            obstructions=exact.obstructions,
            named_components=tuple(
                tuple(names[index] for index in component)
                for component in exact.components
            ),
            named_obstructions=tuple(
                PhraseGluingObstruction(
                    obstruction,
                    names[obstruction.left_cell],
                    names[obstruction.right_cell],
                    correspondence_names[obstruction.mediator],
                )
                for obstruction in exact.obstructions
            ),
            contributors=tuple(stalk.source.ref for stalk in self.stalks),
            policy=self.policy,
        )

    def missing_restrictions(self) -> tuple[tuple[str, str], ...]:
        """Every state/correspondence incidence that must not inherit identity."""
        missing = []
        for correspondence in self.correspondences:
            relation = self._correspondence(correspondence.name)
            for stalk in correspondence.stalks:
                vertex = self._stalk(stalk)
                if (vertex, relation) not in self._R:
                    missing.append((stalk, correspondence.name))
        return tuple(missing)

    def _stalk(self, name: str) -> int:
        try:
            return self._stalk_index[name]
        except KeyError as exc:
            raise PhraseMapError(f"unknown phrase stalk {name!r}") from exc

    def _binding(self, name: str) -> Binding:
        return self.stalks[self._stalk(name)].source

    def _correspondence(self, name: str) -> int:
        try:
            return self._correspondence_index[name]
        except KeyError as exc:
            raise PhraseMapError(f"unknown phrase correspondence {name!r}") from exc

    def _phrase_digest(self, contributors: tuple[SourceRef, ...]) -> str:
        """Digest the selected states and ordered primary correspondence structure."""
        from rexgraph.io.manifest import manifest_digest

        return manifest_digest({
            "object_type": "RCQLPhrase",
            "stalks": [
                {
                    "name": stalk.name,
                    "source": self._source_manifest(source),
                }
                for stalk, source in zip(self.stalks, contributors, strict=True)
            ],
            "correspondences": [
                {"name": item.name, "stalks": list(item.stalks)}
                for item in self.correspondences
            ],
            "version": 1,
        })

    @classmethod
    def _source_manifest(cls, source: SourceRef) -> dict[str, object]:
        return {
            "name": source.name,
            "state_digest": source.state_digest,
            "policy_digest": source.policy_digest,
            "record_id": source.record_id,
            "record_version": source.record_version,
            "record_as_of": source.record_as_of,
            "record_valid_at": source.record_valid_at,
            "contributors": [cls._source_manifest(item) for item in source.contributors],
        }
