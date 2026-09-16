"""The current expression contracts and FROM only forms.

This surface is stdlib only, like binding and inference. A row marked implemented
means a current expression contract exists, not authorization on a particular
source. No source objects are inspected. The inventory reports what the language
does, so a name with no contract has no row.
"""
from __future__ import annotations

from .signatures import catalogued, lookup
from .types import ValueKind

# FROM forms have a separate grammar/evaluator. In particular, the legacy REX
# adapter returns a name; it is not an executable typed RETURN expression.
SOURCE_FORMS = {
    "RCDB": "RCDB(name): resolve an explicitly bound RCDB store; never open a URI",
    "REX": "REX(name): resolve an explicitly bound source",
    "CATALOG": "CATALOG(name): resolve an explicitly bound catalog source",
    "FILE": "FILE(catalog_name, entry_name): load an indexed catalog artifact",
    "PHRASE": "PHRASE(section): derive a source from a policy-aware PhraseSheaf",
    "RCDB_GET": "RCDB_GET(store_source, id): resolve the current record",
    "RCDB_VERSION": "RCDB_VERSION(store_source, id, version): resolve a record version",
    "RCDB_AS_OF": "RCDB_AS_OF(store_source, id, time): select transaction time",
    "RCDB_VALID_AT": "RCDB_VALID_AT(store_source, id, time): select valid time",
    "VALID_AT": "VALID_AT(store_source, id, time): existing RCDB valid time selector",
    "TRANSACTION_AT": "TRANSACTION_AT(store_source, id, time): existing RCDB transaction time selector",
    "AT": "AT(temporal_source, version): reconstruct a snapshot by integer index",
    "AT_TIME": "AT_TIME(temporal_source, time): reconstruct a snapshot by clock time",
}


# These are differences, not aliases to be silently rewritten. Further parameter
# variants still require the argument dependent type rule and runtime predicates.
CONTRACT_NOTES = {
    "TURN_FIELD": "Capture an explicit bound Core TurnField as TemporalRex. Half open turn interval, complete prefix snapshots and stable primary turn IDs; no ambient Agent session lookup.",
    "PATH_CHANGE": "Preview text against a copied conversation baseline using the existing numerical Malaugh gate. No append, semantic sufficiency claim or exact path certificate. Undefined entropy is reported separately.",
    "SYMMETRY": "Explicit rational Euclidean chain symmetries generate a subgroup on the bound Rex. Signed words select products, with transpose inverses. No automatic group discovery, weighted metric or G channel preservation is inferred.",
    "TENSOR_MANIFEST": "Alias of TENSORS on a bound FileCatalog: bounded safetensors names, shapes and dtypes through the same Core reader. No tensor payload, inference runtime or implicit file opening outside the catalog. The model argument sketch is realized as a catalog entry name, not a ModelState.",
    "RESOLVENT_GROUP": "Finite words in compatible Euclidean PSD rational resolvents and their inverses. Empty word is identity. Products retain order; no general commutation, finite group order or single parameter closure is claimed. Exact Core solves or the existing numerical resolvent.",
    "VOID": "Exact missing triangular faces of a declared pairwise C1 region. Parallel relations retain multiplicity. Core sparse boundaries and lazy joint filling rank, not an eigenbasis or a general branching void enumeration. No cells are attached.",
    "ALIGN_BY_LINEAGE": "Core sparse scalar C1 alignment over a selected TemporalRex interval. Persisted IDs or exact anonymous support keys; parallel anonymous cells refused. Presence distinguishes missing cells from measured zeros. No inferred coefficient transport, chain map, higher grade identity or cross type correspondence.",
    "VALIDATE_RELATIONS": "Declared sparse grade 2 or higher boundaries on the FROM source. Core checks the original exact chain law, nonzero columns, integrality and storage compatibility separately. No prediction, normalization, independence test or attachment; primary C1 proposals use a separate incidence grammar.",
    "QUOTIENT": "Core exact relative chain tower C(X)/C(A) for a declared cell selection and its full subcomplex closure. Original branching shares, sparse certified projection and lazy exact homology. Coordinate complex, not a renormalized Rex or an extra collapsed point. Materialize the projection before binding it as a graded map parameter.",
    "APPLY_DELTA": "Core canonical MutationPackage bytes applied against the bound Rex and explicit parent link. Full owned result, not a C1 temporal signal replay. No store writes; signature policy and providers are checked at execution.",
    "REPLICATE": "Core ReplicationPackage bytes replayed against a materialized bound checkpoint and explicit parent commit. Core verifies its state identity, ordered mutations and terminal identity. Returns an owned Rex, including an empty stream; no implicit target store or network transfer.",
    "ACCESSION_DELTA": "Core exact sparse A_new J - J A_old for ambient accessions; A_new J - A_old for identical named output coordinates. Explicit current endpoint declarations, no inferred correspondence or time division.",
    "TEMPORAL_DELTA": "Step is an interior transition index, 1 through T-1, not a grade. The C1 signal is not a full mutation package.",
    "MARKOV_VIEW": "Native C0 participation action and observable transpose through abs(B1) W; exact Q action and uniform dangling columns. No execution policy switch or pair expansion.",
    "TEXT_OVERLAP_VIEW": "Off diagonal unsigned weighted primary Gram action on original C1 relations; exact Q action and symmetric transpose, not generally PSD or stochastic. Sections remain stored partitions, not invented sentence cells.",
    "PAGERANK": "Numerical fixed point of an explicit MARKOV_VIEW with uniform or supplied C0 restart, measured L1 error bound and refusal on exhaustion; no exact fixed point claim.",
    "CAYLEY": "Factored (I-tA)^-1(I+tA) for a canonical Q Euclidean skew action; exact positive solves or the existing numerical resolvent, not an exponential.",
    "COMPLEX_STRUCTURE": "Streamed Q certificate J^3=-J with P=-J^2; only a single rational frequency normalization, not spectral plane selection or the SVD diagnostic.",
    "RATIONAL_ROTATION": "Exact Pythagorean rotation on a certified partial complex support, with identity on its real kernel.",
    "DOCUMENT_FIELD": "C1 mass or coverage from explicit C0 seed cells, using Core sparse rational accumulation; not a text parser, corpus selector or pairwise similarity projection.",
    "SECTION_RESPONSE": "Mass or coverage summed over one stored disjoint C1 section layer; exact scores or a single final numerical rounding. No inferred owner for covers.",
    "SEMANTIC_CLOSURE": "Same explicit C0 structural expansion as CLOSURE; retains contained upper cells and isolated vertices. Repeated structural readings are not a semantic sufficiency certificate.",
    "DIFF": "Exact C1 boundary difference against the bound source on a complete vertex union. Auto uses stable relation IDs when both endpoints carry them; anonymous matching retains primary slot multiplicities. This is not a Rex mutation or a full state delta; numerical matrix export is explicit.",
    "FIELD_DELTA": "Exact B'J-JB and transpose boundary defects applied to a canonical Q Chain, with explicit GradedMap correspondence and identity endpoint metrics. Returns named target coordinate records, not fields on the old Rex. No lineage or time correspondence is inferred.",
    "FIELD_DELTA_MOMENT": "Sum of down and up correspondence defect quadrances in identity coordinate metrics, over all supplied block columns. Not simple field subtraction or a derivative.",
    "ORIENTED_FIELD_DELTA_MOMENT": "Down defect quadrance minus up defect quadrance, with FIELD_DELTA's explicit correspondence and identity metric scope. May be negative. A chain map alone does not imply a zero up defect.",
    "EXPORT_PARQUET": "Explicit column arrays and RexPartition lineage, encoded by the Core exporter in memory. Returns payload, manifest and digest. Schema is measured from arrays, not inferred from Rex cells. Lineage attribution is supplied by the caller, not a derivation proof. Optional PyArrow; use ENCRYPT on returned bytes separately.",
    "VALID_AT": "FROM only, with an explicit store and record identity. Same reader and policy as RCDB_VALID_AT; does not interpret the TemporalRex clock as valid time.",
    "TRANSACTION_AT": "FROM only, with an explicit store and record identity. Same reader and policy as RCDB_AS_OF; does not invent a transaction axis for an unstored Rex.",
    "FILL": "One declared nonzero integral C1 Chain, checked over the original Q boundary. Returns an owned Rex with one C2 cell, preserving prior state and extending B3 by a zero row. No inferred candidate, primitive rescaling or automatic store commit.",
    "HARMONIC_SHADOW": "Exact C1 cycle dimensions before and after Hodge eligible C2 attachment and rank(B2). No eigenvectors or chosen cycle basis. Uses all carried grades for the chain check.",
    "HASH": "Explicit state, bytes, canonical JSON manifest or Core lineage digest. No implicit pickle, float conversion or interchangeable digest domains.",
    "MANIFEST": "Public description of a Core partition, commit, transition, export manifest or native byte envelope. Header inspection does not authenticate payloads or verify signatures.",
    "LINEAGE": "Existing Core partition, commit link or transition lineage. No inferred ancestry, history traversal or signature verification.",
    "TRANSPORT": "Core framing of explicit bytes and public JSON metadata. Returns in memory bytes; does not send, copy records, serialize a Rex or write files.",
    "SHOW_CAPABILITIES": "Current bound SourcePolicy permissions, projected record fields and digest. Does not disclose providers or claim that every permitted operation is installed.",
    "ENCRYPT": "Core authenticated byte envelope using an explicitly configured key provider. Random nonce, not memoized. Requires rexgraph security extra.",
    "DECRYPT": "Core authenticated opening of a byte envelope with configured providers. Returns original bytes, not deserialized objects. Authentication failure propagates.",
    "SIGN": "Core signer provider applied to explicit bytes. A signature of bytes is not an inferred lineage signature; use the Core canonical signing bytes when required.",
    "VERIFY_SIGNATURE": "Core verifier provider and explicit signer identity. Returns false for invalid signatures; missing providers and malformed inputs remain errors.",
    "PSEUDONYMIZE": "Core scoped pseudonym for an explicit text value, scope and key identity. Does not deidentify a Rex or remove contextual identifying information.",
    "FACES": "Stored C2 cells with a nonempty boundary wholly inside the C1 Cell or CellSet. The original exact chain law is required. No new faces are inferred.",
    "RESTRICT": "One Cell or CellSet at any carried grade, with subcomplex closure. Returns an owned Rex retaining full arity, relation identities and metrics, without application metadata. Rebind it before querying its new basis.",
    "PARTITION": "The same explicit restriction with source, result, requested selection and bound policy digests plus source index maps at every grade. One RexPartition per call, not automatic clustering. Lineage is not an authorization grant.",
    "SIMPLE_HOMOLOGY": "Exact dimension of H_k(X/W) over Q. Identify nonzero B_k columns equal up to sign at one declared grade, project the upper map, and retain distinct zero columns. Not a harmonic basis or a simultaneous quotient at all grades.",
    "MULTIPLICITY_HOMOLOGY": "Exact dimension of ker(H_k(X) -> H_k(X/W)), with the same declared quotient as SIMPLE_HOMOLOGY. Filled multiplicity cycles do not count. The two dimensions sum to BETTI(k).",
    "COLUMN_EXPANSION": "Exact P and Lambda coordinates of BOUNDARY(grade), including witness terms and zero primary columns. Internal legs never become cells. The full source boundary state stays bound.",
    "PRIMARY_LIFT": "Reconstruct B=P Lambda from the original factors of one expansion, with exact Q and compiled sparse numerical actions. This is not a new Rex or an inferred metric.",
    "HYPERSLICE": "Immediate below, above and lateral CellSets across every carried grade. C1 uses full primary participant slots, including cancelling loops. C0 has no below grade; top above is empty. Source types and time are retained, not inferred.",
    "ADJUGATE": "Polynomial adj(A) on a canonical square operator with a certified Q action, including singular A. Coefficients use streamed exact power traces; each application costs n(n-1) coefficient actions. No inverse or eigenbasis.",
    "HOMOTOPY": "Exact full tower certificate for G-F=BH+HB between explicit graded maps, with supplied Q witness and empty top component. Checks both chain maps; never infers a witness or a target boundary.",
    "SIGMA_OPERATOR": "Explicit exponential C0 family a_v exp(r_v sigma), unit boundary columns before fixed relation weights, selected trace normalized C1 channels. G is raw and C share/count is explicit; numerical full action, not a rational or higher tower claim.",
    "CRITICAL_COMMUTATOR": "Factored [R(sigma),R(1-sigma)] for the supplied sigma family. Its zero at the center is automatic, not a criticality certificate.",
    "CRITICAL_RATE": "Analytic [R,R'] at sigma=1/2 by default; reading=slope returns -2[R,R'], the derivative of the involution commutator. No finite difference or inferred weighting family.",
    "DELTA": "C1 transition at an explicit step, as TEMPORAL_DELTA. Higher grade alignment is not inferred; numerical amplitude fields remain separate from exact stored event coefficients.",
    "EXISTENCE_DELTA": "Sparse C1 birth/death record over union identities, including deleted relations; not a field on the current cell basis.",
    "SIGNING_DELTA": "Difference of public SIGNING bits on persisting relations, opposite to the older signed polarity increment. Birth/death are separate existence events.",
    "ORIENTATION_DELTA": "Full distinguished participant mask difference over union C1 identities, as HEAD_DELTA; does not collapse branching head motion to the older polarity bit.",
    "HEAD_DELTA": "Sparse full head mask difference; witnesses and cancelling loops have no distinguished negative head.",
    "STRUCTURAL_DELTA": "Full exact B1 column difference over union C1 identities; not the aggregate C0 SIGNAL_SOURCE and not a higher grade temporal defect.",
    "METRIC_DELTA": "Exact stored C1 metric difference with absent relations zero and unweighted relations one; float inputs retain their binary value.",
    "EXISTENCE_HISTORY": "Sparse C1 presence over [start,stop) steps, keyed by explicit relation ID or anonymous exact support; no dense time by cell grid.",
    "BIOES": "Sparse C1 lifetime labels with O implicit. Adjacent source steps determine labels at clipped window boundaries; not the mixed grade topology phase labels.",
    "BETWEEN": "Closed interval on step or rex_time, preserving clocks and full snapshot attribution; valid and transaction axes are refused.",
    "TEMPORAL": "Capture the full bound Rex state as one TemporalRex snapshot through append_snapshot; not the connectivity only constructor.",
    "CHAIN": "Ordered native RexOperator factors with matching intermediate grade, basis and variance; not a vertex walk. Type coordinate paths require separate accessions.",
    "TRANSFER": "Apply a CHAIN to a typed carrier, optionally pair with an explicit target in its codomain; exact Q is the default.",
    "DEPENDENCE": "Exact sparse relations among an explicit ordered carrier family; kernel coordinates index input members, not cells.",
    "STRAIN": "B_k W_k B_(k+1) for an explicit positive diagonal metric; a factored Chain map with a rectangular exact zero for a missing upper grade.",
    "METRIC_HOMOTOPY": "Convex positive diagonal metric segment; parameter in [0,1], exact for Q inputs. No general metric deformation is inferred.",
    "CROSS_METRIC": "Explicit ordered sparse block between two type accession coordinate spaces; neither symmetry nor positivity is inferred.",
    "ACTION": "One half <x,A x> minus <j,x> in the operator's declared self adjoint form; A must include any desired regularization.",
    "VARIATION": "Gradient A x-j in ACTION's form, or pairing with an explicit direction; not symbolic differentiation of an evaluated scalar.",
    "DIFFERENTIAL": "Directional derivative 2 <eta,L_k x> of the complete Hodge moment on real Chains with identity endpoint metrics.",
    "COUNT": "Length of an explicit finite sequence or native CellSet; no implicit subcomplex or distinct projection.",
    "SUM": "Explicit real scalar sequences only; integers use arbitrary precision, rationals remain exact, numerical values use compensated summation. Empty sum is zero.",
    "MEAN": "Explicit real scalar sequences only; exact integer and rational input yields a Fraction. Empty mean is None, not zero.",
    "COMMUTATOR": "Factored AB-BA on one canonical grade/variance or the full Chain tower; exact only when both actions are certified; no implicit graded sign or PSD claim.",
    "ANTICOMMUTATOR": "Factored AB+BA on the same declared operator space; positivity and vanishing are not inferred from operand names.",
    "REX": "FROM resolver only; the legacy direct adapter returns the bound name.",
    "GRADE": "Returns source maximum grade without arguments, or the supplied value's grade.",
    "BOUNDARY": "Lowers Chains (not Cochains); also accepts primary Cell/CellSet values.",
    "CELL": "Current inputs are ordered grade and index, not a general identity.",
    "CELLS": "Requires a grade and optional ordered indices, not a type filter.",
    "ARITY": "Reads one primary C1 relation or its CompositeBinary carrier.",
    "HEAD": "Returns the exact C0 Chain head mask, not a Cell.",
    "STAR": "Returns a GradedCellPattern; grades are not flattened into a CellSet.",
    "SHARE": "Returns a C0 Chain with exact rational share coefficients.",
    "CORELATIONS": "Immediate upper-cell incidence; not the planned CO_RELATE metric operation.",
    "ACCESS": "Apply an explicit sparse TypeAccession from cells to ambient or named rectangular type coordinates; no implicit registry or projector claim.",
    "ACCESS_TYPES": "Apply a declared ordered AccessionFamily, preserving overlapping views and the distinct type axis.",
    "TYPE_VIEW": "Explicit TypeAccession application with the same contract as ACCESS; not implicit name lookup.",
    "SUPPORT": "Lower boundary support; a Cell retains its primary participants, while a Chain retains the nonzero support after exact cancellation. Positive grades only.",
    "DEGREE": "Primary upward incidence count. A self loop counts once at its participant even though its boundary cancels.",
    "FIELD": "Wrap an already bound Cochain or Field. Does not infer a basis for arrays or identify Chain and Cochain variance.",
    "GRAM": "Ordered Gram block of an explicit same-space family; exact by default. An empty family has a zero by zero Gram block.",
    "GRAM_RANK": "Exact span rank using native integer column elimination. A validated positive diagonal metric does not change rank; no Gram materialization or eigenvalues.",
    "GRADIENT": "Numerical C1 down Hodge projection, not the boundary transpose acting on a C0 potential.",
    "CURL": "Numerical C1 upper Hodge projection, not the G channel.",
    "ORIENTED_MOMENT": "Chain contraction with HODGE_DIFFERENCE in identity endpoint metrics; exact by default.",
    "COBOUNDARY_MOMENT": "Chain contraction with HODGE_UP in identity endpoint metrics; exact by default.",
    "FIELD_QUOTIENT": "Projection coefficient from down or up Chain boundary moments in identity metrics; rejects a zero denominator.",
    "COFIELD_QUOTIENT": "Up sector FIELD_QUOTIENT; no implicit identification of Chains and Cochains.",
    "RATE": "Finite scalar or typed coefficient delta divided by a nonzero real interval; integer division remains rational.",
    "TEMPORAL_RATE": "Finite RATE of an explicitly supplied delta, not a history lookup or a choice of clock axis.",
    "MOMENT_RATE": "Finite RATE of an explicitly supplied moment delta, not differentiation of an observation.",
    "MASS": "Sum of absolute real coefficients over the explicit vector or block; distinct from cell count, quadrance and Hodge trace.",
    "ARGMIN": "Select from homogeneous typed values using an equally long scalar measure sequence; first occurrence wins ties; empty input is refused.",
    "ARGMAX": "Select from homogeneous typed values using an equally long scalar measure sequence; first occurrence wins ties; empty input is refused.",
    "TRACE": "Global order one SCALE_MOMENT of an explicit symmetric real operator; exact mode currently requires a channel diagonal.",
    "TYPES": "Ordered names from a declared AccessionFamily, not a global or inferred type registry.",
    "CURVATURE": "Explicit C1 relation metric curvature, with the same reading as METRIC_CURVATURE; not RCFE curvature or a generic curvature tensor.",
    "WEIGHT": "Declared C1 relation metric; exact mode preserves supplied rational weights or the exact stored binary float. A field input selects its grade, not a multiplication by its values.",
    "ORIENTATION": "Primary C1 distinguished participant mask, as HEAD; witnesses and self loops have no head and return zero masks.",
    "SIGNING": "C1 declared gauge sign encoded as a bit: 0 for +1, 1 for -1. Independent of head orientation and metric weight.",
    "PARITY": "Product of declared C1 gauge signs on an explicit selection. Does not claim a cycle or upper grade orientation holonomy; an explicit sequence retains repeated visits.",
    "CHAIN_VALID": "Checks declared C2 validity and the complete carried higher boundary tower over Q. No numerical tolerance or inference from the Hodge face filter alone.",
    "GREEN_OPERATOR": "Numerical (I + alpha L_k)^-1 on an explicit grade with identity metrics; separate from GREEN's C0 pseudoinverse. Uses the existing native resolvent with residual checks.",
    "GREEN_FIELD": "Inject a Cell or CellSet indicator, or a bound Cochain, then apply the regularized grade Green action. alpha defaults to 1.",
    "GREEN_GRAM": "Source pairings x_i* G x_j, not inner products of two resolved fields. Explicit same-grade source vectors; default Green action is (I + L_k)^-1.",
    "GREEN_SPREAD": "Projective separation from GREEN_GRAM; zero quadrance returns None, not zero distance.",
    "CO_RELATE": "Ordered signed/sesquilinear TypeView pairing with sparse cross block, coherent factored FamilyMetric or ambient diagonal metric; CORELATIONS is not an alias.",
    "MOMENT_TENSOR": "TypedFamily Gram with ambient diagonal metric or explicit sparse realizations through a coherent FamilyMetric; independent cross blocks are not a family metric.",
    "CHAIN_MAP": "CHAIN_MAP(declaration) verifies a full source-bound exact GradedMap with explicit target boundaries; both chain laws and every square checked. Core composition is explicit; target inference and homotopies remain planned.",
    "GREEN": "Current C0 pseudoinverse action, optionally applied; not a general resolvent.",
    "RESOLVENT": "Numerical (I + alpha L)^-1 for declared real Euclidean PSD operators or native metric-PSD Hodge down/up/sum; alpha defaults to 1, with explicit tol/maxiter and original-system residual checks.",
    "GREEN_SOLVE": "Apply an explicit Green action using the same checked path as APPLY, without materializing an inverse.",
    "METRIC": "Identity at the declared grade, or an explicit positive diagonal Cochain; not a type registry or implicit relation weight.",
    "MOMENT": "Signed/Hermitian contraction, summed over block columns, with optional diagonal metric and exact flag.",
    "INTEGRATE": "Bilinear Cochain/Chain dual evaluation, summed over block columns; no metric or conjugation; explicit exact Q mode.",
    "QUADRANCE": "Optional third argument supplies a typed diagonal metric; the existing second exact flag is retained.",
    "HODGE": "Current numerical grade-1 cochain split.",
    "HARMONIC": "Current numerical grade-1 cochain harmonic projection.",
    "HODGE_COORDS": "Current numerical grade-1 Hodge coordinates in the native frame.",
    "WINDING": "C1 harmonic frame circulation. Integer and rational inputs retain exact arithmetic; numerical inputs remain numerical. This is not a normalized number of turns.",
    "APPLY": "Typed adjoint/Hodge/bracket actions retain Chain or Cochain variance; full-tower Dirac/brackets require GradedChain. Exact actions require certified factors; Green solves remain numerical and preserve their declared variance.",
    "DIRAC": "Full carried Chain tower: boundary plus positive-diagonal metric adjoint, identity metrics where omitted; no implicit grade/type restriction.",
    "ANTI_DIRAC": "Full carried Chain tower: boundary minus positive-diagonal metric adjoint; metric skew-adjoint, not PSD; exact supported Q factors.",
    "GRADED_CHAIN": "Unique canonical Chain seed components, common block shape; omitted carried grades are zero; never relabel cochains as Chains.",
    "GRADE_COMPONENT": "Project one carried Chain grade of a GradedChain without losing source/basis identity.",
    "CHARACTER": "Returns a character record; exact mode supports rational normalized-G diagonals.",
    "CHANNEL": "Bound C1 T/G/F/C before trace normalization; G follows source selection, F always references raw G. Exact full action/transpose for T/raw-G/F/C; normalized G has exact diagonal only. APPLY retains Field variance and defaults to numerical arithmetic.",
    "STAR_CHARACTER": "One C0 cell's incident-edge character mean, optionally exact; not the Green vertex character phi.",
    "SCALE_MOMENT": "Symmetric real operator trace or diagonal at nonnegative integer order; exact mode supports order 0 or channel order 1.",
    "CHARACTER_ENERGY": "Operator-only local diag(A squared), not squared character; sparse row quadrance, numerical arithmetic.",
    "GLUE": "ExactSheaf compatibility/obstruction reading on one Rex; not a multi-Rex mutation.",
    "SECTION_CHECK": "Compact exact incidence compatibility for rectangular stalk maps; anchor residuals, not all failed pairs or cohomology.",
    "HASH_FILES": "Returns the number of hashed catalog entries, not a hash map.",
    "METRIC_CURVATURE": "C1 relation-metric local deviation; not arbitrary tensor strain.",
    "CLOSURE": "Current grade-0 seed expansion; higher-grade closure is refused.",
    "SHOW_OPERATORS": "Global static inventory with limit/offset; not a capability-filtered list.",
    "SPREAD": "Returns None when either operand has zero quadrance; optional fourth argument supplies a typed diagonal metric.",
}


def _contract(signature):
    from .arguments import EXPRESSION_ARGUMENTS
    kinds = signature.source_kind
    if not isinstance(kinds, tuple):
        kinds = (kinds,)
    result = signature.result
    if callable(result):
        returns = {"depends_on_arguments": True}
    else:
        returns = {
            "kind": result.kind.value,
            "grade": result.grade,
            "domain": None if result.domain is None else result.domain.value,
            "exactness": None if result.exactness is None else result.exactness.value,
            "variance": None if result.variance is None else result.variance.value,
        }
    return {
        "parameter_names": list(EXPRESSION_ARGUMENTS.get(signature.name, ((), ()))[0]),
        "source_kinds": ["any"] if ValueKind.UNKNOWN in kinds else [k.value for k in kinds],
        "inputs": [pattern.describe() for pattern in signature.inputs],
        "arity": list(signature.arity),
        "result": returns,
        "requires": sorted(signature.requires),
        "source_methods": sorted(signature.source_methods),
        "effects": sorted(effect.value for effect in signature.effects),
        "memoizable": signature.memoizable,
        "implementation_key": signature.implementation_key,
        "preconditions": list(signature.preconditions),
        "refusal": signature.unreachable or None,
    }


def operator_inventory(*, limit: int = 1000, offset: int = 0) -> list[dict[str, object]]:
    """Return name sorted detached rows; never infer readiness from historical labels.

    Limit/offset bound the rendered result. This is the global language inventory,
    not source data, not an authorization grant, and not a physical execution plan.
    Argument dependent results must still be inferred for the actual call. An
    implementation key names the adapter contract, not necessarily its kernel.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 1000:
        raise ValueError("operator inventory limit must be an integer in [0, 1000]")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("operator inventory offset must be a nonnegative integer")
    current = catalogued()
    rows = []
    for name in sorted(current | SOURCE_FORMS.keys())[offset:offset + limit]:
        signature = lookup(name) if name in current else None
        status = ("refused" if signature.unreachable else "implemented") if signature else "source-only"
        roles = (["expression"] if signature else []) + (["source"] if name in SOURCE_FORMS else [])
        rows.append({
            "name": name,
            "status": status,
            "roles": roles,
            "current": None if signature is None else _contract(signature),
            "source_form": SOURCE_FORMS.get(name),
            "note": CONTRACT_NOTES.get(name),
        })
    return rows
