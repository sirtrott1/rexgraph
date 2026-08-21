"""Composing the structural reading with the byte energy."""
import os

import numpy as np

os.environ.setdefault("REXGRAPH_RCDB_URI", "file://" + os.path.expanduser(
    "~/projects/rexgraph/data/corpora/gutenberg/rcdb"))

from rexgraph.partition import (compose_substrates, energy_tensor,
                                grade_leverage, section_tensor)
from rexgraph.sectioning import sectionings_of


def sections_of(rex, layer, *, take=None):
    store = sectionings_of(rex)
    s = store[layer]
    allsec = s.as_sections(store)
    items = [(k, v) for k, v in allsec.items() if v]
    if take is not None:
        items = items[:take]
    return dict(items), s


def main(doc="pg76245", layer="sentence", take=60):
    from agent.rcdb import default_store

    store = default_store()
    rex = store.get(doc)
    rec = store.get_record(doc)
    rex._ensure_clean()
    labels = [str(w) for w in (rec.meta or {}).get("vertex_labels", ())]
    print(f"{doc}: nV={int(rex.nV)} nE={int(rex.nE)} labels={len(labels)}")

    sec, s = sections_of(rex, layer, take=take)
    print(f"sectioning {layer!r}: using {len(sec)} of {len(s.labels)} sections\n")

    # `grade_leverage` is the public accessor and it DELEGATES to the same batch at
    # k=1, but it also generalises to any grade and RETURNS THE RANK, which is the
    # Foster self-test. Calling the private batch meant hand-rolling that check, and I
    # got the weighted form wrong doing it.
    reff, rank = grade_leverage(rex, 1)
    reff = np.asarray(reff)
    print(f"Green's leverage: R_eff in [{reff.min():.4f}, {reff.max():.4f}], "
          f"sum {reff.sum():.2f} against rank {rank}   (Foster, from grade_leverage)")

    T, axes = section_tensor(rex, sec, leverage={1: reff})
    print(f"T{T.shape}  grades={axes['grades']}  readings={axes['readings']}")

    E, moments = energy_tensor(rex, sec, labels, moments=("mean",))
    print(f"E{E.shape}  moments={list(moments)}")

    P = compose_substrates(T, E)
    print(f"P{P.shape}   rank-1 check PASSED (compose_substrates verified it)\n")

    names = axes["sections"]
    rd = axes["readings"]
    mass_i = rd.index("mass") if "mass" in rd else 0

    def col(x):
        return np.asarray(x, float)

    mass = col(T[:, 0, mass_i])
    energy = col(E[:, 0])
    comp = col(P[:, 0, mass_i, 0])

    ok = np.isfinite(mass) & np.isfinite(energy) & np.isfinite(comp)
    print("  the structural reading alone vs the composition")
    print(f"    mass      spread {np.nanmax(mass[ok]) / max(np.nanmin(mass[ok]), 1e-12):8.2f}x")
    print(f"    energy    spread {np.nanmax(energy[ok]) / max(np.nanmin(energy[ok]), 1e-12):8.2f}x")
    print(f"    composed  spread {np.nanmax(comp[ok]) / max(np.nanmin(comp[ok]), 1e-12):8.2f}x")

    order = np.argsort(comp[ok])[::-1]
    idx = np.flatnonzero(ok)
    print("\n  top sections by the COMPOSED reading (mass x energy_mean):")
    for j in order[:6]:
        i = int(idx[j])
        print(f"    {names[i][:28]:<30} mass={mass[i]:8.3f} "
              f"energy={energy[i]:12,.0f} composed={comp[i]:14,.0f}")
    print("\n  and by MASS alone, for comparison:")
    for j in np.argsort(mass[ok])[::-1][:6]:
        i = int(idx[j])
        print(f"    {names[i][:28]:<30} mass={mass[i]:8.3f} "
              f"energy={energy[i]:12,.0f} composed={comp[i]:14,.0f}")

    r = np.corrcoef(mass[ok], energy[ok])[0, 1]
    print(f"\n  mass vs energy correlation: {r:+.3f}")
    print("  (near zero is the point: two substrates carrying different information.")
    print("   Theorem 27 says they must be MULTIPLIED, never mixed, so a low")
    print("   correlation is the precondition for the product meaning anything.)")


if __name__ == "__main__":
    main()
