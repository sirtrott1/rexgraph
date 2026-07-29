"""Slice D dogfood: RexGraph learning online from its own RCDB, closing the loop.

Domain-agnostic and dataset-free: a synthetic complex grows and reshapes over a
handful of steps; every external put drives the online Green's-cochain field one
step and (with write_back) persists a guarded derived version. Run:

    micromamba run -n rexgraph python agent/examples/online_loop_demo.py
"""
import numpy as np

from rexgraph.graph import RexGraph
from agent.rcdb import open_store
from agent.temporal_loop import OnlineLoop, ChangeSource


def main() -> None:
    store = open_store("memory://")
    loop = OnlineLoop(store, write_back=True)
    src = ChangeSource(store)
    loop.run_stream(src)

    seqs = [([0, 1], [1, 2]), ([0, 1, 2], [1, 2, 3]),
            ([0, 2], [1, 3]), ([0, 2, 3], [1, 3, 4])]   # step 3 removes edge (1,2)
    for s, t in seqs:
        store.put("svc", RexGraph(sources=np.asarray(s, np.int32),
                                  targets=np.asarray(t, np.int32)))
    src.stop()

    processed = [r for r in loop.history() if r.t >= 0]
    print("external puts:", len(seqs))
    print("events processed (no derived re-entry):", len(processed))
    for r in processed:
        err = r.learn and r.learn.get("error")
        print("  t=%d id=%s v=%d error=%s wrote_back=%s"
              % (r.t, r.id, r.version, err, r.wrote_back))
    print("derived versions of svc::online:",
          sum(1 for _ in range(len(seqs)) if store.get_version("svc::online", _ + 1)))
    store.close()


if __name__ == "__main__":
    main()
