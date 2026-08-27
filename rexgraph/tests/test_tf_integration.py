"""TensorFlow runs the same relational operator torch does, in its own process.

The arithmetic is a property of the complex, not of the framework, so the two
implementations must agree on the same weights. `from_torch` moves the parameters
across so the comparison is of the operator and not of two random initialisations.

WHY A SUBPROCESS. TensorFlow and this ROCm torch cannot share an interpreter:

  * import order alone is fatal: torch then tensorflow aborts outright with
    "LLVM ERROR: inconsistency in registered CommandLine options", which is an
    abort and not an exception, so nothing can catch it; and
  * even with tensorflow imported FIRST, a later ROCm GPU dispatch segfaults
    (measured: rexgraph/tests/test_gpu_dispatch.py exits -11 with tf resident and
    passes without it).

So the comparison runs in a child that imports tensorflow before torch and stays on
CPU, and the parent never touches tf at all. This is the integration working within
a real constraint rather than a workaround for a bug in rexgraph.
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

# find_spec, NOT importorskip: importing tensorflow into THIS process is exactly
# what must not happen. A later ROCm GPU dispatch segfaults with tf resident
# (test_gpu_dispatch exits -11), and collection imports every test module before
# any test runs, so an importorskip here would take the GPU tests down.
if importlib.util.find_spec("tensorflow") is None:      # pragma: no cover
    pytest.skip("tensorflow not installed", allow_module_level=True)


def _in_child(body: str, timeout: int = 300):
    """Run `body` with tensorflow imported first and the GPU hidden; parse its JSON."""
    src = "import tensorflow as tf\nimport torch\nimport numpy as np, json\n" + \
        textwrap.dedent(body)
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
               ROCR_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       env=env, timeout=timeout)
    assert r.returncode == 0, f"child failed ({r.returncode}):\n{r.stderr[-2000:]}"
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    assert line, f"no JSON from child:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    return json.loads(line[-1])


def test_tf_matches_torch_on_the_same_weights():
    out = _in_child("""
        from rexgraph.nn import tf_relational as tfr
        from rexgraph.nn.relational_attention import CausalPropagatorAttention
        res = {}
        for T, w, hops in ((64, 16, 4), (48, 8, 2), (32, 64, 3), (7, 4, 1)):
            torch.manual_seed(0)
            src = CausalPropagatorAttention(32, 4, hops=hops, window=w, sparse=True)
            layer = tfr.from_torch(src)
            x = np.random.default_rng(0).standard_normal((2, T, 32)).astype(np.float32)
            with torch.no_grad():
                a, _ = src(torch.from_numpy(x))
                dense, _ = src.forward_dense(torch.from_numpy(x))
            b = layer(tf.constant(x)).numpy()
            res[f"{T}-{w}-{hops}"] = [float(np.abs(a.numpy() - b).max()),
                                      float(np.abs(dense.numpy() - b).max()),
                                      list(b.shape)]
        print(json.dumps(res))
    """)
    assert out
    for key, (vs_band, vs_dense, shape) in out.items():
        assert vs_band < 1e-4, (key, vs_band)
        assert vs_dense < 1e-4, (key, vs_dense)   # and the dense path too
        assert shape[0] == 2 and shape[2] == 32


def test_the_tf_layer_trains():
    """It has to carry gradients, or it is a calculator and not a layer."""
    out = _in_child("""
        from rexgraph.nn import tf_relational as tfr
        layer = tfr.CausalPropagatorAttention(32, 4, hops=2, window=8)
        x = tf.constant(np.random.default_rng(2).standard_normal((2, 24, 32)).astype("float32"))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.square(layer(x)))
        g = tape.gradient(loss, layer.trainable_variables)
        print(json.dumps({"n_vars": len(layer.trainable_variables),
                          "all_present": all(v is not None for v in g),
                          "max_abs": max(float(tf.reduce_max(tf.abs(v))) for v in g)}))
    """)
    assert out["n_vars"] > 0 and out["all_present"] and out["max_abs"] > 0


def test_the_window_view_addresses_the_right_tokens():
    out = _in_child("""
        from rexgraph.nn import tf_relational as tfr
        z = tf.reshape(tf.range(6 * 2, dtype=tf.float32), [1, 1, 6, 2])
        w = 3
        win = tfr.causal_windows(z, w).numpy()
        valid = tfr.band_valid(6, w).numpy()
        ok = True
        for i in range(6):
            for m in range(w):
                j = i - w + 1 + m
                ok &= bool(valid[i, m]) == (j >= 0)
                if j >= 0:
                    ok &= bool(np.array_equal(win[0, 0, i, m], z.numpy()[0, 0, j]))
        print(json.dumps({"ok": bool(ok), "shape": list(win.shape)}))
    """)
    assert out["ok"] and out["shape"] == [1, 1, 6, 3, 2]


def test_a_tf_tensor_reads_through_the_rexgraph_surface():
    """The other direction: a TF tensor as an edge signal, so a TF model can take an
    exact structural feature without converting by hand."""
    out = _in_child("""
        import rexgraph as rg
        from rexgraph.graph import RexGraph
        r = RexGraph(sources=np.array([0, 1, 2], np.int32),
                     targets=np.array([1, 2, 0], np.int32))
        r._ensure_clean()
        w_tf = r.harmonic_winding(tf.constant([3.0, -1.0, 4.0]))
        w_np = r.harmonic_winding(np.array([3.0, -1.0, 4.0]))
        try:
            r.harmonic_winding(tf.constant([1.0, 2.0]))
            msg = ""
        except ValueError as e:
            msg = str(e)
        print(json.dumps({"tf": np.asarray(w_tf).tolist(),
                          "np": np.asarray(w_np).tolist(),
                          "integer": bool(np.issubdtype(np.asarray(w_tf).dtype, np.integer)),
                          "err": msg}))
    """)
    assert out["tf"] == out["np"] and out["integer"]
    assert "Expected 3 values for the edge flow, got 2" in out["err"]


def test_importing_tf_after_torch_raises_instead_of_aborting():
    """The failure mode that matters most: in a torch-first process the module must
    refuse in a way the caller can see, since the alternative is a bare abort."""
    src = ("import torch\n"
           "from rexgraph.nn import tf_relational as tfr\n"
           "try:\n"
           "    tfr.CausalPropagatorAttention(32, 4, window=8)\n"
           "    print('NO_ERROR')\n"
           "except ImportError as e:\n"
           "    print('RAISED' if 'aborts the process' in str(e) else 'WRONG')\n")
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3"), timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "RAISED" in r.stdout, r.stdout
