"""Does contention-aware routing actually beat the obvious policies?

READ THE PARTITION SWEEP FIRST (bench_device_partition). The contended rates this uses
depend on how the machine was divided, and the first version of this benchmark ran the
CPU worker at 16 threads, found co-scheduling lost, and concluded it always does. At
eight threads it wins by 1.09x. One allocation is not a verdict about the hardware.

Two device-pinned bees on one machine (one spawned with -ngl 0 for CPU, one fully
offloaded to the iGPU) and a batch of independent generations to get through. The question
is the MAKESPAN: when does the last one finish.

The prediction comes from agent.device_routing using measured rates; this runs the same
three policies for real and compares. A model that predicts the ordering but not the
magnitudes is still useful for scheduling; one that gets the ordering wrong is not.

Run:  python -m agent.benchmarks.bench_device_routing [n_tasks] [n_tokens]
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from agent.device_routing import DeviceRate, expected_makespan, plan_split

BIN = "/home/art/llama.cpp/build/bin/llama-server"
MODEL = "/home/art/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain what a hash table is, briefly.",
    "Name ten European capitals.",
    "Write a bash loop that renames files.",
    "Describe the water cycle in four sentences.",
    "Explain recursion with one example.",
]


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _post(url, payload, timeout=900):
    req = urllib.request.Request(url + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _launch(ngl, threads, tag):
    port = _free_port()
    args = [BIN, "-m", MODEL, "--port", str(port), "-ngl", str(ngl), "-c", "2048",
            "-np", "1", "--no-webui"]
    if threads:
        args += ["-t", str(threads)]
    log = open(os.path.join("/tmp", f"bench_route_{tag}.log"), "w")  # noqa: SIM115
    # the handle has to outlive this call: the subprocess below writes to it, and a
    # context manager would close it before the server had produced a line.
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(900):
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            return p, url
        except Exception:
            if p.poll() is not None:
                raise RuntimeError(f"{tag} exited") from None
            time.sleep(0.5)
    raise RuntimeError(f"{tag} did not come up")


def _gen(url, i, n_tokens):
    pr = (f"<|im_start|>user\n{PROMPTS[i % len(PROMPTS)]}"
          f"<|im_end|>\n<|im_start|>assistant\n")
    r = _post(url, {"prompt": pr, "n_predict": n_tokens, "temperature": 0.0,
                    "cache_prompt": False})
    return r["timings"]["predicted_per_second"]


def _solo(url, n_tokens):
    _gen(url, 0, 8)
    return max(_gen(url, i, n_tokens) for i in range(2))


def _contended_rates(gurl, curl, n_tokens, window=25.0):
    """Each bee's rate while the OTHER is continuously busy.

    Both sides loop for `window` seconds so neither gets a solo tail, and each reports
    tokens produced over the time it was actually running. A rate measured while the
    other device idles is a solo rate wearing a contended label.
    """
    stop = threading.Event()
    out = {}

    def _loop(key, url):
        toks = 0
        t0 = time.perf_counter()
        i = 0
        while not stop.is_set():
            r = _post(url, {"prompt": f"<|im_start|>user\n{PROMPTS[i % len(PROMPTS)]}"
                                      f"<|im_end|>\n<|im_start|>assistant\n",
                            "n_predict": n_tokens, "temperature": 0.0,
                            "cache_prompt": False})
            toks += r["timings"]["predicted_n"]
            i += 1
        out[key] = toks / max(time.perf_counter() - t0, 1e-9)

    ts = [threading.Thread(target=_loop, args=("g", gurl)),
          threading.Thread(target=_loop, args=("c", curl))]
    [t.start() for t in ts]
    time.sleep(window)
    stop.set()
    [t.join() for t in ts]
    return out


def run(n_tasks=16, n_tokens=120):
    gp, gurl = _launch(999, None, "igpu")
    cp, curl = _launch(0, 16, "cpu")
    try:
        solo_g, solo_c = _solo(gurl, n_tokens), _solo(curl, n_tokens)
        print(f"  solo rates: igpu {solo_g:.2f} tok/s, cpu {solo_c:.2f} tok/s")

        # Contended rates, measured under SUSTAINED mutual load. One generation each is
        # not enough: the iGPU finishes 120 tokens in ~3s and the CPU takes ~7s, so the
        # CPU spends over half its window running alone and its "contended" rate comes
        # out inflated. Planning off that number over-assigns the slow device and makes
        # it the straggler: measured, it turned a predicted 35.8s into 46.5s.
        got = _contended_rates(gurl, curl, n_tokens)
        print(f"  contended : igpu {got['g']:.2f} ({100*got['g']/solo_g:.0f}%), "
              f"cpu {got['c']:.2f} ({100*got['c']/solo_c:.0f}%)  [sustained]")

        rates = [DeviceRate("igpu", "igpu", solo_g, got["g"]),
                 DeviceRate("cpu", "cpu", solo_c, got["c"])]
        urls = {"igpu": gurl, "cpu": curl}

        print(f"\n  {n_tasks} generations of {n_tokens} tokens\n")
        print(f"  {'policy':13s} {'split':16s} {'predicted':>10s} {'measured':>9s} {'vs fastest':>11s}")
        base = None
        for pol in ("fastest", "round_robin", "contention"):
            split = plan_split(rates, n_tasks, pol)
            pred = expected_makespan(rates, split, n_tokens)
            jobs = []
            for name, count in split.items():
                for k in range(count):
                    jobs.append((urls[name], k))
            t0 = time.perf_counter()
            threads = [threading.Thread(target=_gen, args=(u, k, n_tokens))
                       for u, k in jobs]
            [t.start() for t in threads]; [t.join() for t in threads]
            dt = time.perf_counter() - t0
            if base is None:
                base = dt
            shown = "/".join(f"{k}:{v}" for k, v in split.items())
            print(f"  {pol:13s} {shown:16s} {pred:10.2f} {dt:9.2f} {base/dt:10.2f}x",
                  flush=True)
    finally:
        for p in (gp, cp):
            p.terminate()
            try:
                p.wait(timeout=40)
            except Exception:
                p.kill()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 16,
        int(sys.argv[2]) if len(sys.argv) > 2 else 120)
