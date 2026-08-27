"""Is the CPU+iGPU loss physical, or just a bad partition?

Run:  python -m agent.benchmarks.bench_device_partition

The first measurement gave the CPU bee -t 16 with no pinning while the iGPU bee also
needs host threads for sampling and the CPU-side ops. A 38% collapse looks like
oversubscription, not a bus limit. So sweep the partition instead of concluding from one
point: CPU thread count, and whether the two are pinned to disjoint cores.
"""
import json
import os
import socket
import subprocess
import threading
import time
import urllib.request

BIN="/home/art/llama.cpp/build/bin/llama-server"
M="/home/art/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
S=os.path.dirname(os.path.abspath(__file__))
NCPU=os.cpu_count()
PR=["Write a Python function that merges two sorted lists.",
    "Explain what a hash table is, briefly.",
    "Name ten European capitals."]
def fp():
    s=socket.socket(); s.bind(("127.0.0.1",0)); p=s.getsockname()[1]; s.close(); return p
def post(u,pl,t=900):
    r=urllib.request.Request(u+"/completion",data=json.dumps(pl).encode(),
                             headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=t) as x: return json.loads(x.read())
def launch(ngl, threads, tag, cores=None):
    port=fp()
    args=[BIN,"-m",M,"--port",str(port),"-ngl",str(ngl),"-c","2048","-np","1","--no-webui"]
    if threads: args+=["-t",str(threads)]
    if cores: args=["taskset","-c",cores]+args
    # the handle outlives this call on purpose: the subprocess writes to it, and a
    # context manager would close it before the server produced a line.
    log = open(f"{S}/pt_{tag}.log", "w")  # noqa: SIM115
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
    u=f"http://127.0.0.1:{port}"
    for _ in range(900):
        try: urllib.request.urlopen(u+"/health",timeout=2); return p,u
        except Exception:
            if p.poll() is not None: raise RuntimeError(f"{tag} exited") from None
            time.sleep(0.5)
    raise RuntimeError("timeout")
def sustained(urls, window=18.0, n=120):
    stop=threading.Event(); out={}
    def loop(k,u):
        tk=0; t0=time.perf_counter(); i=0
        while not stop.is_set():
            r=post(u,{"prompt":f"<|im_start|>user\n{PR[i%len(PR)]}<|im_end|>\n<|im_start|>assistant\n",
                      "n_predict":n,"temperature":0.0,"cache_prompt":False})
            tk+=r["timings"]["predicted_n"]; i+=1
        out[k]=tk/max(time.perf_counter()-t0,1e-9)
    ts=[threading.Thread(target=loop,args=(k,u)) for k,u in urls.items()]
    [t.start() for t in ts]; time.sleep(window); stop.set(); [t.join() for t in ts]
    return out

def main():
    print(f"  {NCPU} logical CPUs\n")
    gp,gu=launch(999,None,"g0")
    try:
        solo=sustained({"g":gu},window=12.0)["g"]
        print(f"  iGPU alone (sustained): {solo:.2f} tok/s\n")
    finally:
        gp.terminate(); gp.wait(timeout=40)

    print(f"  {'cpu -t':>7s} {'pinned':>8s} {'igpu':>8s} {'cpu':>7s} {'total':>8s} {'vs solo':>8s}")
    best=(None,0.0)
    for threads, cores_c, cores_g, tag in (
            (2,  None, None, "t2"),
            (4,  None, None, "t4"),
            (8,  None, None, "t8"),
            (16, None, None, "t16"),
            (8,  "0-7", f"8-{NCPU-1}", "t8pin"),
            (4,  "0-3", f"4-{NCPU-1}", "t4pin"),
            (12, "0-11", f"12-{NCPU-1}", "t12pin")):
        gp,gu=launch(999,None,f"g_{tag}",cores=cores_g)
        cp,cu=launch(0,threads,f"c_{tag}",cores=cores_c)
        try:
            r=sustained({"g":gu,"c":cu})
            tot=r["g"]+r["c"]
            if tot>best[1]: best=(tag,tot)
            print(f"  {threads:7d} {str(bool(cores_c)):>8s} {r['g']:8.2f} {r['c']:7.2f} "
                  f"{tot:8.2f} {tot/solo:7.2f}x", flush=True)
        finally:
            for p in (gp,cp):
                p.terminate()
                try: p.wait(timeout=40)
                except Exception: p.kill()
    print(f"\n  best: {best[0]} at {best[1]:.2f} tok/s  ({best[1]/solo:.2f}x the iGPU alone)")

if __name__ == "__main__":
    main()

