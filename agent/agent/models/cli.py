"""CLI for the model-builder framework.

    python -m models list
    python -m models build --archetype cnn --set norm=false --optimizer hodge --steps 300
    python -m models build --archetype mlp --data mydata.csv --optimizer hodge
    python -m models build --archetype lm  --set attention=standard
    python -m models multistep --archetype mlp --stage optimizer=adam,steps=100 --stage optimizer=hodge,steps=200
    python -m models fusion --spec mlp --spec mlp:d_hid=64 --fusion ensemble
"""
import argparse
import json

from . import build, list_archetypes, run


def _kv(pairs):
    out = {}
    for kv in pairs or []:
        k, _, v = kv.partition("=")
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="models", description="Build/train models on the rexgraph.nn substrate.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the selector - archetypes, use-cases, params")
    for name in ("build", "multistep", "fusion"):
        p = sub.add_parser(name)
        p.add_argument("--archetype", "-a", default="mlp")
        p.add_argument("--data", default=None, help="path to your data (else synthetic)")
        p.add_argument("--set", action="append", default=[], metavar="k=v", help="param override")
        p.add_argument("--optimizer", default="hodge")
        p.add_argument("--steps", type=int, default=200)
        p.add_argument("--seed", type=int, default=0)
        if name == "multistep":
            p.add_argument("--stage", action="append", default=[], metavar="k=v,k=v",
                           help="a training stage, repeatable")
        if name == "fusion":
            p.add_argument("--spec", action="append", default=[], metavar="archetype[:k=v,..]",
                           help="a base model, repeatable")
            p.add_argument("--fusion", default="ensemble", choices=["ensemble", "split", "stack"])
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for x in list_archetypes():
            print(f"\n{x['name']}  - {x['use_case']}")
            print(f"  data: {x['data_kind']}   params: {x['params']}")
        return

    params = _kv(a.set)
    if a.cmd == "build":
        res = run(a.archetype, params=params, data=a.data, optimizer=a.optimizer,
                  steps=a.steps, seed=a.seed)
    elif a.cmd == "multistep":
        stages = [_kv(s.split(",")) for s in a.stage] or \
                 [{"optimizer": "adam", "steps": 100}, {"optimizer": "hodge", "steps": 200}]
        res = run(a.archetype, params=params, data=a.data, mode="multistep", stages=stages, seed=a.seed)
    else:  # fusion
        specs = []
        for s in (a.spec or [a.archetype, a.archetype]):
            nm, _, ov = s.partition(":")
            specs.append((nm, _kv(ov.split(",")) if ov else {}))
        res = run(a.archetype, params=params, data=a.data, mode="fusion", specs=specs,
                  fusion=a.fusion, steps=a.steps, seed=a.seed)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
