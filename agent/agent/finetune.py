"""
agent.finetune: LoRA fine-tune of a Hugging Face model on this machine's GPU, plus an
optimizer A/B on the same run.

`finetune()` builds its optimizer with `make_optimizer(optimizer, ...)` and defaults to `"auto"`,
which routes a feature-space model (this is one: LoRA adapts weight matrices) to plain Adam. It
streams the loss through the platform's run log and produces a loadable adapter. `finetune_ab()`
names its two arms deliberately (`("hodge", "adam")` by default): it trains the same model on the
same data with the same seed under each, so the two loss curves are directly comparable, and judges
on a held-out eval split. HodgeAdam appears there as an arm under test, not as a recommendation.

Scope: LoRA optimizes the model's weight matrices; the transformer's attention stays standard
(relational attention is the native-model track, not a llama.cpp/HF retrofit). This exercises the
optimizer and the end-to-end platform, not relational attention.

Heavy deps (transformers/peft/datasets/accelerate) are optional and imported lazily: when absent,
every entry point returns an "install the [finetune] extra" message instead of raising.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("rexgraph.finetune")

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# A tiny built-in instruction set so a run needs no data setup. Small on purpose: it exercises the
# optimizer moving weights, not the training of a strong model.
_TINY_DATA: list[dict] = [
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "Name a primary color.", "response": "Red is a primary color."},
    {"instruction": "What is 2 plus 2?", "response": "2 plus 2 equals 4."},
    {"instruction": "Who wrote Romeo and Juliet?", "response": "William Shakespeare wrote Romeo and Juliet."},
    {"instruction": "What gas do plants absorb?", "response": "Plants absorb carbon dioxide."},
    {"instruction": "What is the boiling point of water in Celsius?", "response": "Water boils at 100 degrees Celsius."},
    {"instruction": "Define a noun.", "response": "A noun is a word that names a person, place, thing, or idea."},
    {"instruction": "What planet do we live on?", "response": "We live on planet Earth."},
    {"instruction": "How many days are in a week?", "response": "There are seven days in a week."},
    {"instruction": "What is the opposite of hot?", "response": "The opposite of hot is cold."},
    {"instruction": "What language is spoken in Brazil?", "response": "Portuguese is spoken in Brazil."},
    {"instruction": "What do bees make?", "response": "Bees make honey."},
    {"instruction": "What is H2O commonly known as?", "response": "H2O is commonly known as water."},
    {"instruction": "Name the closest star to Earth.", "response": "The Sun is the closest star to Earth."},
    {"instruction": "What is the largest ocean?", "response": "The Pacific Ocean is the largest ocean."},
    {"instruction": "What tool tells time?", "response": "A clock tells time."},
    {"instruction": "What is the capital of Japan?", "response": "The capital of Japan is Tokyo."},
    {"instruction": "What metal is liquid at room temperature?", "response": "Mercury is liquid at room temperature."},
    {"instruction": "How many legs does a spider have?", "response": "A spider has eight legs."},
    {"instruction": "What is the freezing point of water in Celsius?", "response": "Water freezes at 0 degrees Celsius."},
    {"instruction": "Who painted the Mona Lisa?", "response": "Leonardo da Vinci painted the Mona Lisa."},
    {"instruction": "What is the chemical symbol for gold?", "response": "The chemical symbol for gold is Au."},
    {"instruction": "What is the tallest mountain on Earth?", "response": "Mount Everest is the tallest mountain on Earth."},
    {"instruction": "What do cows drink?", "response": "Cows drink water."},
    {"instruction": "How many continents are there?", "response": "There are seven continents."},
    {"instruction": "What is the opposite of up?", "response": "The opposite of up is down."},
    {"instruction": "What organ pumps blood?", "response": "The heart pumps blood."},
    {"instruction": "What is the capital of Italy?", "response": "The capital of Italy is Rome."},
    {"instruction": "What color is the sky on a clear day?", "response": "The sky is blue on a clear day."},
    {"instruction": "What is the largest planet in our solar system?", "response": "Jupiter is the largest planet in our solar system."},
    {"instruction": "What season comes after winter?", "response": "Spring comes after winter."},
    {"instruction": "What is the currency of the United States?", "response": "The currency of the United States is the dollar."},
    {"instruction": "How many sides does a triangle have?", "response": "A triangle has three sides."},
    {"instruction": "What gas do humans breathe in to survive?", "response": "Humans breathe in oxygen to survive."},
    {"instruction": "What is the fastest land animal?", "response": "The cheetah is the fastest land animal."},
    {"instruction": "What is the capital of Germany?", "response": "The capital of Germany is Berlin."},
    {"instruction": "What do you call frozen rain?", "response": "Frozen rain is called hail or snow."},
    {"instruction": "What is the smallest prime number?", "response": "The smallest prime number is 2."},
    {"instruction": "Who discovered gravity?", "response": "Isaac Newton is credited with discovering gravity."},
    {"instruction": "What is the main language spoken in Mexico?", "response": "Spanish is the main language spoken in Mexico."},
    {"instruction": "What shape has four equal sides?", "response": "A square has four equal sides."},
    {"instruction": "What is the hottest planet in the solar system?", "response": "Venus is the hottest planet in the solar system."},
    {"instruction": "What do caterpillars turn into?", "response": "Caterpillars turn into butterflies."},
    {"instruction": "What is the capital of Canada?", "response": "The capital of Canada is Ottawa."},
    {"instruction": "How many hours are in a day?", "response": "There are twenty-four hours in a day."},
    {"instruction": "What is the largest mammal?", "response": "The blue whale is the largest mammal."},
    {"instruction": "What is the chemical symbol for oxygen?", "response": "The chemical symbol for oxygen is O."},
    {"instruction": "What is the opposite of fast?", "response": "The opposite of fast is slow."},
    {"instruction": "What planet is known as the Red Planet?", "response": "Mars is known as the Red Planet."},
    {"instruction": "What do bees collect from flowers?", "response": "Bees collect nectar from flowers."},
    {"instruction": "How many colors are in a rainbow?", "response": "A rainbow has seven colors."},
    {"instruction": "What is the capital of Spain?", "response": "The capital of Spain is Madrid."},
    {"instruction": "What is frozen water called?", "response": "Frozen water is called ice."},
    {"instruction": "Who wrote the theory of relativity?", "response": "Albert Einstein wrote the theory of relativity."},
    {"instruction": "What is the longest river in the world?", "response": "The Nile is often cited as the longest river in the world."},
    {"instruction": "What is the square root of 9?", "response": "The square root of 9 is 3."},
    {"instruction": "What do plants need to make food?", "response": "Plants need sunlight, water, and carbon dioxide to make food."},
    {"instruction": "What is the capital of Australia?", "response": "The capital of Australia is Canberra."},
    {"instruction": "What is the opposite of day?", "response": "The opposite of day is night."},
    {"instruction": "What animal is known as man's best friend?", "response": "The dog is known as man's best friend."},
    {"instruction": "How many minutes are in an hour?", "response": "There are sixty minutes in an hour."},
    {"instruction": "What is the primary language of France?", "response": "The primary language of France is French."},
]


def deps_available() -> dict:
    """Report which fine-tune deps are present and whether the phase can run, so the UI/CLI can
    report what to install rather than failing mid-run."""
    have = {}
    for m in ("torch", "transformers", "peft", "datasets", "accelerate", "safetensors"):
        try:
            __import__(m); have[m] = True
        except Exception:
            have[m] = False
    missing = [m for m in ("torch", "transformers", "peft", "safetensors") if not have[m]]
    return {"have": have, "ready": not missing,
            "need": ("pip install -e '.[finetune]'" if missing else ""),
            "missing": missing}


def _artifacts_dir(run_id: str | None) -> Path:
    base = Path(os.environ.get("REXGRAPH_CONFIG_DIR", Path.home() / ".config" / "rexgraph"))
    d = base / "runs" / "artifacts" / (run_id or "finetune")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _format(ex: dict) -> str:
    if "text" in ex and ex["text"]:
        return str(ex["text"])
    return f"Instruction: {ex.get('instruction','')}\nResponse: {ex.get('response','')}"


def load_data(dataset=None, *, text_field: str | None = None,
              instruction_field: str = "instruction", response_field: str = "response",
              split: str = "train", limit: int | None = None) -> list[dict]:
    """Load training rows. `dataset` may be:
      - None              -> the built-in tiny set,
      - a local file      -> .jsonl / .json / .csv (rows) or .txt (one text per line),
      - a HF dataset id   -> loaded via `datasets` (needs the extra).
    Rows are normalized to {'text'} or {'instruction','response'} using the field names given
    (auto-detects common ones). Use `text_field` for a plain-text column."""
    if dataset is None:
        return list(_TINY_DATA)
    rows: list[dict] = []
    p = os.path.expanduser(str(dataset))
    if os.path.exists(p):
        if p.endswith(".jsonl"):
            with open(p) as f:
                rows = [json.loads(ln) for ln in f if ln.strip()]
        elif p.endswith(".json"):
            with open(p) as f:
                d = json.load(f)
            rows = d if isinstance(d, list) else d.get("data", [])
        elif p.endswith(".csv"):
            with open(p, newline="") as f:
                rows = list(csv.DictReader(f))
        else:                                    # .txt / anything -> one text example per line
            with open(p) as f:
                rows = [{"text": ln.rstrip("\n")} for ln in f if ln.strip()]
    else:
        from datasets import load_dataset  # HF dataset id
        rows = [dict(r) for r in load_dataset(dataset, split=split)]
    if limit:
        rows = rows[:int(limit)]
    norm: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            norm.append({"text": str(r)}); continue
        if text_field and r.get(text_field):
            norm.append({"text": str(r[text_field])})
        elif r.get(instruction_field) is not None and r.get(response_field) is not None:
            norm.append({"instruction": str(r[instruction_field]), "response": str(r[response_field])})
        elif r.get("text"):
            norm.append({"text": str(r["text"])})
        elif r.get("instruction") is not None and r.get("response") is not None:
            norm.append({"instruction": str(r["instruction"]), "response": str(r["response"])})
        else:                                    # last resort: concat the string columns
            norm.append({"text": " ".join(str(v) for v in r.values() if isinstance(v, str))})
    return [r for r in norm if (r.get("text") or r.get("response"))]


def finetune(*, model_id: str = DEFAULT_MODEL, optimizer: str = "auto", steps: int = 60,
             lora_r: int = 8, lora_alpha: int = 16, lr: float | None = None,
             seq_len: int = 64, batch: int = 4, seed: int = 0, device: str | None = None,
             data: list[dict] | None = None, dataset=None, text_field: str | None = None,
             instruction_field: str = "instruction", response_field: str = "response",
             split: str = "train", data_limit: int | None = None, full: bool = False,
             target_modules=None, save_dir: str | None = None, on_step: Callable = None,
             label: str | None = None) -> dict:
    """One fine-tune run with the chosen optimizer, on a given model and data. Loads the HF model
    (`model_id`: a hub id or a local path), optionally wraps it in LoRA (or `full=True` for a full
    fine-tune), trains on `dataset` (see `load_data`) streaming loss, and saves the result. Returns
    train and held-out-eval trajectories. Requires the [finetune] extra."""
    dep = deps_available()
    if not dep["ready"]:
        return {"skipped": dep["need"], "optimizer": optimizer, "deps": dep}
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rexgraph.nn import optim

    dev = optim.pick_device(device)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if full:
        model = model.to(dev)                    # full fine-tune: every weight trains
    else:
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=target_modules or ["q_proj", "v_proj"])
        model = get_peft_model(model, lora).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_heads = int(getattr(model.config, "num_attention_heads", 1) or 1)

    from rexgraph import nn
    try:
        opt, opt_class = nn.make_optimizer(optimizer, model, trainable, n_heads=n_heads, lr=lr or 1e-4)
    except Exception as e:                       # graceful fallback to Adam
        logger.warning("optimizer %r failed (%s) -> Adam", optimizer, e)
        opt = torch.optim.Adam(trainable, lr=lr or 1e-4); opt_class = "Adam"; optimizer = "adam"

    # deterministic train/eval split: eval is held-out data the model never trains on, so the
    # A/B verdict compares generalization, not memorization of the training loss.
    ex = list(data or load_data(dataset, text_field=text_field, instruction_field=instruction_field,
                                response_field=response_field, split=split, limit=data_limit))
    if len(ex) < 5:
        return {"skipped": f"need >=5 examples, got {len(ex)}", "optimizer": optimizer}
    order = torch.randperm(len(ex), generator=torch.Generator().manual_seed(1234)).tolist()
    n_eval = max(4, len(ex) // 5)
    eval_rows = [_format(ex[i]) for i in order[:n_eval]]
    train_rows = [_format(ex[i]) for i in order[n_eval:]]

    def _batch(rows, idxs):
        enc = tok([rows[j] for j in idxs], return_tensors="pt", padding=True,
                  truncation=True, max_length=seq_len)
        enc = {k: v.to(dev) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return enc, labels

    def _eval_loss():
        model.eval()
        with torch.no_grad():
            enc, labels = _batch(eval_rows, list(range(len(eval_rows))))
            lo = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       labels=labels).loss
        model.train()
        return round(float(lo.item()), 4)

    model.train()
    traj: list[float] = []
    eval_traj: list[float] = []
    g = torch.Generator().manual_seed(seed)
    for i in range(steps):
        idx = torch.randint(0, len(train_rows), (batch,), generator=g).tolist()
        enc, labels = _batch(train_rows, idx)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        loss = out.loss
        opt.zero_grad(); loss.backward(); opt.step()
        lv = float(loss.item())
        traj.append(round(lv, 4))
        if i % 5 == 0 or i == steps - 1:
            eval_traj.append(_eval_loss())
        if on_step:
            on_step(label or optimizer, i, lv, steps)

    adapter = None
    if save_dir is not False:
        d = Path(save_dir) if save_dir else _artifacts_dir(None) / (label or optimizer)
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(d))
        adapter = str(d)
    return {"optimizer": optimizer, "optimizer_class": opt_class, "model_id": model_id,
            "device": dev, "steps": len(traj), "mode": ("full" if full else f"lora-r{lora_r}"),
            "lora_r": lora_r, "n_heads": n_heads,
            "trainable_params": n_trainable, "n_train": len(train_rows), "n_eval": len(eval_rows),
            "loss_start": traj[0] if traj else None, "loss_final": traj[-1] if traj else None,
            "eval_start": eval_traj[0] if eval_traj else None,
            "eval_final": eval_traj[-1] if eval_traj else None, "eval_trajectory": eval_traj,
            "improved": bool(eval_traj and eval_traj[-1] < eval_traj[0]),
            "trajectory": traj, "adapter": adapter}


def finetune_ab(*, model_id: str = DEFAULT_MODEL, optimizers=("hodge", "adam"), steps: int = 60,
                on_step: Callable = None, **kw) -> dict:
    """The A/B run: fine-tune the same model on the same data and seed under each optimizer, so the
    loss curves are directly comparable. Returns both runs and a verdict on held-out eval loss."""
    dep = deps_available()
    if not dep["ready"]:
        return {"skipped": dep["need"], "deps": dep, "ab": []}
    runs = []
    for opt_name in optimizers:
        logger.info("A/B finetune: %s", opt_name)
        r = finetune(model_id=model_id, optimizer=opt_name, steps=steps,
                     on_step=on_step, label=opt_name, **kw)
        if "skipped" in r:
            return r
        runs.append(r)
    # judge on held-out eval loss (generalization), not training loss
    evals = {r["optimizer"]: r["eval_final"] for r in runs if r.get("eval_final") is not None}
    trains = {r["optimizer"]: r["loss_final"] for r in runs if r.get("loss_final") is not None}
    best = min(evals, key=evals.get) if evals else None
    margin = None
    if len(evals) == 2:
        vals = sorted(evals.values())
        margin = round(vals[1] - vals[0], 4)
    verdict = "inconclusive"
    if best is not None:
        # a sub-1% eval gap is within noise, not a win
        rel = (margin / max(evals.values())) if (margin and max(evals.values())) else 0
        verdict = (f"{best} generalized better (eval gap {margin})" if rel > 0.01
                   else f"tie - eval gap {margin} is within noise")
    return {"ab": runs, "model_id": model_id, "steps": steps,
            "eval_losses": evals, "train_losses": trains, "final_losses": evals,
            "best": best, "margin": margin, "verdict": verdict}
