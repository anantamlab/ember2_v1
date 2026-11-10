"""P0B `chunk_smoke`: one bounded A10 task for P1-11 R3/R4/R7 evidence. NOT RUN; needs written authorization.

Checks, in one container (inference only, no training, no padding experiment, no throughput sweep):
1. kernel provenance: the loaded FLA package file (must be under `unsloth_zoo/_vendored/fla`), its
   `__version__` and `_UNSLOTH*` markers, and the defining module of every bound causal-conv function
   (must be a pinned unsloth-zoo conv module);
2. native single rows, compact and verbose, N = 1, 10, 50, 100: positive preservation evidence (no
   truncation) and a validated predict(), with latency and peak GPU memory;
3. N = 200 single row: the EXPECTED outcome is a refusal (not `OK`). An `OK` contradicts the CPU token
   count (25 845 fixed tokens for compact) and is a FAIL;
4. N = 200 chunked (`CHUNK_PLAN`): compact k = 100 and verbose k = 50 must have every chunk row preserved,
   one validated prediction per row, and a complete joint map assembled fail-closed and resolved by the
   public movement resolver; verbose k = 100 must be refused (over the bound on the CPU count).

Torch and unsloth are reached only through the injected callables, so CPU tests use fakes.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from . import chunk_joint as cj
from . import token_budget as tb
from .synthetic import make_synthetic_observation

SINGLE_NS = (1, 10, 50, 100)
STYLES = ("compact", "verbose")
REFUSAL_N, CHUNK_N = 200, 200
# (style, k, expected). CPU counts with the pinned tokenizer (docs/P1_CHUNK_SMOKE.md): compact k=100 rows
# 15441 and 15255 tokens; verbose k=100 rows 17441 and 17255 (STATE_CUT, over 16384), so verbose needs k=50.
CHUNK_PLAN = (("compact", 100, "pass"), ("verbose", 100, "refusal"), ("verbose", 50, "pass"))
VENDORED_FLA_PATH = "unsloth_zoo/_vendored/fla"
TIMEOUT_S = 1800


FLA_MODULE_NAMES = ("fla", "unsloth_zoo._vendored.fla")
# Modules that may supply the bound conv function: the two conv1d paths of the pinned unsloth-zoo wheel
# (data/p0b/p0b-wheel-provenance.json, packages/unsloth-zoo/conv1d_paths).
UNSLOTH_CONV_MODULES = ("unsloth_zoo.temporary_patches.gdn_causal_conv1d", "unsloth_zoo._vendored.fla")


def kernel_provenance(model: Any, *, modules: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Runtime facts after the model load. Attribute names in the loaded code are NOT assumed: every
    callable module attribute whose name contains `causal_conv1d` is recorded with its defining module.
    PASS needs a loaded FLA package file under the vendored path AND at least one conv binding, all from
    the pinned unsloth-zoo conv modules. No binding found is UNVERIFIED (the attribute names then need a
    source read), never PASS."""
    import sys
    mods = sys.modules if modules is None else modules
    fla = []
    for name in FLA_MODULE_NAMES:
        m = mods.get(name)
        if m is not None:
            fla.append({"name": name, "file": str(getattr(m, "__file__", None)).replace("\\", "/"),
                        "version": getattr(m, "__version__", None),
                        "markers": {a: repr(getattr(m, a))[:80] for a in dir(m) if a.upper().startswith("_UNSLOTH")}})
    bindings, seen = [], set()
    for name, mod in getattr(model, "named_modules", lambda: [])():
        for attr in sorted(vars(mod)) if hasattr(mod, "__dict__") else []:
            fn = vars(mod)[attr]
            if "causal_conv1d" not in attr or not callable(fn) or (attr, id(fn)) in seen:
                continue
            seen.add((attr, id(fn)))
            src = str(getattr(fn, "__module__", None))
            bindings.append({"first_module": name, "attribute": attr, "defined_in": src,
                             "qualname": getattr(fn, "__qualname__", type(fn).__name__),
                             "unsloth": src.startswith(UNSLOTH_CONV_MODULES)})
    vendored = any(VENDORED_FLA_PATH in f["file"] for f in fla)
    conv_ok = bool(bindings) and all(b["unsloth"] for b in bindings)
    status = ("PASS" if vendored and conv_ok else
              "FAIL" if (fla and not vendored) or (bindings and not conv_ok) else "UNVERIFIED")
    return {"fla_modules": fla, "conv_bindings": bindings, "vendored_fla": vendored, "status": status}


def _answers(pred: Mapping[str, Any]) -> dict[str, str]:
    return {pid: a["answer"] for pid, a in pred["answers"].items()}


def chunk_smoke(model: Any, tokenizer: Any, *, encode_one: Callable, predict_answers: Callable,
                synthetic_row: Callable[[int, str], Mapping[str, Any]], timed: Callable[[Callable, int], Mapping],
                peak_memory_gb: Callable[[], float | None], repeats: int = 5,
                provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("at least two repetitions")
    singles, statuses = [], []
    for style in STYLES:
        for n in SINGLE_NS:
            row = synthetic_row(n, style)
            _, ev = encode_one(row, tokenizer, model)
            e = {"style": style, "n": n, "encode_status": ev["status"], "truncated": ev.get("truncated"),
                 "input_tokens": ev.get("input_tokens")}
            if ev["status"] == "OK" and ev.get("truncated") is False:
                first = predict_answers(model, tokenizer, row, evidence=ev)
                e["predict_status"] = first["status"]
                if first["status"] == "OK":
                    e["latency"] = dict(timed(lambda: predict_answers(model, tokenizer, row, evidence=ev), repeats))
            e["status"] = "PASS" if e.get("predict_status") == "OK" else "FAIL"
            singles.append(e)
            statuses.append(e["status"])
    refusals = []
    for style in STYLES:
        _, ev = encode_one(synthetic_row(REFUSAL_N, style), tokenizer, model)
        ok = ev["status"] != "OK"
        refusals.append({"style": style, "n": REFUSAL_N, "encode_status": ev["status"], "expected": "refusal",
                         "status": "PASS" if ok else "FAIL"})
        statuses.append(refusals[-1]["status"])
    chunked = []
    for style, k, expect in CHUNK_PLAN:
        obs = make_synthetic_observation(CHUNK_N)
        rows = tb.chunk_rows(obs, {p["id"]: "stay" for p in obs["people"]}, k=k, group_id=f"synthetic-{CHUNK_N}",
                             source="synthetic-contract-only", question_style=style)
        entry: dict[str, Any] = {"style": style, "n": CHUNK_N, "k": k, "expected": expect, "rows": []}
        if expect == "refusal":
            evs = [encode_one({x: r[x] for x in ("state", "questions", "gold", "group_id", "source")},
                              tokenizer, model)[1] for r in rows]
            entry["rows"] = [{"people": len(r["questions"]), "encode_status": e["status"],
                              "full_tokens": e.get("full_tokens")} for r, e in zip(rows, evs)]
            entry["status"] = "PASS" if all(e["status"] != "OK" for e in evs) else "FAIL"
            chunked.append(entry)
            statuses.append(entry["status"])
            continue
        preds, failed = [], False
        for r in rows:
            row = {k: r[k] for k in ("state", "questions", "gold", "group_id", "source")}
            _, ev = encode_one(row, tokenizer, model)
            re_ = {"people": len(r["questions"]), "encode_status": ev["status"], "truncated": ev.get("truncated"),
                   "input_tokens": ev.get("input_tokens")}
            if ev["status"] != "OK" or ev.get("truncated") is not False:
                failed = True
            else:
                t0 = time.perf_counter()
                p = predict_answers(model, tokenizer, row, evidence=ev)
                re_["predict_s"] = round(time.perf_counter() - t0, 4)
                re_["predict_status"] = p["status"]
                if p["status"] != "OK":
                    failed = True
                else:
                    preds.append(_answers(p))
            entry["rows"].append(re_)
        if not failed:
            try:
                chunks = [r["chunk"]["ids"] for r in rows]
                joint = cj.assemble(obs, chunks, preds)
                res = cj.resolve(obs, joint)
                entry["joint_people"] = len(joint)
                entry["resolver_outcomes"] = {o: sum(v[0] == o for v in res.values()) for o in {v[0] for v in res.values()}}
                entry["cross_chunk_blocked"] = cj.cross_chunk_conflicts(obs, chunks, joint, res)
            except cj.ChunkError as e:
                failed, entry["joint_error"] = True, str(e)[:300]
        entry["status"] = "FAIL" if failed else "PASS"
        chunked.append(entry)
        statuses.append(entry["status"])
    prov = dict(provenance) if provenance is not None else kernel_provenance(model)
    statuses.append(prov["status"])
    overall = "PASS" if all(s == "PASS" for s in statuses) else ("FAIL" if "FAIL" in statuses else "UNVERIFIED")
    return {"task": "chunk_smoke", "kernel_provenance": prov, "single_rows": singles, "n200_single_row": refusals,
            "n200_chunked": chunked, "peak_memory_gb": peak_memory_gb(), "status": overall,
            "scope": "inference only; batch 1; no training, no padding experiment, no throughput sweep"}
