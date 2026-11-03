"""P0B padding-equivalence verdict from *measured* GPU decision probabilities.

This never infers scores or executes a foundation model. A Modal harness must
produce full, untruncated per-person distributions for identical observations
under singleton and mixed-length batched inference. Fail closed on absent IDs,
nonfinite probabilities, unsupported variants or mismatched actions.

Input schema: {"metadata": {...}, "cases": [{"env_id": "e1",
 "variant": "single|right|left|packed", "probabilities":
 {"P1": {"stay": 0.2, "north": 0.2, "south": 0.2,
         "east": 0.2, "west": 0.2}}, "input_tokens": 10, "truncated": false}, ...]}

"packed" is optional and must be a *real* varlen/cu_seqlens execution, not a
renamed right-padded call. The harness also records kernel/packing proof.

`metadata` must pass `validate_provenance` (pinned 40-hex model commit, library
versions, GPU, precision, max_seq, token limit, head path, evidence source and
profiled kernels). `truncated` must be exactly False: None (unverified) fails.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ACTIONS = frozenset(("stay", "north", "south", "east", "west"))
VARIANTS = frozenset(("single", "right", "left", "packed"))
REQUIRED_LIBRARIES = ("torch", "transformers", "unsloth")
_REVISION = re.compile(r"^.+@[0-9a-f]{40}$")


def validate_provenance(metadata: Any, *, require_kernel_profile: bool = True) -> list[str]:
    """Strict run provenance. Raise on absent or placeholder values.

    Returns the recorded gaps that do not block a verdict but must be reported
    (a None tokenizer revision or code commit). Import availability of a kernel
    package is never evidence that the kernel ran.
    """
    if not isinstance(metadata, dict):
        raise ValueError("missing provenance metadata")
    revision = metadata.get("model_revision")
    if not isinstance(revision, str) or not _REVISION.match(revision):
        raise ValueError("model_revision must be '<model>@<40-hex commit>'; unknown revisions never pass")
    versions = metadata.get("runtime_versions")
    if not isinstance(versions, dict) or not all(isinstance(versions.get(k), str) and versions[k]
                                                 for k in REQUIRED_LIBRARIES):
        raise ValueError(f"runtime_versions must record {list(REQUIRED_LIBRARIES)}")
    for key in ("gpu", "head_path", "evidence_source"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"missing provenance field {key}")
    if metadata.get("precision") not in ("bf16", "4bit"):
        raise ValueError("missing or unknown precision")
    for key in ("max_seq", "token_limit"):
        if type(metadata.get(key)) is not int or metadata[key] <= 0:
            raise ValueError(f"missing provenance field {key}")
    kernels = metadata.get("kernel_backend")
    if not isinstance(kernels, dict) or not isinstance(kernels.get("detected_packages"), list):
        raise ValueError("kernel_backend must separate detected packages from executed kernels")
    if require_kernel_profile and (kernels.get("evidence") != "torch_profiler" or not kernels.get("executed")):
        raise ValueError("no profiled kernel evidence; an importable package is not an executed kernel")
    for key in ("tokenizer_revision", "code_commit"):
        if key not in metadata:
            raise ValueError(f"provenance field {key} is absent (record None if unknown)")
    return [key for key in ("tokenizer_revision", "code_commit") if not metadata[key]]


def _validate_case(case: dict[str, Any]) -> None:
    if not isinstance(case, dict) or not isinstance(case.get("env_id"), str) or not case["env_id"]:
        raise ValueError("every case needs a nonempty env_id")
    if case.get("variant") not in VARIANTS:
        raise ValueError("unknown padding variant")
    if case.get("truncated") is not False:
        raise ValueError("truncated or unverified context")
    if type(case.get("input_tokens")) is not int or case["input_tokens"] <= 0:
        raise ValueError("missing actual encoded input_tokens")
    probs = case.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        raise ValueError("missing per-person probabilities")
    for pid, values in probs.items():
        if not isinstance(pid, str) or not pid or not isinstance(values, dict) or set(values) != ACTIONS:
            raise ValueError(f"missing/extra person or action probabilities at {pid!r}")
        if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 or p > 1 for p in values.values()):
            raise ValueError(f"invalid probabilities for {pid!r}")
        if abs(sum(values.values()) - 1.0) > 0.0001:
            raise ValueError(f"probabilities not normalized for {pid!r}")


def assess_padding_parity(
    data: dict[str, Any], *, max_abs_delta: float = 0.01,
    max_l1_delta: float = 0.02, confident_margin: float = 0.02,
) -> dict[str, Any]:
    """Compare all measured variants to the SAME environment's singleton run.

    Fail closed on omitted variants or IDs. Confidence flips are judged with
    margin from the singleton distribution. Numeric thresholds are initial
    preregistration candidates; tuning them requires a recorded justification.
    """
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("expected a JSON object with cases")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
               for v in (max_abs_delta, max_l1_delta, confident_margin)):
        raise ValueError("invalid thresholds")
    metadata = data.get("metadata")
    provenance_gaps = validate_provenance(metadata)
    cases: dict[str, dict[str, dict[str, Any]]] = {}
    for case in data["cases"]:
        _validate_case(case)
        group = cases.setdefault(case["env_id"], {})
        if case["variant"] in group:
            raise ValueError(f"duplicate environment/variant: {case['env_id']}/{case['variant']}")
        group[case["variant"]] = case
    if len(cases) < 2:
        raise ValueError("mixed-length batch needs at least two distinct environments")
    if len({modes["single"]["input_tokens"] for modes in cases.values() if "single" in modes}) < 2:
        raise ValueError("all singleton inputs have the same encoded length: not a mixed-length test")
    # Left padding may be excluded ONLY by an explicit, recorded scope decision
    # (the Clef head uses attention_mask.sum() as the length, i.e. assumes right
    # padding). Unsupported left padding is never reported as passing.
    left_reason = metadata.get("left_unsupported_reason")
    if left_reason is not None and (not isinstance(left_reason, str) or not left_reason.strip()):
        raise ValueError("left_unsupported_reason must be a nonempty string when present")
    if left_reason and any("left" in modes for modes in cases.values()):
        raise ValueError("left results present but left padding is marked unsupported")
    required = {"single", "right"} if left_reason else {"single", "right", "left"}
    results = []
    for env_id, modes in sorted(cases.items()):
        if not required.issubset(modes):
            raise ValueError(f"missing {sorted(required - set(modes))} padding variant(s) for {env_id}")
        ref = modes["single"]["probabilities"]
        for variant in ("right", "left", "packed"):
            if variant not in modes:
                continue
            # input_tokens counts valid (unpadded) tokens. A different count means
            # the prompt was rewritten or truncated, so the comparison is invalid.
            if modes[variant]["input_tokens"] != modes["single"]["input_tokens"]:
                raise ValueError(f"unpadded token count differs from singleton for {env_id}/{variant}")
            cur = modes[variant]["probabilities"]
            if set(cur) != set(ref):
                raise ValueError(f"ID set mismatch for {env_id}/{variant}")
            max_abs = 0.0
            max_l1 = 0.0
            confident_flips = []
            for pid in sorted(ref):
                baseline = ref[pid]
                measured = cur[pid]
                diffs = [abs(baseline[action] - measured[action]) for action in ACTIONS]
                max_abs = max(max_abs, max(diffs))
                max_l1 = max(max_l1, sum(diffs))
                ordered = sorted(ACTIONS, key=lambda a: (-baseline[a], a))
                margin = baseline[ordered[0]] - baseline[ordered[1]]
                # Same tie-break as the singleton ordering (alphabetical first).
                predicted = min(ACTIONS, key=lambda a: (-measured[a], a))
                if margin >= confident_margin and predicted != ordered[0]:
                    confident_flips.append(pid)
            ok = max_abs <= max_abs_delta and max_l1 <= max_l1_delta and not confident_flips
            results.append({"env_id": env_id, "variant": variant,
                            "max_abs_delta": round(max_abs, 8), "max_l1_delta": round(max_l1, 8),
                            "confident_flips": confident_flips, "pass": ok})
    packed = any("packed" in modes for modes in cases.values())
    if packed and not all("packed" in modes for modes in cases.values()):
        raise ValueError("partial packed results across environments")
    if packed and metadata.get("packed_varlen_verified") is not True:
        raise ValueError("packed path not verified; do not label padded batches as packed")
    return {"status": "PASS" if all(r["pass"] for r in results) else "FAIL",
            "environments": len(cases), "packed_tested": packed,
            "left_tested": not left_reason, "left_unsupported_reason": left_reason,
            "provenance_gaps": provenance_gaps,
            "thresholds": {"max_abs_delta": max_abs_delta, "max_l1_delta": max_l1_delta,
                           "confident_margin": confident_margin}, "comparisons": results}


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess P0B padding-parity results collected on a real GPU")
    ap.add_argument("json_file", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    verdict = assess_padding_parity(json.loads(args.json_file.read_text(encoding="utf-8")))
    report = json.dumps(verdict, indent=2, sort_keys=True)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    if verdict["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
