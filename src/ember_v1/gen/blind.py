"""Blind test set from a private salt (issue #6). Readiness: BLOCKED until a salt is provisioned.

The `p1-02-v1` locked split is procedurally locked, not blind: its plan is
public, so anyone can rebuild it. A blind set needs a secret. Here, family
seeds come from HMAC-SHA256 over a private salt, so nobody without the salt
can rebuild the blind scenarios.

Rules (docs/P1_SCENARIO_GENERATOR.md section 11):
- The salt lives in one file on the evaluator machine, outside the repository.
  It never goes into source, manifests, CI, logs or a training or tuning
  environment. `load_secret` refuses a path inside the repository and refuses
  to run under CI. There is no default salt.
- `build_blind` rebuilds the public plan from its index, checks the rebuild
  against the committed manifest hashes, and rejects blind families that
  duplicate or nearly duplicate any public family (all four public splits,
  exits included or ignored). Then it runs the split audit and the wall
  geometry audit over public plus blind scenarios.
- Publish only the commitment (hashes and counts). The private manifest holds
  seeds and labels; it stays with the evaluator.
- Teacher, training and tuning code must not import this module (a test
  enforces it). `ember_v1.gen.splits.materialize` refuses blind manifests.
- A blind set whose scenarios or per-scenario results were inspected, or that
  was used for any choice, is exposed. Record it as exposed and make the next
  blind set from a new salt and a new plan ID.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ember_v1.sim.canonical import digest

from .generator import Generated
from .spec import GENERATOR_REVISION, GeneratorSpec
from .splits import (SPLITS, AuditItem, ManifestError, _regenerate, _scenario_entry, audit, audit_items,
                     build_splits, check_hash, fill_family_pool, generate_lineages, load_index, manifest,
                     plan_from_index, wall_geometry_audit)

BLIND_VERSION = "1.0"
BLIND_SPLIT = "blind_test"
MIN_SECRET_BYTES = 32
CI_ENV = ("CI", "GITHUB_ACTIONS")
_REPO_ROOT = Path(__file__).resolve().parents[3]


class BlindSecretError(PermissionError):
    """The salt is missing, too short, in the repository, or loaded under CI."""


class BlindSecret:
    """Private salt. Never printed, serialized or compared by value outside this class."""

    __slots__ = ("_key",)

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) < MIN_SECRET_BYTES:
            raise BlindSecretError(f"the blind salt must be at least {MIN_SECRET_BYTES} random bytes")
        self._key = key

    def __repr__(self) -> str:
        return "BlindSecret(<redacted>)"

    def __reduce__(self):  # no pickling
        raise TypeError("BlindSecret cannot be serialized")

    def family_seed(self, plan_id: str, slot: int) -> int:
        mac = hmac.new(self._key, f"ember_v1/blind/{plan_id}/family/{slot}".encode("utf-8"), hashlib.sha256)
        return int.from_bytes(mac.digest()[:8], "big")

    def commitment(self) -> str:
        """Hash of the salt. Publishing it reveals nothing; revealing the salt later proves it."""
        return "sha256:" + hashlib.sha256(b"ember_v1/blind/commit/" + self._key).hexdigest()


def load_secret(path: str | Path, *, environ: Mapping[str, str] | None = None,
                repo_root: Path = _REPO_ROOT) -> BlindSecret:
    """Read a hex salt file. Refused under CI and for files inside the repository."""
    env = os.environ if environ is None else environ
    if any(env.get(k) for k in CI_ENV):
        raise BlindSecretError("blind salts must never be loaded in CI")
    p = Path(path).resolve()
    root = repo_root.resolve()
    if p == root or root in p.parents:
        raise BlindSecretError("the blind salt file must be outside the repository")
    try:
        key = bytes.fromhex(p.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise BlindSecretError(f"cannot read a hex salt from the given file: {type(exc).__name__}") from None
    return BlindSecret(key)


@dataclass(frozen=True)
class BlindPlan:
    plan_id: str
    families: int
    lineages_per_family: int

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BlindPlan":
        if not isinstance(d, Mapping) or set(d) != {"plan_id", "families", "lineages_per_family"}:
            raise ValueError("blind plan must contain plan_id, families, lineages_per_family")
        pid = d["plan_id"]
        if not isinstance(pid, str) or not pid or len(pid) > 64 or not all(ch.isalnum() or ch in "-_." for ch in pid):
            raise ValueError("plan_id must be 1-64 characters of [A-Za-z0-9._-]")
        for k, hi in (("families", 100000), ("lineages_per_family", 1000)):
            if type(d[k]) is not int or not 1 <= d[k] <= hi:
                raise ValueError(f"{k} must be an integer in [1, {hi}]")
        return cls(pid, d["families"], d["lineages_per_family"])

    def to_dict(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id, "families": self.families, "lineages_per_family": self.lineages_per_family}


def build_blind(secret: BlindSecret, plan: BlindPlan, public_index_path: str | Path
                ) -> tuple[dict[str, Any], dict[str, Any]]:
    """(private manifest, public commitment). Uses the public plan's spec (in-distribution)."""
    idx = load_index(public_index_path)
    public_plan = plan_from_index(idx)
    public = build_splits(public_plan)
    for s in SPLITS:   # independent check: the rebuild is the committed public plan
        if manifest(public, s)["manifest_hash"] != idx["splits"][s]["manifest_hash"]:
            raise ManifestError(f"public {s} rebuild differs from the committed index")
    spec = public_plan.spec
    reference = [f for s in SPLITS for f, _ in public.families[s]]
    pool, meta, info = fill_family_pool(spec, plan.families, lambda slot: secret.family_seed(plan.plan_id, slot),
                                        reference=reference, wall_rule=True)
    pool.sort(key=lambda f: f.family_id)
    accepted, failed, rejections = generate_lineages(spec, pool, plan.lineages_per_family)
    items = audit_items(public) + [
        AuditItem(BLIND_SPLIT, g.family.family_id, g.lineage_id, (g.family.family_seed, g.lineage_seed, g.scenario.seed),
                  g.scenario) for g, _ in accepted]
    report = audit(items, splits=(*SPLITS, BLIND_SPLIT))
    walls = wall_geometry_audit(items)
    if not report["passed"] or not walls["passed"]:
        raise ManifestError("blind set is not disjoint from the public splits")
    body = {"blind_version": BLIND_VERSION, "split": BLIND_SPLIT, "plan": plan.to_dict(),
            "generator_revision": GENERATOR_REVISION, "spec_hash": spec.spec_hash(),
            "public_index_hash": idx["index_hash"], "public_plan_hash": idx["plan_hash"],
            "secret_commitment": secret.commitment(),
            "families": [{**f.summary(), **meta[f.family_id]} for f in pool],
            "scenarios": [_scenario_entry(g, j) for g, j in accepted],
            "failed_lineages": failed, "lineage_rejections": rejections, "pool": info,
            "audit": report, "wall_geometry_audit": walls}
    private = {**body, "manifest_hash": digest(body)}
    commit_body = {"blind_version": BLIND_VERSION, "plan_id": plan.plan_id, "generator_revision": GENERATOR_REVISION,
                   "spec_hash": spec.spec_hash(), "public_index_hash": idx["index_hash"],
                   "secret_commitment": secret.commitment(), "manifest_commitment": private["manifest_hash"],
                   "families": len(pool), "scenarios": len(accepted), "failed_lineages": len(failed),
                   "disjoint_from_public": True, "distinct_maps_audited": report["distinct_maps"]}
    return private, {**commit_body, "commitment_hash": digest(commit_body)}


def verify_blind(secret: BlindSecret, private: Mapping[str, Any], commitment: Mapping[str, Any],
                 public_index_path: str | Path) -> None:
    """Rebuild from the salt and require the same private manifest and commitment."""
    check_hash(private, "manifest_hash")
    check_hash(commitment, "commitment_hash")
    if private.get("secret_commitment") != secret.commitment():
        raise ManifestError("this salt did not make the blind manifest")
    fresh_private, fresh_commit = build_blind(secret, BlindPlan.from_dict(private["plan"]), public_index_path)
    if fresh_private != private or fresh_commit != commitment:
        raise ManifestError("blind manifest or commitment differs from a rebuild")


def materialize_blind(secret: BlindSecret, private: Mapping[str, Any], spec: GeneratorSpec
                      ) -> Iterator[tuple[dict[str, Any], Generated]]:
    """Regenerate the blind scenarios for the one final evaluation (evaluator machine only)."""
    if private.get("split") != BLIND_SPLIT or private.get("secret_commitment") != secret.commitment():
        raise ManifestError("not a blind manifest made with this salt")
    return _regenerate(private, spec)


def _write_private(path: Path, text: str, repo_root: Path) -> None:
    p = path.resolve()
    if p == repo_root.resolve() or repo_root.resolve() in p.parents:
        raise BlindSecretError("the private blind manifest must be written outside the repository")
    if p.exists():
        raise ManifestError(f"{p} exists; blind manifests are written once")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.gen.blind",
                                 description="Build a blind test set from a private salt (evaluator machine only).")
    ap.add_argument("--salt-file", required=True, help="hex salt file outside the repository")
    ap.add_argument("--public-index", required=True, help="committed public index.json")
    ap.add_argument("--plan-id", required=True)
    ap.add_argument("--families", type=int, required=True)
    ap.add_argument("--lineages-per-family", type=int, required=True)
    ap.add_argument("--private-out", required=True, help="private manifest path outside the repository")
    args = ap.parse_args(argv)
    secret = load_secret(args.salt_file)
    plan = BlindPlan.from_dict({"plan_id": args.plan_id, "families": args.families,
                                "lineages_per_family": args.lineages_per_family})
    private, commitment = build_blind(secret, plan, args.public_index)
    _write_private(Path(args.private_out), json.dumps(private, indent=1, sort_keys=True) + "\n", _REPO_ROOT)
    print(json.dumps(commitment, indent=2, sort_keys=True))   # the only output that may be published


if __name__ == "__main__":
    main()
