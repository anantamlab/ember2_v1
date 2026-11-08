"""Read-only P1-09 preflight. Writes nothing into the repository. Prints one JSON summary."""
import json
import sys
from pathlib import Path

from ember_v1 import provenance as pv
from ember_v1 import trust
from ember_v1.sim.canonical import digest
from ember_v1.teacher import confirmation as C
from ember_v1.teacher import registry as reg
from ember_v1.teacher import selection as sel

PC = "64b2d9873953ba750fd6c08fb84761f1bcd8e0bb"
EXP = {"manifest": "sha256:4e9283b206296d7251514140613f0f8cc234fe9e2dc68877f7a71e906dc5cf9e",
       "audit": "sha256:802b8fa509767c3b3cc881289a81c9285ae254498aa0a1de255fb0e035ea2ef2",
       "protocol": "sha256:ae72c2cf1a0dae4f46b3a6d158f426af0853bc4100cd15706c74f203a6765c21"}
repo = C.ev.REPO
rel = lambda p: p.relative_to(repo).as_posix()  # noqa: E731
out, fail = {}, []

proto_ref = {"kind": "git", "commit": PC, "path": rel(C.OUT_DIR / "protocol.json")}
man_ref = {"kind": "git", "commit": PC, "path": rel(C.MANIFEST_PATH)}
aud_ref = {"kind": "git", "commit": PC, "path": rel(C.AUDIT_PATH)}
p, _ = trust.load_trusted_json(repo, proto_ref)
sel.verify_frozen_protocol(p)
m, _ = trust.load_trusted_json(repo, man_ref)
trust.check_manifest(m)
a, _ = trust.load_trusted_json(repo, aud_ref)
out["hashes"] = {"protocol": p["protocol_hash"], "manifest": m["manifest_hash"], "audit": a["audit_hash"],
                 "protocol_preregisters_manifest": p["confirmation"]["manifest_hash"]}
for k, key in (("protocol", "protocol"), ("manifest", "manifest"), ("audit", "audit")):
    if out["hashes"][key] != EXP[k]:
        fail.append(f"{k} hash mismatch")
if p["confirmation"]["manifest_hash"] != EXP["manifest"]:
    fail.append("protocol does not preregister the manifest")
# Working-tree files equal the trusted objects.
for path, doc in ((C.OUT_DIR / "protocol.json", p), (C.MANIFEST_PATH, m), (C.AUDIT_PATH, a)):
    if json.loads(path.read_text(encoding="utf-8")) != doc:
        fail.append(f"{path.name}: working tree differs from trusted object")

# Teachers: eligibility identities vs registry vs code, and source fingerprints.
pins, structure = reg.check_pins(), reg.check_registry()
if pins or structure:
    fail.append(f"registry drift: {pins} {structure}")
teachers = {}
for c in reg.CANDIDATES:
    e = p["eligibility"][c]
    ident = e["identity"]
    cls = reg.teacher_class(ident["name"], ident["version"])
    seed = p["confirmation"]["policy_seeds"].get(c)
    inst = cls() if seed is None else cls(seed=seed)
    code = inst.identity
    got = {"name": code.name, "policy": code.policy, "version": code.version, "config_hash": code.config_hash}
    if e["status"] != reg.ELIGIBLE or got != {k: ident[k] for k in got}:
        fail.append(f"{c}: identity mismatch or not eligible")
    teachers[c] = {**got, "status": e["status"], "seed": seed, "instance_seed": inst.seed,
                   "code_fingerprint": pv.code_fingerprint(cls)}
out["teachers"] = teachers

# Regenerate the manifest and the audit in memory and compare with the committed objects.
entries, fams, prov = C.historical_sets()
pilots = C.pilot_maps()
m2 = C.build_manifest(entries, fams, pilots=pilots)
a2 = C.isolation_report(m2, entries, prov, pilots)
out["regen"] = {"manifest_hash": m2["manifest_hash"], "audit_hash": a2["audit_hash"], "audit_pass": a2["pass"],
                "families": a2["families"], "scenarios": a2["scenarios"],
                "manifest_equal": m2 == m, "audit_equal": a2 == a,
                "family_ids": len({s["family_id"] for s in m2["scenarios"]}), "scenario_rows": len(m2["scenarios"])}
if not (m2 == m and a2 == a and a2["pass"]):
    fail.append("regenerated manifest/audit differ or audit fails")
# Materialize: every scenario regenerates with the recorded hash (raises otherwise).
out["materialized"] = len(C.materialize(m))

# The real runner preconditions (read-only; raises with every reason).
try:
    pre = C.run_preconditions(repo, proto_ref, man_ref)
    out["run_preconditions"] = {"pass": True, "code_commit": pre["code_commit"]}
except C.ConfirmationError as exc:
    out["run_preconditions"] = {"pass": False, "reasons": str(exc)}
    fail.append("run_preconditions")
out["compute_preflight"] = {k: v for k, v in C.compute_preflight(p).items() if k != "dev_cpu_s"}
out["compute"] = p["compute"]
marker, log = C.attempt_paths(C.EVIDENCE_PATH)
out["no_prior_files"] = not (C.EVIDENCE_PATH.exists() or marker.exists() or log.exists())
out["fail"] = fail
print(json.dumps(out, indent=1, sort_keys=True))
sys.exit(1 if fail else 0)
