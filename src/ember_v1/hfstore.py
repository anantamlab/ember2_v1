"""Private Hugging Face artifact store (run 19: laptop = development, workstation = execution only).

One *artifact* = one immutable folder `<path>` in one private HF repo:

  <path>/<payload files>      the exact local bytes
  <path>/ARTIFACT.json        manifest: identity, status, provenance, every file's size / sha256 / git sha1
  <path>/VERIFIED.json        written only after the remote check of the manifest commit passed

Publishing order: payload (resumable multi-commit `upload_folder`; a re-run skips what is already there) ->
manifest commit -> remote verification at that exact commit (file set, sizes, LFS sha256 or git blob sha1,
plus a downloaded content-hash check of the manifest, every critical file and a sample) -> VERIFIED.json.
An artifact is COMPLETE in the registry only after that. A path that already holds a *different* manifest is
refused: versions are immutable (publish a new `@N` path instead). Re-publishing the same manifest is a no-op.

The registry (`data/hf/registry.json`) is the machine-readable upload manifest: repo ids, manifest sha256,
manifest / verification commits, file counts and the verification result of every artifact.

Works with huggingface_hub 1.x (workstation lock) and 2.x (laptop tools venv). Tokens come from the normal
HF login / HF_TOKEN; they are never written to any file this module produces.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

VERSION = "ember-hfstore@1.0.0"
MANIFEST = "ARTIFACT.json"
VERIFIED = "VERIFIED.json"
REPOS = {"datasets": ("ember-v1-datasets", "dataset"), "models": ("ember-v1-models", "model"),
         "experiments": ("ember-v1-experiments", "dataset")}
STATUSES = ("COMPLETE", "INCOMPLETE", "SUPERSEDED", "PREPARED_NOT_RUN", "RESULT", "FAILED")
REGISTRY = Path("data/hf/registry.json")
# Never uploaded: temp files, caches, the hub's own local-dir metadata.
EXCLUDE = ("*.tmp", "*.pyc", "*/__pycache__/*", "__pycache__/*", ".cache/*", "*/.cache/*", "*.lock", "*.partial")
# Credentials that must never leave the laptop (HF, Modal, GitHub, OpenAI-style keys).
SECRET_RE = re.compile(rb"hf_[A-Za-z0-9]{30,}|\b(?:ak|as)-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}"
                       rb"|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}")
# Manifests / identity files: always downloaded and content-hashed (plus a seeded sample of the rest).
CRITICAL = ("SHA256SUMS", "MANIFEST.sha256", "*manifest*.json", "dataset.json", "plan.json", "progress.json",
            "evidence.json", "summary.json", "*_report.json", "*config*.json", "batch.json", "*.safetensors",
            "*.jinja", "tokenizer.json", "identity.json", "*.sha256")
CRITICAL_MAX_BYTES = 512 * 2 ** 20
CHUNK = 8 * 2 ** 20


class StoreError(RuntimeError):
    """Refused or failed store operation (never silently ignored)."""


# ----- local hashing -----------------------------------------------------------------------------------------
def _excluded(rel: str, extra: Sequence[str] = ()) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in (*EXCLUDE, *extra)) or rel in (MANIFEST, VERIFIED)


def hash_file(path: Path, *, scan: bool = True) -> dict[str, Any]:
    """size, sha256 and git blob sha1 of one file in one pass; `secrets` = matched credential patterns."""
    size = path.stat().st_size
    s256, s1 = hashlib.sha256(), hashlib.sha1(b"blob %d\0" % size)
    hits: list[str] = []
    tail = b""
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            s256.update(b)
            s1.update(b)
            if scan:
                for m in SECRET_RE.finditer(tail + b):
                    hits.append(m.group(0)[:6].decode("ascii", "replace") + "...")
                tail = b[-96:]
    return {"size": size, "sha256": s256.hexdigest(), "git_sha1": s1.hexdigest(), "secrets": sorted(set(hits))}


def file_table(root: Path, *, exclude: Sequence[str] = (), log: Callable[[str], None] = lambda s: None
               ) -> list[dict[str, Any]]:
    """Every payload file under `root` (posix relative paths, sorted). Refuses a payload with a credential."""
    root = Path(root)
    if not root.is_dir():
        raise StoreError(f"{root}: not a directory")
    out, bad = [], []
    rels = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    for i, rel in enumerate(rels):
        if _excluded(rel, exclude):
            continue
        h = hash_file(root / rel)
        if h.pop("secrets"):
            bad.append(rel)
        out.append({"path": rel, **h})
        if i % 500 == 499:
            log(f"hashed {i + 1}/{len(rels)}")
    if bad:
        raise StoreError(f"credential-like strings in {bad[:5]} (+{max(0, len(bad) - 5)}); nothing uploaded")
    if not out:
        raise StoreError(f"{root}: no payload files")
    return out


def files_digest(files: Sequence[Mapping[str, Any]]) -> str:
    canon = json.dumps([[f["path"], f["size"], f["sha256"]] for f in files], separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def dumps(doc: Any) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def build_manifest(*, artifact_id: str, kind: str, status: str, repo_id: str, repo_type: str, path: str,
                   files: Sequence[Mapping[str, Any]], source: Mapping[str, Any],
                   provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if status not in STATUSES:
        raise StoreError(f"status {status!r} not in {STATUSES}")
    if not re.fullmatch(r"[A-Za-z0-9._@+-]+", artifact_id):
        raise StoreError(f"bad artifact id {artifact_id!r}")
    if path.startswith("/") or ".." in path.split("/") or "\\" in path:
        raise StoreError(f"bad repo path {path!r}")
    return {"manifest_version": VERSION, "artifact_id": artifact_id, "kind": kind, "status": status,
            "repo": {"id": repo_id, "type": repo_type, "path": path}, "source": dict(source),
            "provenance": dict(provenance or {}),
            "totals": {"files": len(files), "bytes": sum(f["size"] for f in files)},
            "files_digest": files_digest(files), "files": [dict(f) for f in files]}


def verify_local_tree(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Every manifest file present with its size and sha256; no extra payload file."""
    want = {f["path"]: f for f in manifest["files"]}
    have = {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()}
    have = {r for r in have if not _excluded(r)}
    missing, extra = sorted(set(want) - have), sorted(have - set(want))
    bad = []
    for rel, f in want.items():
        if rel in have:
            h = hash_file(Path(root) / rel, scan=False)
            if h["size"] != f["size"] or h["sha256"] != f["sha256"]:
                bad.append(rel)
    ok = not (missing or extra or bad)
    return {"result": "PASS" if ok else "FAIL", "files": len(want), "missing": missing[:20],
            "extra": extra[:20], "hash_mismatch": bad[:20]}


# ----- remote ------------------------------------------------------------------------------------------------
def api() -> Any:
    from huggingface_hub import HfApi
    return HfApi()


def namespace(hf: Any) -> str:
    return hf.whoami()["name"]


def repo_ids(ns: str) -> dict[str, tuple[str, str]]:
    return {k: (f"{ns}/{name}", t) for k, (name, t) in REPOS.items()}


README = """---
license: other
---
# {name} (private)

EMBER v1 {what}. Managed by `python -m ember_v1.hfstore` (repository anantamlab/ember_v1).
Every artifact is an immutable folder with `ARTIFACT.json` (manifest + hashes) and `VERIFIED.json`
(remote verification). Never overwrite a folder: publish a new `@N` version.
"""


def init_repos(hf: Any, ns: str) -> dict[str, Any]:
    """Create (or reuse) the three PRIVATE repos; refuse a public one."""
    out = {}
    what = {"datasets": "datasets and teacher evidence", "models": "student checkpoints (adapters + heads)",
            "experiments": "execution artifacts (evaluations, rollouts, reports)"}
    for key, (rid, rtype) in repo_ids(ns).items():
        hf.create_repo(rid, repo_type=rtype, private=True, exist_ok=True)
        info = hf.repo_info(rid, repo_type=rtype)
        if not getattr(info, "private", False):
            raise StoreError(f"{rid} is NOT private; stop and fix visibility on the Hub")
        if not hf.file_exists(rid, "README.md", repo_type=rtype):
            hf.upload_file(path_or_fileobj=README.format(name=rid.split("/")[1], what=what[key]).encode(),
                           path_in_repo="README.md", repo_id=rid, repo_type=rtype,
                           commit_message="repo card (private EMBER v1 store)")
            info = hf.repo_info(rid, repo_type=rtype)
        out[key] = {"id": rid, "type": rtype, "private": True, "head": info.sha}
    return out


def _download(hf: Any, rid: str, rtype: str, rel: str, rev: str, tmp: Path) -> Path:
    return Path(hf.hf_hub_download(rid, rel, repo_type=rtype, revision=rev, cache_dir=str(tmp)))


def _not_found() -> tuple[type[BaseException], ...]:
    try:
        from huggingface_hub.utils import EntryNotFoundError
        return (EntryNotFoundError, FileNotFoundError)
    except ImportError:                                  # test doubles raise FileNotFoundError
        return (FileNotFoundError,)


def _remote_manifest(hf: Any, rid: str, rtype: str, path: str, rev: str = "main") -> tuple[bytes | None, Any]:
    EntryNotFoundError = _not_found()
    try:
        with tempfile.TemporaryDirectory() as td:
            data = _download(hf, rid, rtype, f"{path}/{MANIFEST}", rev, Path(td)).read_bytes()
    except EntryNotFoundError:
        return None, None
    try:
        with tempfile.TemporaryDirectory() as td:
            ver = json.loads(_download(hf, rid, rtype, f"{path}/{VERIFIED}", rev, Path(td)).read_bytes())
    except EntryNotFoundError:
        ver = None
    return data, ver


def _lfs_sha256(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    return lfs.get("sha256") if isinstance(lfs, Mapping) else getattr(lfs, "sha256", None)


def remote_files(hf: Any, rid: str, rtype: str, path: str, rev: str) -> dict[str, Any]:
    return {e.path[len(path) + 1:]: e for e in hf.list_repo_tree(rid, path, recursive=True, revision=rev,
                                                                   repo_type=rtype)
            if hasattr(e, "size") and not hasattr(e, "tree_id")}


def verify_remote(hf: Any, manifest: Mapping[str, Any], manifest_bytes: bytes, rev: str, *,
                  sample: int = 6, sample_max_bytes: int = 256 * 2 ** 20, full: bool = False,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """Check the artifact at commit `rev`: exact file set, sizes, LFS sha256 / git blob sha1 from the Hub's
    tree metadata, then download and sha256 the manifest, critical files and a seeded sample (all if `full`)."""
    rid, rtype, path = manifest["repo"]["id"], manifest["repo"]["type"], manifest["repo"]["path"]
    tree = remote_files(hf, rid, rtype, path, rev)
    want = {f["path"]: f for f in manifest["files"]}
    remote = {k: v for k, v in tree.items() if k not in (MANIFEST, VERIFIED)}
    problems = []
    missing, extra = sorted(set(want) - set(remote)), sorted(set(remote) - set(want))
    if missing:
        problems.append(f"missing {len(missing)}: {missing[:5]}")
    if extra:
        problems.append(f"extra {len(extra)}: {extra[:5]}")
    meta_checked = {"lfs_sha256": 0, "git_sha1": 0}
    for rel, f in want.items():
        e = remote.get(rel)
        if e is None:
            continue
        if e.size != f["size"]:
            problems.append(f"{rel}: size {e.size} != {f['size']}")
            continue
        lfs = _lfs_sha256(e)
        if lfs is not None:
            meta_checked["lfs_sha256"] += 1
            if lfs != f["sha256"]:
                problems.append(f"{rel}: lfs sha256 differs")
        else:
            meta_checked["git_sha1"] += 1
            if getattr(e, "blob_id", None) != f["git_sha1"]:
                problems.append(f"{rel}: git blob sha1 differs")
    if MANIFEST not in tree:
        problems.append("ARTIFACT.json absent at this revision")
    # downloaded content hashes
    crit = [r for r, f in want.items() if f["size"] <= CRITICAL_MAX_BYTES
            and any(fnmatch.fnmatch(r.rsplit("/", 1)[-1], p) for p in CRITICAL)]
    rest = sorted(set(want) - set(crit))
    rng = random.Random(manifest["artifact_id"])
    pick = list(rest) if full else rng.sample(rest, min(sample, len(rest)))
    if not full:
        budget, chosen = sample_max_bytes, []
        for r in sorted(pick, key=lambda r: want[r]["size"]):
            if want[r]["size"] <= budget:
                chosen.append(r)
                budget -= want[r]["size"]
        pick = chosen
    content = {"checked": 0, "bytes": 0, "mismatch": []}
    with tempfile.TemporaryDirectory() as td:
        got = _download(hf, rid, rtype, f"{path}/{MANIFEST}", rev, Path(td)).read_bytes()
        if got != manifest_bytes:
            problems.append("downloaded ARTIFACT.json differs from the local manifest bytes")
        for rel in sorted(set(crit) | set(pick)):
            if rel not in remote:
                continue
            p = _download(hf, rid, rtype, f"{path}/{rel}", rev, Path(td))
            h = hash_file(p, scan=False)
            content["checked"] += 1
            content["bytes"] += h["size"]
            if h["sha256"] != want[rel]["sha256"]:
                content["mismatch"].append(rel)
            p.unlink(missing_ok=True)
    if content["mismatch"]:
        problems.append(f"content sha256 mismatch: {content['mismatch'][:5]}")
    log(f"verify {manifest['artifact_id']} @ {rev[:12]}: {len(want)} files, meta {meta_checked}, "
        f"content {content['checked']} files / {content['bytes']} B, problems {len(problems)}")
    return {"result": "PASS" if not problems else "FAIL", "revision": rev, "files": len(want),
            "metadata_checked": meta_checked, "content_checked": content, "problems": problems,
            "manifest_sha256": sha256_bytes(manifest_bytes), "verifier": VERSION}


def _commit_oid(info: Any) -> str:
    oid = getattr(info, "oid", None) or getattr(info, "commit_id", None)
    if oid is None and isinstance(info, str):          # upload_file may return a URL string
        oid = info.rsplit("/", 1)[-1]
    if not (isinstance(oid, str) and re.fullmatch(r"[0-9a-f]{40}", oid)):
        raise StoreError(f"cannot read the commit id from {info!r}")
    return oid


def publish(hf: Any, manifest: Mapping[str, Any], local_root: Path, *, full_verify: bool = False,
            exclude: Sequence[str] = (), log: Callable[[str], None] = print) -> dict[str, Any]:
    """Upload, seal, verify. Idempotent; refuses to replace a different manifest at the same path."""
    rid, rtype, path = manifest["repo"]["id"], manifest["repo"]["type"], manifest["repo"]["path"]
    mbytes = dumps(manifest)
    msha = sha256_bytes(mbytes)
    old, ver = _remote_manifest(hf, rid, rtype, path)
    if old is not None and old != mbytes:
        raise StoreError(f"{rid}/{path} already holds another manifest ({sha256_bytes(old)}); versions are "
                         f"immutable: publish under a new @N path")
    if old is not None and ver and ver.get("result") == "PASS" and ver.get("manifest_sha256") == msha:
        log(f"{manifest['artifact_id']}: already published and verified (manifest commit {ver['revision'][:12]})")
        return entry(manifest, msha, {**ver, "verified_commit": _last_commit(hf, rid, rtype, f"{path}/{VERIFIED}")})
    lv = verify_local_tree(local_root, manifest)
    if lv["result"] != "PASS":
        raise StoreError(f"local tree differs from its manifest: {lv}")
    if old is None:
        t = time.monotonic()
        log(f"{manifest['artifact_id']}: uploading {manifest['totals']['files']} files "
            f"({manifest['totals']['bytes'] / 2 ** 30:.2f} GiB) to {rid}/{path}")
        hf.upload_folder(repo_id=rid, repo_type=rtype, folder_path=str(local_root), path_in_repo=path,
                         ignore_patterns=[*EXCLUDE, *exclude, MANIFEST, VERIFIED],
                         commit_message=f"{manifest['artifact_id']}: payload")
        log(f"{manifest['artifact_id']}: payload uploaded in {time.monotonic() - t:.0f} s")
        info = hf.upload_file(path_or_fileobj=mbytes, path_in_repo=f"{path}/{MANIFEST}", repo_id=rid,
                              repo_type=rtype, commit_message=f"{manifest['artifact_id']}: manifest {msha[7:19]}")
        rev = _commit_oid(info)
    else:  # sealed earlier, verification not recorded (interrupted): find the sealing commit
        rev = _manifest_commit(hf, rid, rtype, path)
    res = verify_remote(hf, manifest, mbytes, rev, full=full_verify, log=log)
    if res["result"] != "PASS":
        raise StoreError(f"remote verification FAILED for {manifest['artifact_id']}: {res['problems'][:5]}")
    res["verified_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    info = hf.upload_file(path_or_fileobj=dumps(res), path_in_repo=f"{path}/{VERIFIED}", repo_id=rid,
                          repo_type=rtype, commit_message=f"{manifest['artifact_id']}: verified PASS")
    res["verified_commit"] = _commit_oid(info)
    return entry(manifest, msha, res)


def _last_commit(hf: Any, rid: str, rtype: str, rel: str) -> str:
    """The commit that last changed `rel` on main."""
    tree = hf.get_paths_info(rid, [rel], expand=True, repo_type=rtype)
    lc = getattr(tree[0], "last_commit", None) if tree else None
    oid = getattr(lc, "oid", None) if lc is not None else None
    if not oid:
        raise StoreError(f"cannot find the last commit of {rid}/{rel}")
    return oid


def _manifest_commit(hf: Any, rid: str, rtype: str, path: str) -> str:
    return _last_commit(hf, rid, rtype, f"{path}/{MANIFEST}")


def entry(manifest: Mapping[str, Any], msha: str, ver: Mapping[str, Any]) -> dict[str, Any]:
    return {"artifact_id": manifest["artifact_id"], "kind": manifest["kind"], "status": manifest["status"],
            "repo_id": manifest["repo"]["id"], "repo_type": manifest["repo"]["type"],
            "path": manifest["repo"]["path"], "manifest_sha256": msha, "manifest_commit": ver["revision"],
            "verified_commit": ver.get("verified_commit"), "verification": ver["result"],
            "files": manifest["totals"]["files"], "bytes": manifest["totals"]["bytes"],
            "files_digest": manifest["files_digest"], "source_commit": manifest["source"].get("git_commit")}


# ----- fetch (workstation and laptop) -------------------------------------------------------------------------
def fetch(hf: Any, *, repo_id: str, repo_type: str, path: str, revision: str, dest: Path,
          manifest_sha256: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Download `<path>` at an exact commit into `dest/<path>` and verify every file against ARTIFACT.json
    (and ARTIFACT.json against `manifest_sha256` when given). Re-running resumes / re-verifies."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise StoreError(f"revision must be a full 40-hex commit, got {revision!r}")
    snap = getattr(hf, "snapshot_download", None)
    if callable(snap):
        snap(repo_id=repo_id, repo_type=repo_type, revision=revision, local_dir=str(dest), allow_patterns=[f"{path}/*"])
    else:                                                # clients without snapshot_download: file by file
        for rel in remote_files(hf, repo_id, repo_type, path, revision):
            hf.hf_hub_download(repo_id, f"{path}/{rel}", repo_type=repo_type, revision=revision, local_dir=str(dest))
    root = Path(dest) / path
    mb = (root / MANIFEST).read_bytes()
    msha = sha256_bytes(mb)
    if manifest_sha256 and msha != manifest_sha256:
        raise StoreError(f"{repo_id}/{path}@{revision[:12]}: manifest {msha} != expected {manifest_sha256}")
    man = json.loads(mb)
    lv = verify_local_tree(root, man)
    log(f"fetch {man['artifact_id']}: {lv['result']} ({lv['files']} files)")
    if lv["result"] != "PASS":
        raise StoreError(f"downloaded artifact fails its manifest: {lv}")
    return {"artifact_id": man["artifact_id"], "root": str(root), "manifest_sha256": msha, "revision": revision,
            "status": man["status"], "local_verification": lv}


# ----- local evidence checks (reuse the file-table hashes; no second read) ------------------------------------
def local_checks(files: Sequence[Mapping[str, Any]], root: Path, checks: Sequence[Mapping[str, Any]]
                 ) -> dict[str, Any]:
    """`sha256sums` (`<hex> *./rel` lines, relative to the sums file), `sft_manifest` (its `files` map) and
    `record_files` (dataset.json group -> records/<group>.jsonl). Any mismatch or missing file = FAIL."""
    by = {f["path"]: f["sha256"] for f in files}
    out = []
    for c in checks:
        name, kind = c["file"], c["type"]
        base = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
        want: dict[str, str] = {}
        if kind == "sha256sums":
            for line in (Path(root) / name).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    h, _, p = line.strip().partition(" ")
                    want[base + p.lstrip("*").strip().removeprefix("./")] = h
        elif kind == "sft_manifest":
            doc = json.loads((Path(root) / name).read_text(encoding="utf-8"))
            want = {base + k: v.removeprefix("sha256:") for k, v in doc["files"].items()}
        elif kind == "record_files":
            doc = json.loads((Path(root) / name).read_text(encoding="utf-8"))
            want = {f"{base}records/{g}.jsonl": v.removeprefix("sha256:") for g, v in doc["record_files"].items()}
        else:
            raise StoreError(f"unknown local check {kind!r}")
        missing = sorted(p for p in want if p not in by)
        bad = sorted(p for p in want if p in by and by[p] != want[p])
        out.append({"file": name, "type": kind, "entries": len(want), "missing": missing[:10],
                    "mismatch": bad[:10], "result": "PASS" if not (missing or bad) else "FAIL"})
    return {"result": "PASS" if all(o["result"] == "PASS" for o in out) else "FAIL", "checks": out}


def publish_plan(hf: Any, plan: Mapping[str, Any], evidence_root: Path, reg_path: Path, *,
                 repo: str | None = None, only: Sequence[str] = (), log: Callable[[str], None] = print) -> int:
    """Publish the plan's artifacts in order (one repo at a time keeps commits sequential). Skips artifacts
    already verified in the registry; stops at the first failure (nothing later is attempted)."""
    reg = load_registry(reg_path)
    ns = reg.get("namespace") or ""
    for it in plan["artifacts"]:
        if (repo and it["repo"] != repo) or (only and it["artifact_id"] not in only):
            continue
        done = reg["artifacts"].get(it["artifact_id"])
        if done and done["verification"] == "PASS":
            log(f"skip {it['artifact_id']}: verified at {done['manifest_commit'][:12]}")
            continue
        local = Path(evidence_root) / it["local"]
        rid, rtype = repo_ids(ns)[it["repo"]]
        files = file_table(local, exclude=it.get("exclude", ()), log=log)
        prov = dict(it.get("provenance", {}))
        if it.get("checks"):
            lc = local_checks(files, local, it["checks"])
            prov["local_checks"] = lc
            if lc["result"] != "PASS":
                raise StoreError(f"{it['artifact_id']}: local evidence check FAILED: {lc}")
        man = build_manifest(artifact_id=it["artifact_id"], kind=it["kind"], status=it["status"], repo_id=rid,
                             repo_type=rtype, path=it["path"], files=files,
                             source={"git_commit": it.get("source_commit"), "uploader": VERSION,
                                     "local_name": it["local"]}, provenance=prov)
        keep = Path(evidence_root) / "hf-manifests" / f"{it['artifact_id']}.json"
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_bytes(dumps(man))
        e = publish(hf, man, local, full_verify=it.get("full_verify", False), exclude=it.get("exclude", ()),
                    log=log)
        reg = load_registry(reg_path)
        record(reg, e)
        save_registry(reg, reg_path)
        log(f"DONE {it['artifact_id']} manifest {e['manifest_commit'][:12]} verified {e['verified_commit'][:12]}")
    return 0


# ----- registry ----------------------------------------------------------------------------------------------
def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"registry_version": VERSION, "namespace": None, "repos": {}, "artifacts": {}}


def save_registry(reg: Mapping[str, Any], path: Path = REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(dumps(reg))
    os.replace(tmp, path)


def record(reg: dict[str, Any], e: Mapping[str, Any]) -> None:
    old = reg["artifacts"].get(e["artifact_id"])
    if old and old["manifest_sha256"] != e["manifest_sha256"]:
        raise StoreError(f"registry already has {e['artifact_id']} with another manifest")
    reg["artifacts"][e["artifact_id"]] = dict(e)


# ----- CLI ---------------------------------------------------------------------------------------------------
def _parse_kv(items: Iterable[str]) -> dict[str, str]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not k or not _:
            raise StoreError(f"--prov needs key=value, got {it!r}")
        out[k] = v
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.hfstore", description=__doc__.split("\n")[0])
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-repos", help="create / reuse the three private repos and record them")
    p = sub.add_parser("publish", help="hash, upload, seal and verify one artifact folder")
    p.add_argument("--artifact-id", required=True)
    p.add_argument("--repo", required=True, choices=sorted(REPOS))
    p.add_argument("--path", required=True, help="immutable folder path in the repo (ends with the @version)")
    p.add_argument("--local", required=True, type=Path)
    p.add_argument("--kind", required=True)
    p.add_argument("--status", required=True, choices=STATUSES)
    p.add_argument("--source-commit", default="")
    p.add_argument("--prov", action="append", default=[], help="provenance key=value (repeatable)")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--full-verify", action="store_true", help="download and hash every file after upload")
    p.add_argument("--manifest-out", type=Path, help="also keep the manifest here (local record)")
    v = sub.add_parser("verify", help="re-verify a registered artifact at its manifest commit")
    v.add_argument("artifact_id")
    v.add_argument("--full", action="store_true")
    f = sub.add_parser("fetch", help="download a registered artifact at its exact commit and verify it")
    f.add_argument("artifact_id")
    f.add_argument("--dest", required=True, type=Path)
    sub.add_parser("status", help="print the registry summary")
    pp = sub.add_parser("publish-plan", help="publish every planned artifact not yet verified (resumable)")
    pp.add_argument("--plan", type=Path, default=Path("data/hf/publish_plan.json"))
    pp.add_argument("--evidence-root", required=True, type=Path)
    pp.add_argument("--repo", choices=sorted(REPOS))
    pp.add_argument("--only", default="", help="comma-separated artifact ids")
    a = ap.parse_args(argv)
    reg = load_registry(a.registry)
    try:
        if a.cmd == "status":
            for k, e in sorted(reg["artifacts"].items()):
                print(f"{e['verification']:4} {e['status']:16} {e['repo_id']}:{e['path']} "
                      f"{e['files']} files {e['bytes'] / 2 ** 20:.1f} MiB @ {e['manifest_commit'][:12]}")
            return 0
        hf = api()
        if a.cmd == "publish-plan":
            plan = json.loads(a.plan.read_text(encoding="utf-8"))
            return publish_plan(hf, plan, a.evidence_root, a.registry, repo=a.repo,
                                only=[x for x in a.only.split(",") if x])
        if a.cmd == "init-repos":
            ns = namespace(hf)
            reg["namespace"], reg["repos"] = ns, init_repos(hf, ns)
            save_registry(reg, a.registry)
            print(json.dumps(reg["repos"], indent=1))
            return 0
        if a.cmd == "publish":
            if not reg.get("namespace"):
                raise StoreError("run init-repos first")
            rid, rtype = repo_ids(reg["namespace"])[a.repo]
            files = file_table(a.local, exclude=a.exclude, log=print)
            man = build_manifest(artifact_id=a.artifact_id, kind=a.kind, status=a.status, repo_id=rid,
                                 repo_type=rtype, path=a.path, files=files,
                                 source={"git_commit": a.source_commit or None, "uploader": VERSION},
                                 provenance=_parse_kv(a.prov))
            if a.manifest_out:
                a.manifest_out.parent.mkdir(parents=True, exist_ok=True)
                a.manifest_out.write_bytes(dumps(man))
            e = publish(hf, man, a.local, full_verify=a.full_verify, exclude=a.exclude)
            reg = load_registry(a.registry)            # other publishes may have run meanwhile
            record(reg, e)
            save_registry(reg, a.registry)
            print(json.dumps(e, indent=1))
            return 0
        e = reg["artifacts"].get(a.artifact_id)
        if e is None:
            raise StoreError(f"{a.artifact_id} is not in {a.registry}")
        if a.cmd == "fetch":
            r = fetch(hf, repo_id=e["repo_id"], repo_type=e["repo_type"], path=e["path"],
                      revision=e["manifest_commit"], dest=a.dest, manifest_sha256=e["manifest_sha256"])
            print(json.dumps(r, indent=1))
            return 0
        with tempfile.TemporaryDirectory() as td:
            mb = _download(hf, e["repo_id"], e["repo_type"], f"{e['path']}/{MANIFEST}", e["manifest_commit"],
                           Path(td)).read_bytes()
        r = verify_remote(hf, json.loads(mb), mb, e["manifest_commit"], full=a.full)
        print(json.dumps(r, indent=1))
        return 0 if r["result"] == "PASS" else 1
    except StoreError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
