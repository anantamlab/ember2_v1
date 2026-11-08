"""CPU-only wheel provenance inspection for the P0B lock. Reads zip central directories over HTTP range
requests (and selected members); never installs, imports or runs package code.

Usage: python -I experiments/p0b_wheel_inspect.py infra/p0b-requirements.lock.txt <out.json>
(Output committed as data/p0b/p0b-wheel-provenance.json, with the lock path made relative.)
"""
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile

UA = {"User-Agent": "ember-v1-wheel-inspect/1.0"}
TAGS_OK = re.compile(r"-(cp311|py3|py2\.py3|cp39-abi3|cp38-abi3|cp310-abi3|cp311-abi3|cp37-abi3|cp36-abi3)-")
PLAT_OK = re.compile(r"(manylinux[^.]*x86_64|none-any|linux_x86_64)\.whl$")
NEEDLES = {
    "fla_gated_delta_kernel": rb"chunk_gated_delta_rule_fwd_kernel_h",
    "fla_kkt_solve_kernel": rb"chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "fla_chunk_fwd_kernel_o": rb"def chunk_fwd_kernel_o",
    "ChunkGatedDeltaRuleFunction": rb"class ChunkGatedDeltaRuleFunction",
    "causal_conv1d_fwd_kernel": rb"def _causal_conv1d_fwd_kernel",
    "import_fla": rb"(?m)^\s*(from fla[ .]|import fla\b)",
    "import_causal_conv1d": rb"(?m)^\s*(from causal_conv1d|import causal_conv1d)",
    "fla_in_string": rb"[\"']fla[\"']",
    "kernels_hub": rb"get_kernel\(|from kernels import|kernels-community",
}


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.fetched = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def lock_entries(path):
    text = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"(?m)^([A-Za-z0-9_.-]+)==([^ \\\n]+)((?:\s*\\\n\s+--hash=sha256:[0-9a-f]{64})+)", text):
        out[m.group(1)] = {"version": m.group(2), "hashes": re.findall(r"sha256:([0-9a-f]{64})", m.group(3))}
    return out


def pick(name, version, hashes):
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        files = json.load(r)["urls"]
    cands = [f for f in files if f["digests"]["sha256"] in hashes and f["filename"].endswith(".whl")
             and TAGS_OK.search(f["filename"]) and PLAT_OK.search(f["filename"])]
    cands.sort(key=lambda f: ("cp311" not in f["filename"], f["filename"]))
    return cands[0] if cands else None


def inspect(name, version, hashes, full_limit):
    f = pick(name, version, hashes)
    if f is None:
        return {"version": version, "wheel": None, "note": "no locked linux/py3.11 wheel found"}
    rf = RangeFile(f["url"])
    full = rf.size <= full_limit
    if full:
        data = urllib.request.urlopen(urllib.request.Request(f["url"], headers=UA), timeout=300).read()
        sha_ok = hashlib.sha256(data).hexdigest() == f["digests"]["sha256"]
        zf = zipfile.ZipFile(io.BytesIO(data))
    else:
        sha_ok = None
        zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
    names = zf.namelist()
    tops = sorted({n.split("/")[0] for n in names})
    rec = {"version": version, "wheel": f["filename"], "size": rf.size, "pypi_sha256": f["digests"]["sha256"],
           "pypi_sha256_in_lock": f["digests"]["sha256"] in hashes,
           "downloaded_sha256_ok": sha_ok, "mode": "full" if full else "central_directory_only",
           "members": len(names), "top_level": tops,
           "fla_paths": [n for n in names if re.search(r"(^|/)fla(/|$)", n)][:20],
           "conv1d_paths": [n for n in names if "causal_conv" in n.lower()][:20],
           "gated_delta_paths": [n for n in names if "gated_delta" in n.lower() or "delta_rule" in n.lower()][:20]}
    meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
    if meta and (full or zf.getinfo(meta).file_size < 1 << 20):
        md = zf.read(meta).decode("utf-8", "replace")
        rec["requires_dist_kernel"] = [l for l in md.splitlines() if l.startswith("Requires-Dist") and re.search(
            r"flash-linear|fla-core|fla\b|causal.conv|kernels", l, re.I)]
    if full:
        hits = {}
        for n in names:
            if not n.endswith(".py"):
                continue
            src = zf.read(n)
            for key, pat in NEEDLES.items():
                if re.search(pat, src):
                    hits.setdefault(key, []).append(n)
        rec["source_hits"] = {k: v[:15] for k, v in sorted(hits.items())}
    rec["bytes_fetched"] = rf.fetched if not full else rf.size
    return rec


def main(lock, out):
    entries = lock_entries(lock)
    res = {}
    for name in sorted(entries):
        e = entries[name]
        limit = 60 * 2**20
        try:
            res[name] = inspect(name, e["version"], set(e["hashes"]), limit)
        except Exception as ex:  # noqa: BLE001
            res[name] = {"version": e["version"], "error": f"{type(ex).__name__}: {ex}"}
        r = res[name]
        print(name, r.get("mode"), r.get("size"), r.get("fla_paths", [])[:2], r.get("conv1d_paths", [])[:2],
              sorted(r.get("source_hits", {}))[:6], r.get("error", ""), flush=True)
    json.dump({"lock": lock, "packages": res}, open(out, "w", encoding="utf-8"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
