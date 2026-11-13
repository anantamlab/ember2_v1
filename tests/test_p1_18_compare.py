"""p1_18_compare: matched pairing by scenario hash, family-cluster bootstrap, missing pairs never imputed."""
from __future__ import annotations

import json

from ember_v1 import p1_18_compare as C


def _group(h, cu):
    pol = {t: {"outcome": {"people": 10, "evacuated": 10 - cu[i], "casualties": cu[i], "unresolved": 0,
                           "movement_outcomes": {"moved": 5, "blocked_wall": i}}, "failure": None}
           for i, t in enumerate(C.TEACHERS)}
    return {"group_complete": True, "scenario_hash": h, "policies": pol}


def test_compare_pairs_by_hash_and_reports_family_intervals(tmp_path):
    root = tmp_path / "root"
    (root / "groups").mkdir(parents=True)
    lst, stu = tmp_path / "list.jsonl", tmp_path / "s.jsonl"
    with open(lst, "w") as fl, open(stu, "w") as fs:
        for i in range(8):
            h = f"sha256:{i}"
            (root / "groups" / f"g{i}.json").write_text(json.dumps(_group(h, [5, 2, 3, 4, 2])))
            fl.write(json.dumps({"set": "x", "family_id": f"f{i // 2}", "scenario_id": f"s{i}",
                                 "scenario_hash": h}) + "\n")
            if i < 7:                                          # s7 missing for the student
                fs.write(json.dumps({"set": "x", "family_id": f"f{i // 2}", "scenario_id": f"s{i}",
                                     "scenario_hash": h, "people": 10, "evacuated": 9, "casualties": 1,
                                     "unresolved": 0, "movement_outcomes": {"blocked_occupied": 2},
                                     "failure": None}) + "\n")
    r = C.compare([lst], {"V1": stu}, [root])["sets"]["x"]
    assert r["scenarios"] == 7 and r["missing_pairs"] == ["s7"] and r["families"] == 4
    assert r["policies"]["T1"]["cu"] == 14 and r["policies"]["V1"]["cu"] == 7
    assert r["policies"]["best_T1-T4"]["cu"] == 14 and r["policies"]["V1"]["collisions"] == 14
    assert r["policies"]["T3"]["blocked"] == 21
    c = r["contrasts"]["V1-best_T1-T4"]["cu"]
    assert c["diff"] == -0.1 and c["ci95"] == [-0.1, -0.1] and c["excludes_zero"] and c["families"] == 4
