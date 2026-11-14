import pytest

from ember_v1.full_evidence import assert_disjoint,file_table,paired_intervals,verify_table,write_once


def test_records_are_write_once_and_content_verified(tmp_path):
    write_once(tmp_path/'record.json',{'n':2});table=file_table(tmp_path);verify_table(tmp_path,table)
    with pytest.raises(FileExistsError):write_once(tmp_path/'record.json',{'n':3})
    (tmp_path/'record.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):verify_table(tmp_path,table)
    with pytest.raises(ValueError):verify_table(tmp_path,[{'path':'../escape','size':0,'sha256':'x'}])


def test_topology_overlap_fails_closed():
    assert assert_disjoint({'train':['a','b'],'select':['c'],'report':['d']})['overlap']==0
    with pytest.raises(ValueError,match='overlap'):assert_disjoint({'train':['a'],'report':['a']})
    with pytest.raises(ValueError,match='duplicate'):assert_disjoint({'train':['a','a']})


def test_paired_bootstrap_preserves_families_and_population_weighting():
    rows=[{'scenario_id':'a','people':1,'evacuated':1,'casualties':0,'unresolved':0},
          {'scenario_id':'b','people':9,'evacuated':0,'casualties':0,'unresolved':9}]
    same=paired_intervals(rows,list(reversed(rows)),resamples=5000)
    assert all(v=={'delta':0.,'low':0.,'high':0.} for v in same['metrics'].values())
    other=[{**r,'evacuated':0,'unresolved':r['people']} for r in rows]
    result=paired_intervals(rows,other,resamples=5000)
    assert result['metrics']['evacuated_rate']['delta']==pytest.approx(.1)
    assert result['metrics']['utility']['delta']==pytest.approx(.175)
    with pytest.raises(ValueError,match='identical'):paired_intervals(rows,other[:1])
