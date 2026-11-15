"""CPU regressions for the post-stop provenance and selection repairs."""
from pathlib import Path
import pytest
from ember_v1.full_evidence import validate_plan_roots, report_checkpoint, verify_original, select_ppo_seed, file_table, sha256, write_once

def test_declared_roots_reject_relocated_plan_and_markers(tmp_path):
    ev=tmp_path/'evidence';p=ev/'config/plan.json'
    plan={'source_head':'a'*40,'evidence_root':str(ev),'attempt_markers_root':str(ev/'attempts')}
    assert validate_plan_roots(p,plan)==ev.resolve()
    with pytest.raises(ValueError,match='root'):validate_plan_roots(tmp_path/'copy/config/plan.json',plan)
    with pytest.raises(ValueError,match='root'):validate_plan_roots(p,{**plan,'attempt_markers_root':str(tmp_path/'other')})
    with pytest.raises(ValueError,match='root'):validate_plan_roots(p,{'source_head':'a'*40})

def test_report_uses_frozen_arm_checkpoint_and_rejects_conflicting_cli(tmp_path):
    freeze={'checkpoint':str(tmp_path/'ppo'),'dagger_checkpoint':str(tmp_path/'dagger')}
    assert report_checkpoint('candidate_raw',freeze,None)==tmp_path/'ppo'
    assert report_checkpoint('candidate_masked',freeze,tmp_path/'ppo')==tmp_path/'ppo'
    assert report_checkpoint('dagger_raw',freeze,None)==tmp_path/'dagger'
    with pytest.raises(ValueError,match='checkpoint'):report_checkpoint('dagger_raw',freeze,tmp_path/'ppo')

def test_original_manifest_and_weights_rechecked_before_load(tmp_path):
    (tmp_path/'weights').write_bytes(b'original');write_once(tmp_path/'ARTIFACT.json',{'files':file_table(tmp_path)})
    pinned={'root':str(tmp_path),'manifest_sha256':sha256(tmp_path/'ARTIFACT.json'),'files_checked':1}
    verify_original(pinned)
    (tmp_path/'weights').write_bytes(b'changed!')
    with pytest.raises(ValueError,match='hash'):verify_original(pinned)
    (tmp_path/'ARTIFACT.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'):verify_original(pinned)

def test_cross_seed_selection_uses_masked_parent_and_earliest_seed_tie():
    def seed(n,u,c):return {'guard_passed':True,'selected_utility':u,'selected':{'casualty_rate':c},'config':{'seed':n}}
    masked={'utility':.7,'casualty_rate':.1};raw={'utility':.9,'casualty_rate':0}
    a,b=seed(2,.8,.11),seed(0,.8,.11)
    assert select_ppo_seed([a,b],masked) is b
    assert select_ppo_seed([a,b],raw) is None
    assert select_ppo_seed([seed(0,.7,.1),seed(1,.9,.121)],masked) is None


def test_wrong_report_checkpoint_is_rejected_before_spending_arm(tmp_path,monkeypatch):
    from test_full_report_protocol import setup
    phase,run=setup(tmp_path,monkeypatch)
    real=phase.report_checkpoint
    def wrong(arm,freeze,requested):return real(arm,freeze,tmp_path/'other')
    monkeypatch.setattr(phase,'report_checkpoint',wrong)
    with pytest.raises(ValueError,match='checkpoint'):run('candidate_raw',tmp_path/'out.json')
    assert not (tmp_path/'attempts/report-candidate_raw.started.json').exists()
