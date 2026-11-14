import importlib.util
import json
from pathlib import Path

import pytest

from ember_v1 import full_dagger as fd
from sim_helpers import scenario


@pytest.mark.parametrize('last_loss,accepted', [(.5, True), (float('nan'), False)])
def test_fit_handoff_keeps_loss_and_dataset_hash_gates(tmp_path, last_loss, accepted):
    path = Path(__file__).parents[1]/'scripts/full_view_boundary_handoff.py'
    spec = importlib.util.spec_from_file_location('fit_handoff', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = scenario(['.E'], {'A': (0, 0)}, max_steps=2)
    rows, stats = fd.collect([fixture])
    data = tmp_path/'d0'
    manifest = fd.write_dataset(data, rows, {'split':'train','view_version':'full-view@1','stats':stats})
    output = tmp_path/'s0'
    output.mkdir()
    config = {'epochs':1,'grad_accum_draws':8,'lr':.0002,'head_lr':.0001,'warmup_ratio':.03}
    (output/'full-fit.json').write_text(json.dumps({'seed':0,'config':config,
        'datasets':[{'path':str(data),'rows_sha256':manifest['rows_sha256']}],
        'report':{'rows':1,'first_loss':1.,'last_loss':last_loss}}))
    if accepted:
        result = module.verify_fit(output, {'dagger':{'fit':config}}, 0)
        assert result['rows']==1 and result['last_loss']==.5
        assert any(item['path']=='full-fit.json' for item in result['checkpoint_files'])
    else:
        with pytest.raises(ValueError, match='loss'):
            module.verify_fit(output, {'dagger':{'fit':config}}, 0)
