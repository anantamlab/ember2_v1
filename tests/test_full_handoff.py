import importlib.util
from pathlib import Path

import pytest

from ember_v1 import full_dagger as fd
from sim_helpers import scenario


def helper():
    path = Path(__file__).parents[1] / 'scripts/full_view_boundary_handoff.py'
    spec = importlib.util.spec_from_file_location('boundary_handoff', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('count,accepted', [(1, True), (2, False)])
def test_handoff_validates_completed_mock_dataset_and_frozen_family_count(tmp_path, count, accepted):
    module = helper()
    fixture = scenario(['.E'], {'A': (0, 0)}, max_steps=2)
    rows, stats = fd.collect([fixture])
    namespace = 'ember2-full-view-handoff-cpu'
    fd.write_dataset(tmp_path/'data', rows, {'split':'train','view_version':'full-view@1',
        'namespace':namespace,'count':count,'round':1,'beta':.5,'teacher':'T1',
        'stats':stats,'topology_hashes':[fd.topology(fixture)]})
    plan = {'teacher':{'name':'T1'}, 'dagger':{'rounds':[{}, {
        'namespace':namespace,'count':1,'beta':.5}]}}
    panels = {'dagger-r1':[fd.topology(fixture)],'select':['other'],'report':['another']}
    if accepted:
        result = module.verify_collection(tmp_path/'data', plan, panels, 1)
        assert result['families'] == 1 and result['rows'] == 1
    else:
        with pytest.raises(ValueError, match='count'):
            module.verify_collection(tmp_path/'data', plan, panels, 1)
