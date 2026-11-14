"""Exercise the original checkpoint bridge through the real CPU evaluation pipeline."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from ember_v1.full_evidence import sha256
from ember_v1 import student_train
from sim_helpers import scenario


class FixedScores(student_train.MockDecisionAdapter):
    def scores(self, row):
        return {person: {'east': 10., 'north': 4., 'south': 3., 'stay': 5., 'west': 2.}
                for person in row['questions']}


@pytest.mark.parametrize('arm,wall_blocks,stays', [
    ('original_v1_raw', 2, 0), ('original_v1_masked', 0, 2),
])
def test_original_bridge_executes_real_raw_and_masked_policies(tmp_path, monkeypatch, arm, wall_blocks, stays):
    # Calling the wrong policy API, or wiring the wrong arm, breaks these literal outcomes.
    path = Path(__file__).resolve().parents[1] / 'experiments/full_view_phase.py'
    spec = importlib.util.spec_from_file_location('full_phase_cpu_control', path)
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    fixture = scenario(['.#E', '...'], {'A': (0, 0)}, max_steps=2, sid='cpu-control')
    monkeypatch.setattr(phase, 'scenarios', lambda *args: [fixture])
    # Replace only GPU model loading; keep the original encoder, ArmPolicy and simulator real.
    monkeypatch.setattr(student_train, 'ClefLoraAdapter', lambda cfg, checkpoint: FixedScores())
    config = tmp_path / 'config'
    config.mkdir()
    panels = config / 'family-panels.json'
    panels.write_text(json.dumps({'select': [], 'report': [], 'teacher-reference': [], 'token-preflight': []}))
    plan = config / 'plan.json'
    plan.write_text(json.dumps({
        'status': 'frozen', 'id': 'ember2-v1-full-view@1', 'authorization': {'source': 'owner prompt'},
        'config_hashes': {'family-panels.json': sha256(panels)},
        'teacher': {'name': 'T1', 'seed': 0}, 'max_steps': 2,
        'panels': {'select': {'namespace': 'ember2-full-view-cpu-control', 'count': 1}},
        'original_v1': {'root': str(tmp_path / 'original')},
    }))
    plan.with_suffix('.sha256').write_text(sha256(plan) + '\n')
    out = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', [str(path), 'eval', '--out', str(out), '--plan', str(plan),
                                    '--split', 'select', '--arm', arm])
    phase.main()
    result = json.loads(out.read_text())
    assert result['summary']['episodes'] == 1
    assert result['summary']['people'] == 1
    assert result['summary']['unresolved'] == 1
    assert result['summary']['movements']['blocked_wall'] == wall_blocks
    assert result['summary']['movements']['stayed'] == stays
    assert result['arm'] == arm
