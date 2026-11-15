"""Compact lossless full-view Clef row and fixed-grid small-network features."""
from __future__ import annotations

import json

from .contract import ACTIONS
from .full_view import validate_view
from .human_encoding import QUESTION_ID, view_question
from .human_view import MOVES
from .sim.movement import OUTCOMES


def encode_view(view):
    validate_view(view)
    obs = view['observation']; m = obs['map']
    grid = [['.' for _ in range(m['width'])] for _ in range(m['height'])]
    for r, c in m['obstacles']:
        grid[r][c] = '#'
    # Terrain, fire, exits and people remain separate; no overlay loses information.
    for r, c in obs['burning']:
        grid[r][c] = 'F'
    pos = lambda p: f"{p['position'][0]},{p['position'][1]}"
    return '\n'.join([
        f"FULL={obs['observation_version']} STEP={obs['step']} SIZE={m['height']},{m['width']} SELF={view['person_id']}",
        'RULES=' + json.dumps(obs['rules'], sort_keys=True, separators=(',', ':')),
        'MAP #=wall F=fire .=floor rows north-to-south columns west-to-east',
        *(''.join(row) for row in grid),
        'EXITS ' + ';'.join(f"{p['id']}={pos(p)}" for p in m['exits']),
        'PEOPLE ' + ';'.join(f"{p['id']}={pos(p)}:{p['status']}" for p in obs['people']),
        'ALLOWED ' + ','.join(obs['allowed_actions']),
        f"LAST action={view['last_action'] or 'none'} outcome={view['last_outcome'] or 'none'}",
        'NEXT ' + ' '.join(f'{a}={view["moves"][a]}' for a in MOVES)])


def view_row(view, gold):
    if gold not in ACTIONS:
        raise ValueError('unsupported full-view label')
    return {'state': encode_view(view), 'questions': {QUESTION_ID: view_question()}, 'gold': {QUESTION_ID: gold}}


def feature_size():
    return 10 * 16 * 16 + 6 + len(OUTCOMES) + 1 + 9 + 4 * 5


def view_features(view):
    validate_view(view); obs = view['observation']; m = obs['map']
    if m['height'] > 16 or m['width'] > 16:
        raise ValueError('small actor grid capacity exceeded; no truncation')
    walls = {tuple(p) for p in m['obstacles']}; fire = {tuple(p) for p in obs['burning']}
    exits = {tuple(p['position']) for p in m['exits']}
    active = {tuple(p['position']) for p in obs['people'] if p['status'] == 'active'}
    dead = {tuple(p['position']) for p in obs['people'] if p['status'] == 'casualty'}
    own = tuple(next(p['position'] for p in obs['people'] if p['id'] == view['person_id']))
    evacuated = {tuple(p['position']) for p in obs['people'] if p['status'] == 'evacuated'}
    sets = [walls, fire, exits, active, dead, {own}, evacuated, {(r,c) for r in range(m['height']) for c in range(m['width'])}]
    result = [float((r,c) in layer) for layer in sets for r in range(16) for c in range(16)]
    ranks = {tuple(p['position']): (i + 1) / len(obs['people']) for i,p in enumerate(obs['people']) if p['status'] == 'active'}
    exit_ranks = {tuple(p['position']): (i + 1) / max(1,len(m['exits'])) for i,p in enumerate(m['exits'])}
    result += [ranks.get((r,c),0.) for r in range(16) for c in range(16)]
    result += [exit_ranks.get((r,c),0.) for r in range(16) for c in range(16)]
    result += [float(view['last_action'] == a) for a in (None, *ACTIONS)]
    result += [float(view['last_outcome'] == a) for a in (None,*OUTCOMES)]
    rules = obs['rules']
    result += [obs['step']/rules['max_steps'], m['height']/16, m['width']/16,
               float(rules['fire_model'] == 'stochastic'), rules['spread_interval']/16,
               rules['spread_probability'] or 0., float(rules['exits_flammable']), float(rules['enter_fire'] == 'casualty'),rules['max_steps']/256]
    result += [float(view['moves'][a] == k) for a in MOVES for k in ('wall','fire','person','exit','floor')]
    assert len(result) == feature_size()
    return result
