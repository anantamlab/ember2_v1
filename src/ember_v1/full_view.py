"""Lossless full public observation per person; no hidden simulator fields."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .contract import ACTIONS, validate_observation
from .human_env import HumanEvacuationEnv, HumanRewardConfig
from .sim.movement import DELTAS, OUTCOMES

VERSION = 'full-view@1'
MOVES = ('north', 'south', 'east', 'west')


def neighbour_status(observation, person_id):
    obs = validate_observation(observation)
    person = next((p for p in obs['people'] if p['id'] == person_id and p['status'] == 'active'), None)
    if person is None:
        raise ValueError('full view requires an active person')
    r, c = person['position']
    walls = {tuple(p) for p in obs['map']['obstacles']}
    fire = {tuple(p) for p in obs['burning']}
    people = {tuple(p['position']) for p in obs['people'] if p['status'] == 'active'}
    exits = {tuple(p['position']) for p in obs['map']['exits']}
    result = {}
    for a in MOVES:
        dr, dc = DELTAS[a]; q = (r + dr, c + dc)
        result[a] = ('wall' if not (0 <= q[0] < obs['map']['height'] and 0 <= q[1] < obs['map']['width']) or q in walls
                     else 'fire' if q in fire else 'person' if q in people else 'exit' if q in exits else 'floor')
    return result


def validate_view(view: Mapping[str, Any]):
    if set(view) != {'view_version', 'observation', 'person_id', 'last_action', 'last_outcome', 'moves'} or view['view_version'] != VERSION:
        raise ValueError('not a full public view')
    validate_observation(view['observation'])
    if view['last_action'] is not None and view['last_action'] not in ACTIONS:
        raise ValueError('invalid last action')
    if view['last_outcome'] is not None and view['last_outcome'] not in OUTCOMES:
        raise ValueError('invalid last outcome')
    if (view['last_action'] is None) != (view['last_outcome'] is None):
        raise ValueError('last action and outcome must be paired')
    if view['moves'] != neighbour_status(view['observation'], view['person_id']):
        raise ValueError('neighbours differ from public observation')


class FullObserver:
    def reset(self):
        pass

    def observe(self, observation, record=None):
        obs = validate_observation(observation)
        views = {}
        for p in obs['people']:
            if p['status'] != 'active':
                continue
            pid = p['id']
            last = record is not None and pid in record['actions']
            views[pid] = {'view_version': VERSION, 'observation': deepcopy(obs), 'person_id': pid,
                          'last_action': record['actions'][pid] if last else None,
                          'last_outcome': record['movements'][pid]['outcome'] if last else None,
                          'moves': neighbour_status(obs, pid)}
        return views

    def plans(self):
        return None


class FullEvacuationEnv(HumanEvacuationEnv):
    """Reuse per-person rewards and unchanged simulator, replacing only the observer."""
    def __init__(self, scenario, reward=HumanRewardConfig()):
        super().__init__(scenario, reward=reward)
        self.observer = FullObserver()
