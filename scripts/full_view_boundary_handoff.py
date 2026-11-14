"""Attach to one existing collection, preserve its completion, and retire its old controller."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from ember_v1.full_dagger import read_dataset
from ember_v1.full_evidence import assert_disjoint, sha256, write_once
from ember_v1.full_resume import HeldChild, process_snapshot


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify_collection(output, plan, panels, round_index):
    rows, meta = read_dataset(output)
    cfg = plan['dagger']['rounds'][round_index]
    expected = {'split':'train','namespace':cfg['namespace'],'count':cfg['count'],
                'round':round_index,'beta':cfg['beta'],'teacher':plan['teacher']['name']}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError('collection metadata mismatch: ' + key)
    if meta['stats']['episodes']['episodes'] != cfg['count']:
        raise ValueError('collection episode count mismatch')
    assert_disjoint(panels)
    if set(meta['topology_hashes']) != set(panels['dagger-r' + str(round_index)]):
        raise ValueError('collection topology hash mismatch')
    return {'families':cfg['count'],'rows':len(rows),'rows_sha256':meta['rows_sha256']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    repo, ev = args.repo.resolve(), args.evidence.resolve()
    assert ev.is_relative_to(Path('/home/darshith/ember-ws/ember2-evidence'))
    assert not ev.is_relative_to(repo)
    assert re.fullmatch(r'boundary-handoff-[0-9]+', args.name)
    match = re.fullmatch(r'dagger-r([0-3])-collect-[0-9]+', args.phase)
    assert match
    round_index = int(match[1])
    root = ev/'supervisor'/args.name
    root.mkdir(exist_ok=False)
    started = read(ev/'attempts'/f'{args.phase}.started.json')
    process = read(ev/'attempts'/f'{args.phase}.process.json')
    result_path = ev/'attempts'/f'{args.phase}.result.json'
    assert not result_path.exists() and not (ev/'supervisor/run.stopped.json').exists()
    plan = read(ev/'config/plan.json')
    plan_hash = sha256(ev/'config/plan.json')
    assert plan_hash == started['plan_sha256']
    assert plan['source_head'] == started['source_head']
    assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() == started['source_head']
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=repo, text=True)
    command = started['command']
    assert command[1:3] == ['experiments/full_view_phase.py', 'collect']
    assert command[command.index('--plan')+1] == str(ev/'config/plan.json')
    output = Path(command[command.index('--out')+1])
    assert output == ev/('d' + str(round_index))
    assert int(command[command.index('--round')+1]) == round_index
    parent_pid, child_pid = process['supervisor_pid'], process['pid']
    parent = process_snapshot(parent_pid)
    parent_command = parent['command']
    script = Path(parent_command[1])
    assert (repo/script).resolve() == repo/'scripts/full_view_supervisor.py'
    assert parent_command[parent_command.index('--repo')+1] == str(repo)
    assert parent_command[parent_command.index('--evidence')+1] == str(ev)
    child = process_snapshot(child_pid)
    assert child['command'] == command
    prior = [read(p) for p in (ev/'attempts').glob('*.result.json')]
    assert all(p['status']=='PASS' for p in prior)
    gpu = sum(p['seconds'] for p in prior if p['category']=='gpu')
    initial = read(ev/'config/budget-initial.json')
    cpu = initial['cpu_overhead_seconds'] + sum(p['seconds'] for p in prior if p['category']=='cpu')
    start_mono, cpu_start = time.monotonic(), time.process_time()
    age = max(float(Path('/proc/uptime').read_text().split()[0]) - child['start_ticks']/os.sysconf('SC_CLK_TCK'),
              time.time() - dt.datetime.fromisoformat(started['utc']).timestamp())
    # Include a second of prelaunch overhead; retain the original five-minute cleanup margin.
    age += 1
    remaining = min(started['timeout_seconds']-age, 60*3600-gpu-age-300)
    assert started['category']=='gpu' and remaining > 0
    write_once(root/'started.json', {'utc':utc(),'pid':os.getpid(),'phase':args.phase,
        'parent_pid':parent_pid,'child_pid':child_pid,'phase_age_seconds_charged':age,
        'remaining_deadline_seconds':remaining,'plan_sha256':plan_hash,
        'helper_sha256':sha256(__file__), 'resume_module_sha256':sha256(Path(__file__).parents[1]/'src/ember_v1/full_resume.py')})
    last_heartbeat = 0.
    def guard():
        nonlocal last_heartbeat
        if time.monotonic()-last_heartbeat >= 30:
            size = sum(p.stat().st_size for p in ev.rglob('*') if p.is_file())
            if size > 150*1024**3:
                raise RuntimeError('evidence budget exhausted during handoff')
            with (root/'heartbeat.jsonl').open('a') as stream:
                stream.write(json.dumps({'utc':utc(),'child_pid':child_pid,'evidence_bytes':size})+'\n')
            last_heartbeat = time.monotonic()
    code, status, details, error_record = None, 'FAIL', None, None
    held_active = False
    try:
        with HeldChild(parent_pid, child_pid, parent_command, command) as held:
            held_active = True
            write_once(root/'parent-held.json', {'utc':utc(),'parent_pid':parent_pid,
                'child_pid':child_pid,'child_not_paused':True})
            code = held.wait(remaining, guard)
            if code != 0:
                raise RuntimeError('existing collection exited ' + str(code))
            details = verify_collection(output, plan, read(ev/'config/family-panels.json'), round_index)
            status = 'PASS'
            # Its real exit status and verifier are recorded while its parent still holds the zombie.
            seconds = age + time.monotonic()-start_mono
            write_once(result_path, {**started,'completed_utc':utc(),'status':status,'exit_code':code,
                'seconds':seconds,'log_sha256':sha256(ev/'logs'/f'{args.phase}.log'),
                'gpu_seconds_total':gpu+seconds,'cpu_seconds_total':cpu,
                'completion_method':'Linux child exit status plus independent boundary completion verifier',
                'handoff':args.name,'verification':details})
            held.kill_parent()
    except BaseException as error:
        status = 'FAIL'
        error_record = {'error':str(error),'traceback':traceback.format_exc()}
        if not held_active:
            write_once(root/'result.json', {'status':'FAIL_SETUP','utc':utc(),
                'phase':args.phase,'no_model_result_manufactured':True,**error_record})
            raise
    seconds = age + time.monotonic()-start_mono
    if not result_path.exists():
        write_once(result_path, {**started,'completed_utc':utc(),'status':'FAIL','exit_code':code,
            'seconds':seconds,'log_sha256':sha256(ev/'logs'/f'{args.phase}.log'),
            'gpu_seconds_total':gpu+seconds,'cpu_seconds_total':cpu,'handoff':args.name,**(error_record or {})})
    phase_result = read(result_path)
    write_once(ev/'supervisor/run.stopped.json', {'status':'STOPPED','utc':utc(),
        'type':'CONTROLLED_SOURCE_TRANSITION' if status=='PASS' else 'HANDOFF_FAILURE',
        'reason':'Original-V1 report API repair before any report; no attempt relaunched.',
        'source_head':started['source_head'],'phase':args.phase,'handoff':args.name,
        'cpu_seconds':cpu,'gpu_seconds':phase_result['gpu_seconds_total'],**(error_record or {})})
    write_once(root/'result.json', {'status':status,'utc':utc(),'phase':args.phase,
        'phase_result_sha256':sha256(result_path),'existing_child_exit_code':code,
        'completion_verification':details,'cpu_process_seconds':time.process_time()-cpu_start,
        'source_head':started['source_head'],'no_attempt_relaunched':True,**(error_record or {})})
    print(json.dumps({'status':status,'phase':args.phase,'child_exit':code,'verification':details}),flush=True)
    if status != 'PASS':
        raise RuntimeError('boundary handoff failed; preserve all records')


if __name__ == '__main__':
    main()
