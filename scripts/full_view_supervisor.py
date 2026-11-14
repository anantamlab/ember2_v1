"""Persistent, bounded full-view phase supervisor with completion-triggered verification."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from ember_v1.full_dagger import read_dataset,topology
from ember_v1.full_evidence import assert_disjoint,file_table,paired_intervals,sha256,verify_table,write_once
from ember_v1.rl_online_scenarios import build_scenarios


def read(path):return json.loads(Path(path).read_text())
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


class Supervisor:
    def __init__(self,repo,evidence):
        self.repo=Path(repo).resolve();self.ev=Path(evidence).resolve();self.py=sys.executable
        assert self.ev.is_relative_to(Path('/home/darshith/ember-ws/ember2-evidence'))
        assert not self.ev.is_relative_to(self.repo)
        self.base_env=dict(os.environ,PYTHONPATH=str(self.repo/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            HF_HOME='/home/darshith/ember-ws/hf-home',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        # Credentials are never copied to command/start/result records or logs.
        self.head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.repo,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=self.repo,text=True)
        receipts=[read(p) for p in (self.repo/'.git/ember-local-verify/receipts').glob(self.head+'*.json')]
        assert any(r['status']=='PASS' and r['tree_clean'] and sha256(r['log'])==r['log_sha256'].removeprefix('sha256:') for r in receipts)
        prior=list((self.ev/'attempts').glob('*.result.json'))
        initial=read(self.ev/'config/budget-initial.json') if (self.ev/'config/budget-initial.json').exists() else {}
        overhead=initial.get('cpu_overhead_seconds',0.)
        assert overhead>=0
        self.cpu=overhead+sum(read(p)['seconds'] for p in prior if read(p)['category']=='cpu')
        self.gpu=sum(read(p)['seconds'] for p in prior if read(p)['category']=='gpu')
        self.started=time.monotonic()

    def phase(self,name,args,category,verify,cap=None):
        assert category in ('cpu','gpu')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.repo,text=True).strip()==self.head
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=self.repo,text=True)
        if sum(p.stat().st_size for p in self.ev.rglob('*') if p.is_file())>150*1024**3:
            raise RuntimeError('evidence budget exhausted')
        # Leave five minutes for termination and completion verification inside the hard limit.
        timeout=(60*3600-self.gpu-300) if category=='gpu' else (16*3600-self.cpu-300)
        if cap is not None:timeout=min(timeout,cap)
        if timeout<=0:raise RuntimeError(category+' budget exhausted')
        command=[self.py]+list(map(str,args));env=self.base_env.copy()
        if category=='cpu':env['CUDA_VISIBLE_DEVICES']=''
        cache=self.ev/'cache'/name;cache.mkdir(parents=True,exist_ok=False)
        env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),UNSLOTH_COMPILE_LOCATION=str(cache/'unsloth'))
        record={'phase':name,'command':command,'category':category,'source_head':self.head,'utc':utc(),
                'timeout_seconds':timeout,'plan_sha256':sha256(self.ev/'config/plan.json') if (self.ev/'config/plan.json').exists() else None}
        write_once(self.ev/'attempts'/f'{name}.started.json',record)
        lock=None;start=time.monotonic();code=None;status='FAIL'
        try:
            if category=='gpu':
                lock=open(self.ev.parent.parent/'.gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if active:raise RuntimeError('another GPU compute process is active')
            with (self.ev/'logs'/f'{name}.log').open('xb') as out:
                child=subprocess.Popen(command,cwd=self.repo,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                write_once(self.ev/'attempts'/f'{name}.process.json',{'pid':child.pid,'supervisor_pid':os.getpid(),'utc':utc()})
                try:code=child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError(category+' budget/phase cap reached')
            if code!=0:raise RuntimeError(f'{name} exited {code}; preserve attempt')
            details=verify();status='PASS'
            return details
        finally:
            duration=time.monotonic()-start
            if category=='gpu':self.gpu+=duration
            else:self.cpu+=duration
            if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
            write_once(self.ev/'attempts'/f'{name}.result.json',{**record,'completed_utc':utc(),'status':status,
                'exit_code':code,'seconds':duration,'log_sha256':sha256(self.ev/'logs'/f'{name}.log') if (self.ev/'logs'/f'{name}.log').exists() else None,
                'gpu_seconds_total':self.gpu,'cpu_seconds_total':self.cpu})

    def op(self,name,phase,out,category='cpu',extra=(),verify=None,cap=None,planned=True):
        args=['experiments/full_view_phase.py',phase,'--out',out]
        if planned:args+=['--plan',self.ev/'config/plan.json']
        args+=list(extra)
        return self.phase(name,args,category,verify or (lambda:read(out)),cap)

    def prepare(self):
        original='/home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1'
        run_id='ember2-full-view-'+self.ev.name.lower()
        self.op('token-preflight-01','preflight',self.ev/'token-preflight.json',extra=['--namespace',run_id+'-tokens','--original',original],planned=False)
        self.op('teacher-reference-01','teacher-reference',self.ev/'teacher-reference.json',extra=['--namespace',run_id+'-teacher-ref'],planned=False,cap=7200)
        write_once(self.ev/'supervisor/prepare-complete.json',{'status':'PASS','source_head':self.head,'gpu_seconds':self.gpu,'cpu_seconds':self.cpu,'utc':utc()})

    def run(self):
        from runpy import run_path
        plan=read(self.ev/'config/plan.json');assert sha256(self.ev/'config/plan.json')==(self.ev/'config/plan.sha256').read_text().split()[0]
        assert plan['status']=='frozen' and plan['source_head']==self.head
        for name,digest in plan['config_hashes'].items():assert sha256(self.ev/'config'/name)==digest
        panels=read(self.ev/'config/family-panels.json');assert_disjoint(panels)
        self.op('select-reference-01','reference',self.ev/'select-reference.json')
        def gpu_check(name,checkpoint=None):
            out=self.ev/(name+'.json');args=['experiments/full_gpu_checks.py','--namespace',plan['namespace']+'-'+name,'--out',out]
            if checkpoint:args+=['--checkpoint',checkpoint]
            return self.phase(name,args,'gpu',lambda: self.require_pass(out))
        gpu_check('gpu-check-01')
        data=[];best=None;best_utility=-float('inf');fits=[]
        teacher=read(self.ev/'select-reference.json')['teacher']['summary']
        for i in range(4):
            d=self.ev/f'd{i}'
            extra=['--round',str(i)]
            if best:extra+=['--student',best]
            rows,meta=self.op(f'dagger-r{i}-collect-01','collect',d,'cpu' if i==0 else 'gpu',extra,
                              verify=lambda d=d:read_dataset(d))
            data.append(d)
            # Collection agreement counts only actual student decisions; beta=1 never qualifies.
            if i and meta['stats']['student_agreement'] is not None and meta['stats']['student_agreement']>=.99:
                break
            ck=self.ev/f's{i}'
            extra=['--data',*data]
            if best:extra+=['--student',best]
            self.op(f'dagger-r{i}-fit-01','fit',ck,'gpu',extra,verify=lambda ck=ck:self.verify_fit(ck))
            selected={}
            for arm in ('candidate_raw','candidate_masked'):
                out=self.ev/f's{i}-select-{arm}.json'
                selected[arm]=self.op(f's{i}-select-{arm}-01','eval',out,'gpu',extra=['--student',ck,'--arm',arm,'--data',*data])
            utility=selected['candidate_raw']['summary']['utility'];fits.append({'checkpoint':str(ck),'selection':selected})
            if utility>best_utility:best,best_utility=ck,utility
        assert best is not None
        last=next(x for x in fits if x['checkpoint']==str(best))['selection']['candidate_raw']['summary']
        gate=last['movements']['blocked_wall']==0 and last['movements']['blocked_boundary']==0 and last['utility']>=teacher['utility']-.05
        write_once(self.ev/'pre-ppo-gate.json',{'status':'PASS' if gate else 'FAIL','checkpoint':str(best),
            'raw':last,'teacher':teacher,'utility_gap':teacher['utility']-last['utility'],'fits':fits,
            'reason':'zero wall/boundary blocks and raw utility within .05 required; failed gate skips PPO'})
        candidate=best;seeds=[]
        if gate:
            gpu_check('dagger-gpu-check-01',best)
            self.op('ppo-small-01','ppo',self.ev/'ppo-small','cpu',extra=['--actor','small','--config',self.ev/'config/ppo-small.json'],
                    verify=lambda:read(self.ev/'ppo-small/summary.json'))
            for seed in plan['ppo']['llm_seeds']:
                out=self.ev/f'ppo-llm-seed-{seed}'
                summary=self.op(f'ppo-llm-seed-{seed}-01','ppo',out,'gpu',extra=['--actor','llm','--student',best,'--config',self.ev/f'config/ppo-llm-{seed}.json'],verify=lambda out=out:self.verify_ppo(out))
                seeds.append(summary)
            eligible=[x for x in seeds if x['guard_passed'] and x['selected_utility']>last['utility']
                      and x['selected']['casualty_rate']<=last['casualty_rate']+.02]
            if eligible:
                winner=max(eligible,key=lambda x:(x['selected_utility'],-x['config']['seed']))
                candidate=self.ev/f"ppo-llm-seed-{winner['config']['seed']}"/f"update-{winner['selected_update']:04d}"/'actor'
        write_once(self.ev/'candidate-freeze.json',{'utc':utc(),'checkpoint':str(candidate),'files':file_table(candidate),
            'dagger_checkpoint':str(best),'dagger_files':file_table(best),'pre_ppo_gate':gate,'ppo_seeds':seeds,
            'report_panel':plan['panels']['report'],'report_arms':plan['report_arms'],'source_head':self.head})
        reports={};(self.ev/'report').mkdir()
        for arm in plan['report_arms']:
            ck=best if arm=='dagger_raw' else candidate;out=self.ev/'report'/f'{arm}.json'
            category='cpu' if arm in ('teacher','random_legal') else 'gpu'
            reports[arm]=self.op('report-'+arm+'-01','eval',out,category,
                extra=['--split','report','--arm',arm,'--student',ck,'--data',*data])
            self.validate_report(reports[arm],plan)
        intervals={}
        for left,right in [('candidate_raw','original_v1_raw'),('candidate_masked','original_v1_masked'),('candidate_raw','dagger_raw')]:
            intervals[left+'-minus-'+right]=paired_intervals(reports[left]['episodes'],reports[right]['episodes'],resamples=5000,seed=0)
        write_once(self.ev/'paired-intervals.json',intervals)
        freeze=read(self.ev/'candidate-freeze.json');verify_table(candidate,freeze['files']);verify_table(best,freeze['dagger_files'])
        write_once(self.ev/'supervisor/run-complete.json',{'status':'PASS','utc':utc(),'source_head':self.head,'gpu_seconds':self.gpu,
            'cpu_seconds':self.cpu,'cloud_spend_usd':0,'evidence_bytes':sum(p.stat().st_size for p in self.ev.rglob('*') if p.is_file()),
            'report_arms':{k:v['summary'] for k,v in reports.items()},'candidate_freeze_sha256':sha256(self.ev/'candidate-freeze.json'),
            'plan_sha256':sha256(self.ev/'config/plan.json'),'intervals_sha256':sha256(self.ev/'paired-intervals.json'),
            'report_sha256':{k:sha256(self.ev/'report'/f'{k}.json') for k in reports}})

    @staticmethod
    def require_pass(path):
        result=read(path);assert result['status']=='PASS';return result

    @staticmethod
    def verify_fit(path):
        import math
        result=read(Path(path)/'full-fit.json')
        assert math.isfinite(result['report']['first_loss']) and math.isfinite(result['report']['last_loss'])
        assert result['report']['last_loss']<result['report']['first_loss']
        return result

    @staticmethod
    def verify_ppo(path):
        import math
        rows=[json.loads(x) for x in (Path(path)/'log.jsonl').read_text().splitlines()]
        summary=read(Path(path)/'summary.json');assert len(rows)==summary['config']['updates']
        for row in rows:
            for key in ('actor_loss','critic_loss','entropy','imitation','clip_fraction','replay_error'):
                assert math.isfinite(row[key])
            assert row['replay_error']<=summary['config']['replay_tolerance']
        for cp in Path(path).glob('update-*'):
            identity=read(cp/'identity.json')
            for name,digest in identity['files'].items():assert sha256(cp/name)==digest
        return summary

    @staticmethod
    def validate_report(result,plan):
        s=result['summary'];assert s['episodes']==plan['panels']['report']['count']
        assert s['people']==s['evacuated']+s['casualties']+s['unresolved']
        assert len(result['episodes'])==len({r['scenario_id'] for r in result['episodes']})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--evidence',required=True)
    ap.add_argument('--mode',choices=['prepare','run'],required=True);a=ap.parse_args()
    supervisor=Supervisor(a.repo,a.evidence)
    write_once(supervisor.ev/'supervisor'/f'{a.mode}.started.json',{'pid':os.getpid(),'utc':utc(),'head':supervisor.head})
    try:
        supervisor.prepare() if a.mode=='prepare' else supervisor.run()
    except BaseException as error:
        write_once(supervisor.ev/'supervisor'/f'{a.mode}.stopped.json',{'status':'STOPPED','utc':utc(),'error':str(error),
            'traceback':traceback.format_exc(),'cpu_seconds':supervisor.cpu,'gpu_seconds':supervisor.gpu,'no_attempt_restarted':True})
        raise

if __name__=='__main__':main()
