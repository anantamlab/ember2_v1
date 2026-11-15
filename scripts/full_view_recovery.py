"""Single named diagnostic repair; no completed collection, fit or report is relaunched."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback

from full_view_supervisor import Supervisor,read,utc
from ember_v1.full_evidence import sha256,write_once
from ember_v1.full_recovery import load_recovery,validate_reused,compare_losses


class RecoverySupervisor(Supervisor):
    def __init__(self,repo,evidence,recovery_path):
        super().__init__(repo,evidence)
        self.recovery_path=Path(recovery_path).resolve();self.recovery=load_recovery(self.recovery_path,self.head)
        self.recovery_hash=sha256(self.recovery_path)
        ledger=read(self.ev/self.recovery['budget_ledger']['path'])['budget']
        self.cpu=ledger['cpu_seconds_total_conservatively_charged']+self.recovery['additional_cpu_seconds']
        self.gpu=ledger['gpu_seconds']
        assert self.cpu<16*3600 and self.gpu<60*3600
        # Original run's scientific schedule stays unchanged. This is a separate recovery, not the
        # former control-only amendment (its loader and PASS-only reuse rules remain unchanged).
        self.amendment={'new_source_head':self.head,'controller_stop':self.recovery['stop'],
            'additional_cpu_seconds':self.recovery['additional_cpu_seconds']}
        self.amendment_sha256=self.recovery_hash

    def repaired_fit(self):
        audit=read(self.ev/'saved-s3-train-audit-01.json')
        assert audit['status']=='PASS' and audit['historical_fit_status']=='FAIL_PRESERVED'
        assert audit['recovery_sha256']==self.recovery_hash
        assert audit['panel_sha256']==self.recovery['audit']['panel']['sha256']
        assert audit['rows']==4096 and audit['training_steps']==audit['collection_episodes']==audit['select_episodes']==audit['report_episodes']==0
        compare_losses(audit['parent_mean_ce'],audit['child_mean_ce'])
        attempt=read(self.ev/'attempts/saved-s3-train-audit-01.result.json')
        assert attempt['status']=='PASS' and attempt['exit_code']==0 and attempt['source_head']==self.head
        assert attempt['log_sha256']==sha256(self.ev/'logs/saved-s3-train-audit-01.log')
        return read(self.ev/'s3/full-fit.json')

    def phase(self,name,args,category,verify,cap=None):
        if sha256(self.recovery_path)!=self.recovery_hash:raise ValueError('live recovery hash mismatch')
        declaration=self.recovery['completed_phases'].get(name)
        if declaration is not None:
            record=read(self.ev/'attempts'/f'{name}.result.json')
            validate_reused(self.ev,name,[self.py]+list(map(str,args)),category,record['plan_sha256'],declaration,
                            failed_fit=self.recovery['failed_phase'])
            repaired=name==self.recovery['failed_phase']
            value=self.repaired_fit() if repaired else verify()
            write_once(self.ev/'supervisor'/self.recovery['run_name']/'reuse'/f'{name}.json',{
                'status':'DIAGNOSTIC_REPAIR_NO_FIT_RESTART' if repaired else 'PASS_REUSED_WITHOUT_LAUNCH',
                'historical_status':record['status'],'result_sha256':declaration['result_sha256'],
                'new_charge_seconds':0,'audit_sha256':sha256(self.ev/'saved-s3-train-audit-01.json') if repaired else None})
            return value
        if any((self.ev/'attempts'/f'{name}.{suffix}.json').exists() for suffix in ('started','result')):
            raise FileExistsError('existing attempt cannot restart: '+name)
        amendment=self.amendment;self.amendment=None
        try:return super().phase(name,args,category,verify,cap)
        finally:self.amendment=amendment

    def recover(self):
        out=self.ev/'saved-s3-train-audit-01.json'
        self.phase('saved-s3-train-audit-01',['experiments/full_saved_weight_audit.py',
            '--recovery',self.recovery_path,'--out',out],'gpu',lambda:self.require_pass(out),cap=7200)
        self.repaired_fit()
        self.run()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--evidence',required=True)
    ap.add_argument('--recovery',type=Path,required=True);a=ap.parse_args()
    obj=RecoverySupervisor(a.repo,a.evidence,a.recovery);name=obj.recovery['run_name']
    write_once(obj.ev/'supervisor'/f'{name}.started.json',{'pid':os.getpid(),'head':obj.head,'utc':utc(),
        'recovery_sha256':obj.recovery_hash,'cpu_seconds_initial':obj.cpu,'gpu_seconds_initial':obj.gpu})
    try:obj.recover()
    except BaseException as error:
        write_once(obj.ev/'supervisor'/f'{name}.stopped.json',{'status':'STOPPED','utc':utc(),
            'error':str(error),'traceback':traceback.format_exc(),'cpu_seconds':obj.cpu,'gpu_seconds':obj.gpu,
            'no_attempt_restarted':True,'recovery_sha256':obj.recovery_hash})
        raise


if __name__=='__main__':main()
