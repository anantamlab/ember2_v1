from pathlib import Path
import json,math,statistics,time
root=Path('/home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3')
us=[json.loads(s) for s in (root/'updates.jsonl').open()]
ms=[m for u in us for m in u['minibatches']]
applied=[m for m in ms if 'gradient_norms' in m]
prior=sum(json.loads(p.read_text())['elapsed_seconds'] for p in root.parent.glob('attempt-*/failure.json'))
execution=json.loads((root/'execution.json').read_text())
elapsed=time.time()-execution['started_unix']
result={'status':'HEALTHY','completed_updates':len(us),'on_policy_episodes':sum(u['rollout']['episodes'] for u in us),'environment_steps':sum(u['rollout']['steps'] for u in us),'optimizer_steps':sum(u['optimizer_steps'] for u in us),'max_probability_replay_error':max(u['max_probability_replay_error'] for u in us),'all_applied_gradients_finite_nonzero':all(all(math.isfinite(n) and n>0 for n in m['gradient_norms'].values()) for m in applied),'gradient_checks':len(applied),'mean_joint_kl':statistics.mean(m['joint_kl'] for m in ms),'recent_person_entropy':statistics.mean(m['person_entropy'] for u in us[-8:] for m in u['minibatches']),'recent_stay_rate':sum(u['rollout']['stay'] for u in us[-8:])/sum(u['rollout']['person_steps'] for u in us[-8:]),'prior_failed_attempt_seconds':prior,'current_elapsed_seconds':elapsed,'remaining_process_budget_seconds':execution['config']['wall_cap_seconds']-prior-elapsed,'phase3_held':True}
assert result['all_applied_gradients_finite_nonzero'] and result['max_probability_replay_error']<=1e-5 and result['remaining_process_budget_seconds']>0
(root/'latest_health_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))