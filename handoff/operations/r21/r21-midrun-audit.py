from pathlib import Path
import json,shutil
from ember_v1.rl_verify import verify
root=Path('/home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3')
progress=json.loads((root/'progress.json').read_text())
through=progress['completed_updates']
snapshot=root.parent/f'midrun-replay-{through:04d}'
snapshot.mkdir(exist_ok=False)
for name in ('execution.json','scenarios.json'):shutil.copy2(root/name,snapshot/name)
def allowed(row):
 label=row['label']
 return not label.startswith('train-') or int(label.split('-')[1]) < through
complete={(r['label'],r['scenario_id']) for r in (json.loads(line) for line in (root/'episodes.jsonl').open() if line.endswith('\n')) if allowed(r)}
for name in ('episodes.jsonl','transitions.jsonl','ppo_targets.jsonl'):
 with (snapshot/name).open('w') as out:
  for line in (root/name).open():
   if not line.endswith('\n'):continue
   row=json.loads(line)
   if allowed(row) and (row['label'],row['scenario_id']) in complete:out.write(line)
result=verify(snapshot)
result['through_update']=through
result['snapshot']=str(snapshot)
(root/f'midrun_replay-{through:04d}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))