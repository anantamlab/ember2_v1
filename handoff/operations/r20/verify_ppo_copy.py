import hashlib
import json
from pathlib import Path

root=Path('/mnt/c/Users/Admin/Documents/ember-evidence/r20-ppo-pilot@1/attempt-6')
manifest=json.loads((root/'EVIDENCE_MANIFEST.json').read_text())
errors=[]
for item in manifest['files']:
    path=root/item['path']
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    if h.hexdigest()!=item['sha256'] or path.stat().st_size!=item['size']:
        errors.append(item['path'])
result={'verified_files':len(manifest['files']),'bytes':manifest['bytes'],'errors':errors}
(root.parent/'windows-copy-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
assert not errors
