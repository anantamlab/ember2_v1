"""Optimizer-free full-view Clef checks; require PASS before any fit."""
import argparse
import json
import math
import tempfile
from pathlib import Path

from ember_v1.full_encoding import QUESTION_ID,view_row
from ember_v1.full_evidence import write_once
from ember_v1.full_view import FullObserver
from ember_v1.rl_online_scenarios import build_scenarios
from ember_v1.sim import Simulator
from ember_v1.student_train import ACTION_ORDER,ClefLoraAdapter,config_from_protocol,load_protocol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--namespace',required=True)
    ap.add_argument('--checkpoint',type=Path);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    import torch
    torch.set_num_threads(4)
    cfg=config_from_protocol(load_protocol())
    views=[]
    for s,_ in build_scenarios(a.namespace,'train',8,48):
        views.extend(FullObserver().observe(Simulator(s).observe()).values())
    adapter=ClefLoraAdapter(cfg);tokens=[];normalizer=0.
    for view in views:
        row=view_row(view,'stay');item=adapter.item(row);tokens.append(len(item['input_ids']))
        scores=adapter.scores(row)[QUESTION_ID]
        assert list(scores)==list(ACTION_ORDER) and all(math.isfinite(x) for x in scores.values())
        normalizer=max(normalizer,abs(math.log(sum(math.exp(x) for x in scores.values()))))
    assert normalizer<=1e-3
    loss=adapter.loss_and_backward(view_row(views[0],'stay'),1.);norms=adapter.grad_norms()
    assert math.isfinite(loss) and all(math.isfinite(x) and x>0 for x in norms.values())
    adapter.opt.zero_grad(set_to_none=True)
    # Exact actor saver and reload are checked without ever updating the fresh weights.
    from unsloth.models.decision import _save_clef_adapter
    scratch=Path(tempfile.mkdtemp(prefix='full-gpu-exact-',dir=a.out.parent))
    cp=scratch/'actor';cp.mkdir();_save_clef_adapter(adapter.model,str(cp),adapter.tok,exact=True)
    reference=[adapter.scores(view_row(v,'stay'))[QUESTION_ID] for v in views[:4]]
    del adapter;torch.cuda.empty_cache()
    from ember_v1.full_ppo import FullLLMActor
    from ember_v1.human_ppo import mask_tensor,masked_log_probs,ppo_loss
    actor=FullLLMActor(cfg,a.checkpoint or cp);vs=views[:4];mask=mask_tensor(vs,actor.device)
    with torch.enable_grad():
        first=masked_log_probs(actor.logits(vs),mask)
        second=masked_log_probs(actor.logits(vs),mask)
    replay=float((first-second).detach().masked_fill(~mask,0).abs().max())
    assert replay<=1e-5
    parity=0.
    if a.checkpoint is None:
        for v,ref in zip(vs,reference):
            z=actor.logits([v]).float().log_softmax(-1)[0]
            parity=max(parity,max(abs(float(z[i].detach())-ref[k]) for i,k in enumerate(ACTION_ORDER)))
        assert parity<=5e-3
    choice=first.argmax(-1);loss2,_=ppo_loss(first.gather(-1,choice[:,None]).squeeze(-1),second.detach().gather(-1,choice[:,None]).squeeze(-1),torch.ones(len(vs),device=actor.device),.2)
    loss2.backward()
    grads={g['name']:math.sqrt(sum(float(p.grad.float().square().sum()) for p in g['params'] if p.grad is not None))
           for g in actor.parameter_groups({'lora_lr':0.,'head_lr':0.})}
    assert all(math.isfinite(x) and x>0 for x in grads.values())
    result={'status':'PASS','rows':len(views),'max_tokens':max(tokens),'option_order':list(ACTION_ORDER),
            'max_log_normalizer_error':normalizer,'first_loss':loss,'grad_norms':norms,
            'autograd_replay_error':replay,'actor_vs_adapter':parity,'ppo_grad_norms':grads,
            'optimizer_steps':0,'checkpoint':str(a.checkpoint or ''),'exact_save_scratch':str(scratch)}
    write_once(a.out,result);print(json.dumps(result))

if __name__=='__main__':main()
