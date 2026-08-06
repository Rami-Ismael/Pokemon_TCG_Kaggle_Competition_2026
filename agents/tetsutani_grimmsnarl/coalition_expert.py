from __future__ import annotations
from collections import Counter, defaultdict
from typing import Callable
from pathlib import Path
import json, math
from policies.v22 import main as _v22
from policies.rule_v24 import main as _rule_v24
from policies.v26_reexport import main as _v26_reexport
from policies.handwritten_v26 import main as _handwritten_v26
from policies.v28 import main as _v28
from policies.v32 import main as _v32

DECK=[7,7,7,7,7,7,7,7,7,7,104,104,112,112,112,112,646,646,646,646,647,647,647,648,648,648,860,860,1079,1079,1079,1080,1086,1086,1086,1086,1097,1097,1097,1122,1137,1152,1152,1152,1152,1182,1182,1219,1219,1219,1219,1227,1227,1227,1227,1231,1259,1259,1259,1259]
BASE=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
_W=json.loads((BASE/'coalition_weights.json').read_text())
_VOTERS: tuple[tuple[str,Callable[[dict],list[int]]],...]=(
 ('v22',_v22.agent),('rule',_rule_v24.agent),('plan',_v26_reexport.agent),
 ('manual',_handwritten_v26.agent),('v28',_v28.agent),('v32',_v32.agent),
)
_STATS=Counter()

def _legal(obs,a):
 s=obs.get('select') or {}; n=len(s.get('option') or []); mn=int(s.get('minCount',0) or 0); mx=int(s.get('maxCount',0) or 0)
 return isinstance(a,list) and mn<=len(a)<=mx and len(a)==len(set(a)) and all(isinstance(i,int) and 0<=i<n for i in a)

def _fallback(obs):
 s=obs.get('select') or {}; n=len(s.get('option') or []); mn=int(s.get('minCount',0) or 0); mx=min(n,int(s.get('maxCount',0) or 0)); return list(range(max(mn,mx)))

def _rate(pair,prior,alpha):
 c,n=pair; return (float(c)+alpha*prior)/(float(n)+alpha)

def reset():
 _STATS.clear()
 for _,fn in _VOTERS:
  try: fn({})
  except Exception: pass

def agent(obs):
 if not obs or obs.get('select') is None: reset(); return list(DECK)
 votes=[]
 for name,fn in _VOTERS:
  try: a=list(fn(obs))
  except Exception: continue
  if _legal(obs,a): votes.append((name,a))
 if not votes: return _fallback(obs)
 grouped=defaultdict(list)
 # Map shortened public-neutral names back to weight-table voter keys.
 keyname={'rule':'rule_v24','plan':'v26_reexport','manual':'handwritten_v26'}
 for name,a in votes: grouped[tuple(a)].append(keyname.get(name,name))
 max_support=max(len(v) for v in grouped.values()); winners=[a for a,v in grouped.items() if len(v)==max_support]
 if max_support>=4 and len(winners)==1: return list(winners[0])
 context=str(int((obs.get('select') or {}).get('context',-1))); family=_W['family_by_context'].get(context,'other'); sm=_W['smoothing']; scores={}
 for action,supporters in grouped.items():
  global_rates=[_rate(_W['global_agent'][a],0.5,sm['agent_global']) for a in supporters]
  fallback=sum(global_rates)/len(global_rates); key='|'.join(sorted(supporters))
  base=_rate(_W['global_coalition'].get(key,[0,0]),fallback,sm['coalition_global'])
  p=_rate(_W['family_coalition'].get(f'{family}::{key}',[0,0]),base,sm['coalition_family'])
  p=min(.99999,max(.00001,p)); scores[action]=math.log(p/(1-p))+sm['support_bonus']*len(supporters)
 best=max(scores.values()); tied={a for a,v in scores.items() if abs(v-best)<1e-12}
 if len(tied)==1: return list(next(iter(tied)))
 chosen=None; bp=-1.0
 for name,a in votes:
  ta=tuple(a)
  if ta not in tied: continue
  wn=keyname.get(name,name); gp=_rate(_W['global_agent'][wn],0.5,sm['agent_global']); fp=_rate(_W['family_agent'].get(family,{}).get(wn,[0,0]),gp,sm['tie_agent_family'])
  if fp>bp: bp=fp; chosen=ta
 return list(chosen if chosen is not None else winners[0])
