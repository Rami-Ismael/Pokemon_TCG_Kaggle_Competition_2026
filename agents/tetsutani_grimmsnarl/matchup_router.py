from __future__ import annotations
"""Public-information matchup router for a fixed Grimmsnarl deck.

All experts receive every observation so their internal state stays coherent.
The router records visible opposing cards by serial number, infers only broad
archetype roles, locks a plan once evidence is sufficient, and then selects the
expert whose priorities match that role.  No player names or policy IDs are
used.
"""
from collections import Counter

# Opponent-visible signatures.
MIRROR={646:4,647:7,648:12,104:4,112:3,1259:5}
GRASS={96:12,25:7,18:2,1094:2,1213:2,1251:2}
WALL={345:12,344:6,117:10,756:7,11:2,14:2,20:2}
ARCH={190:12,169:8,666:5,57:4}
PSYCHIC={743:12,742:8,741:6,245:6,66:2,305:2}
SIG={'mirror':MIRROR,'grass':GRASS,'wall':WALL,'arch':ARCH,'psychic':PSYCHIC}
S={}
def reset():S.clear();S.update(step=-1,turn=-1,seen_serial={},seen=Counter(),profile='unknown',confidence=0.0,locked=False,opp_hand=0,discard=Counter(),threat='unknown')
reset()
def cid(x):return int(x.get('id',x.get('cardId',0)) or 0) if isinstance(x,dict) else 0
def serial(x):return int(x.get('serial',-1) if isinstance(x,dict) else -1)
def _opp(obs):
 c=obs.get('current') or {};ps=c.get('players') or [];y=int(c.get('yourIndex',0) or 0);return ps[1-y] if len(ps)>1 else {}
def _cards(p):return [x for x in (p.get('active') or [])+(p.get('bench') or [])+(p.get('discard') or []) if isinstance(x,dict)]
def update(obs):
 step=int(obs.get('step',0) or 0);turn=int((obs.get('current') or {}).get('turn',0) or 0)
 if S['step']>=0 and step<S['step']:reset()
 S['step']=step;S['turn']=turn;p=_opp(obs);S['opp_hand']=int(p.get('handCount',0) or 0);S['discard']=Counter(cid(x) for x in (p.get('discard') or []) if isinstance(x,dict))
 for x in _cards(p):
  s=serial(x);c=cid(x)
  if s>=0:
   if s not in S['seen_serial']:S['seen_serial'][s]=c;S['seen'][c]+=1
  elif c:S['seen'][c]=max(1,S['seen'][c])
 scores={k:sum(w*min(4,S['seen'][c]) for c,w in sig.items()) for k,sig in SIG.items()}
 best=max(scores,key=scores.get);vals=sorted(scores.values(),reverse=True);second=vals[1] if len(vals)>1 else 0;margin=scores[best]-second
 # Lock only on a distinctive visible card or two-card evidence.
 if not S['locked'] and (scores[best]>=7 or margin>=6):S['profile']=best;S['confidence']=min(1.0,(scores[best]+margin)/24.0);S['locked']=True
 # A small human-style threat label is retained for diagnostics and future rules.
 if S['profile']=='grass':S['threat']='fast weakness pressure'
 elif S['profile']=='wall':S['threat']='damage-prevention control'
 elif S['profile']=='mirror':S['threat']='support-engine mirror'
 elif S['profile']=='arch':S['threat']='durable metal tempo'
 elif S['profile']=='psychic':S['threat']='hand-scaled psychic pressure'
 else:S['threat']='unclassified'
 return S['profile']
def legal(obs,a):
 sel=obs.get('select') or {};n=len(sel.get('option') or []);mn=int(sel.get('minCount',0) or 0);mx=int(sel.get('maxCount',0) or 0)
 return isinstance(a,list) and mn<=len(a)<=mx and len(a)==len(set(a)) and all(isinstance(i,int) and 0<=i<n for i in a)
def choose(obs,baseline,mirror,tempo):
 profile=update(obs)
 preferred=mirror if profile=='mirror' else (tempo if profile=='grass' else baseline)
 if legal(obs,preferred):return preferred
 if legal(obs,baseline):return baseline
 for a in (mirror,tempo):
  if legal(obs,a):return a
 return None
def diagnostics():return {'profile':S.get('profile'),'confidence':S.get('confidence'),'threat':S.get('threat'),'visible_cards':dict(S.get('seen',{}))}
