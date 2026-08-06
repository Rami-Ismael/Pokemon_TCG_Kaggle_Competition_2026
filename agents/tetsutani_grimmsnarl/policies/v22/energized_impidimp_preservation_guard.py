from __future__ import annotations
from .manual_observation import get_card_id, semantic
MUNKIDORI_ID=112; IMPIDIMP_ID=646; SNORUNT_ID=860; GRIMMSNARL_EX_ID=648
def _i(v,d=0):
 try:return int(v if v is not None else d)
 except:return d
def _e(c):return len((c or {}).get('energies') or [])
def choose(obs:dict):
 sel=obs.get('select') or {};opts=sel.get('option') or []
 if _i(sel.get('context'),-1)!=4 or _i(sel.get('minCount'))!=1 or _i(sel.get('maxCount'))!=1 or len(opts)!=3:return None,None
 cur=obs.get('current') or {};ps=cur.get('players') or [];y=_i(cur.get('yourIndex'),-1)
 if len(ps)!=2 or y not in (0,1):return None,None
 me,opp=ps[y],ps[1-y]
 if me.get('active') or len(me.get('prize') or [])<5 or len(opp.get('prize') or [])>2:return None,None
 if not any(get_card_id(c)==GRIMMSNARL_EX_ID for c in (me.get('hand') or [])):return None,None
 oa=opp.get('active') or []
 if len(oa)!=1 or _i(oa[0].get('hp'))<150 or _e(oa[0])<2:return None,None
 bench=me.get('bench') or [];seen={MUNKIDORI_ID:0,IMPIDIMP_ID:0,SNORUNT_ID:0};pick=None
 for oi,o in enumerate(opts):
  if _i((o or {}).get('area'),-1)!=5:return None,None
  bi=_i((o or {}).get('index'),-1)
  if not 0<=bi<len(bench):return None,None
  c=bench[bi];cid=get_card_id(c);hp=_i(c.get('hp'));mh=_i(c.get('maxHp'))
  if cid not in seen or hp<=0 or hp!=mh:return None,None
  seen[cid]+=1
  if cid==MUNKIDORI_ID:
   if _e(c)!=1 or pick is not None:return None,None
   pick=oi
  elif cid==IMPIDIMP_ID:
   if _e(c)<2:return None,None
  elif _e(c)!=0:return None,None
 if seen!={MUNKIDORI_ID:1,IMPIDIMP_ID:1,SNORUNT_ID:1} or pick is None:return None,None
 return [pick],[semantic(obs,o) for o in opts]
