from __future__ import annotations
from .manual_observation import get_card_id, semantic
BOSS_ID=1182
IMPIDIMP_ID=646
ALAKAZAM_ID=743
KADABRA_ID=742
DUNSPARCE_ID=305

def _i(v,d=0):
 try:return int(v if v is not None else d)
 except:return d

def _e(c):return len((c or {}).get('energies') or [])

def choose(obs:dict):
 sel=obs.get('select') or {};opts=sel.get('option') or []
 eff=sel.get('effect') or {}
 if _i(sel.get('context'),-1)!=3 or get_card_id(eff)!=BOSS_ID:return None,None
 if _i(sel.get('minCount'))!=1 or _i(sel.get('maxCount'))!=1 or len(opts)<2:return None,None
 cur=obs.get('current') or {};ps=cur.get('players') or [];y=_i(cur.get('yourIndex'),-1)
 if len(ps)!=2 or y not in (0,1) or _i(cur.get('turn'))>6:return None,None
 me,opp=ps[y],ps[1-y]
 ma=me.get('active') or [];oa=opp.get('active') or []
 if len(ma)!=1 or get_card_id(ma[0])!=IMPIDIMP_ID or _e(ma[0])!=1:return None,None
 if _i(ma[0].get('hp'))!=_i(ma[0].get('maxHp')):return None,None
 if len(me.get('prize') or [])<5 or len(opp.get('prize') or [])!=4:return None,None
 if len(oa)!=1 or get_card_id(oa[0])!=ALAKAZAM_ID or _e(oa[0])<1:return None,None
 bench=opp.get('bench') or [];kad=[];has_duns=False
 for oi,o in enumerate(opts):
  if _i((o or {}).get('area'),-1)!=5 or _i((o or {}).get('playerIndex'),-1)==y:return None,None
  bi=_i((o or {}).get('index'),-1)
  if not 0<=bi<len(bench):return None,None
  c=bench[bi];cid=get_card_id(c);hp=_i(c.get('hp'));mh=_i(c.get('maxHp'))
  if cid==KADABRA_ID and hp==mh and hp>0 and _e(c)==0:kad.append(oi)
  if cid==DUNSPARCE_ID and hp==mh and hp>0 and _e(c)==0:has_duns=True
 if len(kad)!=1 or not has_duns:return None,None
 return [kad[0]],[semantic(obs,o) for o in opts]
