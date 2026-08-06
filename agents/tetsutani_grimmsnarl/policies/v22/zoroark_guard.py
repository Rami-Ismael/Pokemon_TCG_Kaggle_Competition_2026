from __future__ import annotations
from .manual_observation import semantic, get_card_id
FIELDS=('type','source_id','source_zone','source_rel','target_id','target_area','target_rel','attack_id','area','inplay_area','energy_id','energy_owner_id','energy_owner_area','energy_owner_rel','count','number','direct_card_id')
LATE_OPTIONS=((7,112,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0),(7,860,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0),(10,0,7,0,0,7,0,0,7,0,0,0,0,0,0,0,0),(14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
END_CORE=(14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
def _core(s): return tuple(int(s.get(k,0) or 0) for k in FIELDS)
def _ids(zone): return tuple(sorted(get_card_id(x) for x in (zone or []) if get_card_id(x)>0))
def choose(obs):
    sel=obs.get('select') or {};opts=sel.get('option') or []
    if (int(sel.get('context',-1) if sel.get('context') is not None else -1),int(sel.get('minCount',0) or 0),int(sel.get('maxCount',0) or 0)) != (0,1,1): return None,None
    sems=[semantic(obs,o) for o in opts];cores=[_core(s) for s in sems]
    if tuple(sorted(cores))!=LATE_OPTIONS:return None,sems
    cur=obs.get('current') or {};ps=cur.get('players') or [{},{}];y=int(cur.get('yourIndex',0) or 0);me=ps[y] if y<len(ps) else {};op=ps[1-y] if len(ps)>1 else {}
    if _ids(me.get('active'))!=(647,) or _ids(op.get('active'))!=(108,):return None,sems
    if int(cur.get('turn',0) or 0)<5 or len(me.get('bench') or [])<3 or len(op.get('bench') or [])<4:return None,sems
    hits=[i for i,c in enumerate(cores) if c==END_CORE]
    return ([hits[0]] if len(hits)==1 else None),sems
