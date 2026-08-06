from __future__ import annotations
from .manual_observation import semantic,get_card_id

def _ids(z): return [get_card_id(x) for x in (z or [])]

def choose(obs):
    sel=obs.get('select') or {};opts=sel.get('option') or []
    if int(sel.get('context',-1) if sel.get('context') is not None else -1)!=7:return None,None
    if int(sel.get('minCount',0) or 0)!=0 or int(sel.get('maxCount',0) or 0)!=1:return None,None
    if get_card_id(sel.get('effect'))!=1219:return None,None
    cur=obs.get('current') or {};ps=cur.get('players') or [{},{}];y=int(cur.get('yourIndex',0) or 0)
    if len(ps)<2:return None,None
    me=ps[y];op=ps[1-y];ma=me.get('active') or [];oa=op.get('active') or []
    if _ids(ma)!=[112] or _ids(oa)!=[96]:return None,None
    if not oa or int((oa[0] or {}).get('hp',999) or 999)>40:return None,None
    if len(me.get('prize') or [])!=1 or len(op.get('prize') or [])!=1:return None,None
    bench=_ids(me.get('bench'))
    if bench.count(104)<2 or bench.count(112)<1:return None,None
    if not bool(cur.get('supporterPlayed')):return None,None
    sems=[semantic(obs,o) for o in opts]
    pad=[i for i,s in enumerate(sems) if int(s.get('source_id',0))==1152 and int(s.get('source_zone',0))==1]
    stretcher=[i for i,s in enumerate(sems) if int(s.get('source_id',0))==1097 and int(s.get('source_zone',0))==1]
    if len(pad)!=1 or not stretcher:return None,sems
    return [pad[0]],sems
