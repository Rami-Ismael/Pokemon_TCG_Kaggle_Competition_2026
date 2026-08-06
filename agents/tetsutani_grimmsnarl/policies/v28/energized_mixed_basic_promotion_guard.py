from __future__ import annotations
from .manual_observation import get_card_id, semantic
MUNKIDORI_ID=112
IMPIDIMP_ID=646

def _i(v,d=0):
    try:return int(v if v is not None else d)
    except Exception:return d

def _e(card):return len((card or {}).get('energies') or [])

def choose(obs:dict):
    """Promote the only energized Munkidori in a mixed early Basic bench."""
    sel=obs.get('select') or {};opts=sel.get('option') or []
    if _i(sel.get('context'),-1)!=4 or _i(sel.get('minCount'),0)!=1 or _i(sel.get('maxCount'),0)!=1 or len(opts)!=3:return None,None
    cur=obs.get('current') or {};players=cur.get('players') or [];your=_i(cur.get('yourIndex'),-1)
    if len(players)!=2 or your not in (0,1):return None,None
    me,opp=players[your],players[1-your]
    if me.get('active') or len(me.get('prize') or [])!=6 or len(opp.get('prize') or [])!=4:return None,None
    oa=opp.get('active') or []
    if len(oa)!=1 or _i(oa[0].get('hp'),0)<200 or _e(oa[0])<3:return None,None
    bench=me.get('bench') or [];pick=None;zero_munk=zero_imp=0
    for oi,opt in enumerate(opts):
        if _i((opt or {}).get('area'),-1)!=5:return None,None
        bi=_i((opt or {}).get('index'),-1)
        if not 0<=bi<len(bench):return None,None
        card=bench[bi];hp=_i(card.get('hp'),0);mh=_i(card.get('maxHp'),0)
        if hp<=0 or hp!=mh:return None,None
        cid=get_card_id(card);en=_e(card)
        if cid==MUNKIDORI_ID and en==1:
            if pick is not None:return None,None
            pick=oi
        elif cid==MUNKIDORI_ID and en==0:zero_munk+=1
        elif cid==IMPIDIMP_ID and en==0:zero_imp+=1
        else:return None,None
    if pick is None or zero_munk!=1 or zero_imp!=1:return None,None
    return [pick],[semantic(obs,o) for o in opts]
