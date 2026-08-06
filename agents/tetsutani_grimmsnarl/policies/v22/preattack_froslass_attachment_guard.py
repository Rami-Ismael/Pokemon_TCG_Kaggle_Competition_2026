from __future__ import annotations
from .manual_observation import get_card_id, semantic
DARK_ENERGY_ID=7
IMPIDIMP_ID=646
FROSLASS_ID=104

def _i(v,d=0):
    try:return int(v if v is not None else d)
    except Exception:return d

def _e(c):
    xs=(c or {}).get('energyCards') or (c or {}).get('energies') or []
    return len(xs)

def choose(obs:dict):
    """Bank the turn attachment on a developed Froslass before Impidimp attacks.

    In an even-prize setup turn, a zero-Energy Active Impidimp facing a very
    high-HP Active cannot create a meaningful knockout with its small attack.
    If a damaged zero-Energy Froslass is already developed on the Bench, keep
    the once-per-turn Energy on that persistent attacker rather than investing
    it into the disposable Active. Exact replay states are excluded by caller.
    """
    sel=obs.get('select') or {};opts=sel.get('option') or []
    if _i(sel.get('context'),-1)!=0 or _i(sel.get('minCount'),0)!=1 or _i(sel.get('maxCount'),0)!=1:return None,None
    cur=obs.get('current') or {}
    if _i(cur.get('turn'),0)>6 or bool(cur.get('energyAttached')):return None,None
    ps=cur.get('players') or [];yi=_i(cur.get('yourIndex'),-1)
    if len(ps)!=2 or yi not in (0,1):return None,None
    me,opp=ps[yi],ps[1-yi]
    if len(me.get('prize') or [])!=6 or len(opp.get('prize') or [])!=6:return None,None
    act=me.get('active') or []
    if len(act)!=1:return None,None
    a=act[0]
    if get_card_id(a)!=IMPIDIMP_ID or _e(a)!=0:return None,None
    if _i(a.get('hp'),0)!=_i(a.get('maxHp'),-1):return None,None
    oa=opp.get('active') or []
    if len(oa)!=1 or _i(oa[0].get('maxHp'),0)<250 or _i(oa[0].get('hp'),0)<250:return None,None
    if DARK_ENERGY_ID not in [get_card_id(x) for x in (me.get('hand') or [])]:return None,None
    sems=[semantic(obs,o) for o in opts]
    # Require that attaching to the Active is a real competing option; this is
    # an allocation choice, not a forced Bench attachment state.
    if not any(_i(s.get('type'),-1)==8 and _i(s.get('source_id'),0)==DARK_ENERGY_ID
               and _i(s.get('target_id'),0)==IMPIDIMP_ID and _i(s.get('target_area'),0)==4
               for s in sems):
        return None,sems
    bench=me.get('bench') or [];cand=[]
    for i,s in enumerate(sems):
        if (_i(s.get('type'),-1)==8 and _i(s.get('source_id'),0)==DARK_ENERGY_ID
            and _i(s.get('target_id'),0)==FROSLASS_ID and _i(s.get('target_area'),0)==5):
            bi=_i(s.get('inplay_index'),-1)
            if 0<=bi<len(bench):
                c=bench[bi];hp=_i(c.get('hp'),0);mh=_i(c.get('maxHp'),0)
                if 0<hp<mh and _e(c)==0:cand.append((hp/mh,hp,bi,i))
    if not cand:return None,sems
    cand.sort()
    return [cand[0][3]],sems
