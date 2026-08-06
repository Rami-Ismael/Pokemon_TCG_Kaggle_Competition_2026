from __future__ import annotations
from collections import Counter

# Public-card signatures only. Names, player IDs and policy IDs are never used.
SIG = {
    'mirror': {646:5, 647:8, 648:15, 104:5, 112:4, 1259:6},
    'grass_fast': {96:3, 18:3, 1127:10, 1118:9, 1119:9, 1213:8, 1251:8, 1201:7},
    'grass_control': {96:3, 25:13, 1123:6, 1116:9, 1221:9, 1081:7, 1097:5, 1152:4},
    'wall': {345:16, 344:9, 117:14, 756:8, 11:4, 14:3, 20:4},
    'metal': {190:16, 169:10, 666:6, 57:4},
    'psychic': {743:15, 742:10, 741:7, 245:7, 66:3, 305:3},
}
FAST_DISTINCTIVE={1118,1119,1213,1251,1201}
CONTROL_DISTINCTIVE={25,1123,1116,1221,1081,1097,1152}



STATE = {}

def reset():
    STATE.clear()
    STATE.update(
        step=-1, turn=-1, seen_serial={}, seen=Counter(), discard=Counter(),
        profile='unknown', confidence=0.0, profile_scores={}, plan='engine',
        opp_hand=0, own_prizes=6, opp_prizes=6, opp_took_prizes=False,
        own_took_prizes=False, active_ko_risk=False, predicted_damage=0,
        threat='unclassified', history=[], last_snapshot=None,
    )
reset()

def _cid(x):
    return int(x.get('id', x.get('cardId', 0)) or 0) if isinstance(x, dict) else 0

def _serial(x):
    return int(x.get('serial', -1) if isinstance(x, dict) else -1)

def _energy_n(p):
    return len((p or {}).get('energyCards') or []) if isinstance(p, dict) else 0

def _cards(player):
    if not isinstance(player, dict): return []
    out=[]
    for key in ('active','bench','discard'):
        out.extend(x for x in (player.get(key) or []) if isinstance(x,dict))
    for p in (player.get('active') or [])+(player.get('bench') or []):
        if not isinstance(p,dict): continue
        out.extend(x for x in (p.get('energyCards') or []) if isinstance(x,dict))
        out.extend(x for x in (p.get('tools') or []) if isinstance(x,dict))
    return out

def _players(obs):
    cur=obs.get('current') or {}; ps=cur.get('players') or []
    y=int(cur.get('yourIndex',0) or 0)
    me=ps[y] if len(ps)>y else {}
    opp=ps[1-y] if len(ps)>1-y else {}
    return cur,me,opp

def _active(player):
    a=(player or {}).get('active') or []
    return a[0] if a and isinstance(a[0],dict) else None

def _weak_to_grass(card):
    return _cid(card) in {646,647,648}

def _predict_damage(me,opp,profile):
    oa=_active(opp); ma=_active(me); oid=_cid(oa); oe=_energy_n(oa); me_e=_energy_n(ma)
    dmg=0
    if oid==96 and oe>=3:
        dmg=30*(1+oe+me_e)
        if _weak_to_grass(ma): dmg*=2
    elif oid==648 and oe>=2: dmg=180
    elif oid==345 and oe>=3: dmg=120
    elif oid==117 and oe>=3: dmg=140
    elif profile.startswith('grass') and oe>=3: dmg=180 if _weak_to_grass(ma) else 100
    elif profile=='mirror' and oe>=2: dmg=180
    return dmg

def _count_inplay(player,cids):
    return sum(1 for x in (player.get('active') or [])+(player.get('bench') or []) if isinstance(x,dict) and _cid(x) in cids)

def _profile_scores(seen):
    scores={k:0.0 for k in SIG}
    for name,sig in SIG.items():
        scores[name]=sum(w*min(4,seen[c]) for c,w in sig.items())
    return scores

def _choose_profile(scores, previous):
    ordered=sorted(scores.items(),key=lambda kv:(-kv[1],kv[0]))
    best,bv=ordered[0]; sv=ordered[1][1] if len(ordered)>1 else 0.0
    # Hysteresis: keep a previously credible read unless the new evidence is clearly stronger.
    if previous in scores and scores[previous] >= bv-3.0 and scores[previous]>=6.0:
        best=previous; bv=scores[best]
        sv=max(v for k,v in scores.items() if k!=best)
    if bv<5.0: return 'unknown', min(0.35,bv/14.0)
    return best, min(1.0,(bv+max(0.0,bv-sv))/28.0)

def update(obs):
    if not obs or obs.get('select') is None:
        reset(); return STATE
    cur,me,opp=_players(obs)
    step=int(obs.get('step',0) or 0); turn=int(cur.get('turn',0) or 0)
    if STATE['step']>=0 and (step<STATE['step'] or turn<STATE['turn']-1): reset()
    if step==STATE.get('step') and turn==STATE.get('turn') and STATE.get('last_snapshot') is not None:
        return STATE

    own_pr=len(me.get('prize') or []); opp_pr=len(opp.get('prize') or [])
    STATE['opp_took_prizes']=opp_pr < STATE.get('opp_prizes',opp_pr)
    STATE['own_took_prizes']=own_pr < STATE.get('own_prizes',own_pr)
    STATE['own_prizes']=own_pr; STATE['opp_prizes']=opp_pr
    STATE['step']=step; STATE['turn']=turn; STATE['opp_hand']=int(opp.get('handCount',0) or 0)
    STATE['discard']=Counter(_cid(x) for x in (opp.get('discard') or []) if isinstance(x,dict))

    for x in _cards(opp):
        c=_cid(x); s=_serial(x)
        if not c: continue
        if s>=0:
            if s not in STATE['seen_serial']:
                STATE['seen_serial'][s]=c; STATE['seen'][c]+=1
        else:
            STATE['seen'][c]=max(STATE['seen'][c],1)

    scores=_profile_scores(STATE['seen'])
    profile,conf=_choose_profile(scores,STATE.get('profile','unknown'))
    # A visible control package is strategically decisive even when the deck also
    # contains the fast acceleration shell.
    if any(STATE['seen'][c] for c in (25,1116,1221,1081,1123,1152,1097)):
        profile='grass_control'; conf=max(conf,0.75)
    if profile=='grass_fast' and not any(STATE['seen'][c] for c in FAST_DISTINCTIVE):
        profile='grass_unknown'; conf=min(conf,0.35)
    elif profile=='grass_control' and not any(STATE['seen'][c] for c in CONTROL_DISTINCTIVE):
        profile='grass_unknown'; conf=min(conf,0.35)
    STATE['profile_scores']=scores; STATE['profile']=profile; STATE['confidence']=conf

    ma=_active(me); predicted=_predict_damage(me,opp,profile)
    hp=int((ma or {}).get('hp',0) or 0)
    STATE['predicted_damage']=predicted
    STATE['active_ko_risk']=bool(ma and predicted>=hp>0)

    grim=_count_inplay(me,{648}); line=_count_inplay(me,{646,647,648})
    fros=_count_inplay(me,{104}); munki=_count_inplay(me,{112})
    ready=any(_cid(x)==648 and _energy_n(x)>=2 for x in (me.get('active') or [])+(me.get('bench') or []) if isinstance(x,dict))
    opp_active=_active(opp); opp_hp=int((opp_active or {}).get('hp',0) or 0)

    if opp_pr<=2 or (ready and 0<opp_hp<=180): plan='finish'
    elif grim==0 or line<1: plan='recover'
    elif STATE['active_ko_risk'] and not (ready and 0<opp_hp<=180): plan='survive'
    elif turn<=4 and (grim==0 or fros==0 or munki<1): plan='engine'
    elif profile in ('grass_fast','grass_control','wall','mirror') and (fros<1 or munki<2): plan='damage_map'
    else: plan='race'
    STATE['plan']=plan
    STATE['threat']={
        'grass_fast':'accelerating weakness attacker','grass_control':'energy-shift prize race',
        'wall':'damage-prevention control','mirror':'damage-transfer mirror',
        'metal':'durable metal tempo','psychic':'hand-scaled psychic pressure',
    }.get(profile,'unclassified')
    snap={'turn':turn,'profile':profile,'plan':plan,'predicted_damage':predicted,'own_prizes':own_pr,'opp_prizes':opp_pr}
    if not STATE['history'] or STATE['history'][-1]!=snap:
        STATE['history']=(STATE['history']+[snap])[-16:]
    STATE['last_snapshot']=snap
    return STATE

def diagnostics():
    return {k:(dict(v) if isinstance(v,Counter) else v) for k,v in STATE.items() if k not in ('seen_serial',)}
