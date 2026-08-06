from __future__ import annotations
"""Narrow public-state residual advisor.

It never changes an action class.  It only accepts two high-confidence
corrections: finish a visibly damaged target with a 30-damage allocation, or
use accelerated Energy to build the next evolved attacker instead of
repeatedly overloading an already functional/basic target.
"""
import human_controller as hc
import human_memory as hm

GRIM=648; MORGREM=647; IMPIDIMP=646
PSYCHIC_SIGNATURES=(741,742,743,245,66,305)


def _ctx(obs):
    s=obs.get('select') or {}
    return int(s.get('context',-1) if s.get('context') is not None else -1)


def _safe_profile(mem):
    seen=mem.get('seen') or {}
    if mem.get('profile')=='psychic': return False
    return not any(int(seen.get(c,0) or 0)>0 for c in PSYCHIC_SIGNATURES)


def _single_card(obs, action):
    opts=(obs.get('select') or {}).get('option') or []
    if not isinstance(action,list) or len(action)!=1: return None,None
    i=action[0]
    if not isinstance(i,int) or not (0<=i<len(opts)): return None,None
    return hc._card_at(obs,opts[i]),opts[i]


def _preserve_active_grim(obs, baseline):
    if _ctx(obs)!=21: return False
    card,opt=_single_card(obs,baseline)
    return bool(card and hc._cid(card)==GRIM and hc._energy(card)<2 and int(opt.get('area',0) or 0)==4)


def _finish_damaged_target(obs, baseline, direct, mem):
    ctx=_ctx(obs)
    if ctx not in (13,15): return baseline
    bc,_=_single_card(obs,baseline); dc,do=_single_card(obs,direct)
    if bc is None or dc is None: return baseline
    hp=hc._hp(dc); maxhp=int(dc.get('maxHp',0) or 0); damage=hc._damage(dc)
    # A 30-damage allocation is accepted only when it visibly completes a KO
    # on an already damaged target, or on a high-HP two-prize target.  This
    # deliberately excludes generic undamaged 30-HP pivots.
    if not (0<hp<=30 and (damage>0 or maxhp>=180)): return baseline
    if ctx==15 and int(do.get('area',0) or 0)==5 and hc._cid(dc) in hc.TERA_BENCH: return baseline
    bs=hc._damage_target_score(bc,mem,ctx==15,int((_single_card(obs,baseline)[1] or {}).get('area',0) or 0)==5)
    ds=hc._damage_target_score(dc,mem,ctx==15,int(do.get('area',0) or 0)==5)
    return direct if ds-bs>=45 else baseline


def _fund_next_attacker(obs, baseline, direct, mem):
    if _ctx(obs)!=21 or _preserve_active_grim(obs,baseline): return baseline
    bc,bo=_single_card(obs,baseline); dc,do=_single_card(obs,direct)
    if bc is None or dc is None: return baseline
    bcid,dcid=hc._cid(bc),hc._cid(dc); be,de=hc._energy(bc),hc._energy(dc)
    if int(do.get('area',0) or 0)==4: return baseline
    # Only redirect to an evolved Grimmsnarl-line target that still needs fuel.
    useful=False
    if dcid==GRIM and de<2:
        useful=(bcid==IMPIDIMP and be>=1) or (bcid==MORGREM and be>=2) or (bcid==GRIM and be>=3)
    elif dcid==MORGREM and de<2:
        useful=(bcid==GRIM and be>=3) or (bcid==IMPIDIMP and be>=2)
    if not useful: return baseline
    _,me,opp=hc._players(obs)
    bs=hc._attach_target_score(bc,mem,me,opp,int(bo.get('area',0) or 0)==4)
    ds=hc._attach_target_score(dc,mem,me,opp,False)
    return direct if ds-bs>=70 else baseline


def choose(obs, baseline):
    if not hc._legal(obs,baseline): return baseline
    ctx=_ctx(obs)
    if ctx not in (13,15,21): return baseline
    mem=hm.update(obs)
    if not _safe_profile(mem): return baseline
    direct=hc._direct_selection(obs,mem)
    if direct is None or direct==baseline or not hc._legal(obs,direct): return baseline
    if ctx in (13,15): return _finish_damaged_target(obs,baseline,direct,mem)
    return _fund_next_attacker(obs,baseline,direct,mem)
