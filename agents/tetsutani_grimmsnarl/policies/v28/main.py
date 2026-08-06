from __future__ import annotations
import os
from .v22_policy import agent as _v22_agent
from .v26_manual_policy import agent as _v26_agent
from .semantic_resolver import action_semantic, action_role_label, legal

PLAYER_NAME='fixed-deck expert'
BUILD_ID='ptcg_fixed_deck_expert_complete_handwritten_hierarchical_hybrid_v28_public_state_build'
DECK=[7,7,7,7,7,7,7,7,7,7,104,104,112,112,112,112,646,646,646,646,647,647,647,648,648,648,860,860,1079,1079,1079,1080,1086,1086,1086,1086,1097,1097,1097,1122,1137,1152,1152,1152,1152,1182,1182,1219,1219,1219,1219,1227,1227,1227,1227,1231,1259,1259,1259,1259]
_V26_CONTEXTS={7,21,22}
# Explicit conflict rules recovered from repeated tactical motifs. These are
# card-role/decision-context rules, not observation hashes or learned weights.
_CONFLICT_RULES={
    (0,'END','RETREAT'):'v26',
    (0,'PLAY_SNORUNT','END'):'v26',
    (0,'PLAY_BOSS','ATTACK_937'):'v26',
    (0,'ATTACH_MORGREM','RETREAT'):'v26',
    (0,'PLAY_BOSS','ATTACH_IMPIDIMP'):'v26',
    (0,'ABILITY_MUNKIDORI','ABILITY_MUNKIDORI'):'v26',
    (5,'3:646:0:0:0:n1+3:646:0:0:0:n1','3:646:0:0:0:n1+3:860:0:0:0:n1'):'v26',
    (7,'3:1259:0:0:0:n1','3:1079:0:0:0:n1'):'v22',
    (7,'3:646:0:0:0:n1','3:104:0:0:0:n1'):'v22',
    (7,'3:1227:0:0:0:n1','3:1219:0:0:0:n1'):'v22',
}

def _read_deck() -> list[int]:
    paths=[os.path.join(os.path.dirname(__file__),'deck.csv'),'deck.csv','/kaggle_simulations/agent/deck.csv']
    for path in paths:
        try:
            with open(path,encoding='utf-8') as f: deck=[int(x.strip()) for x in f if x.strip()]
            if len(deck)==60:return deck
        except OSError: pass
    return list(DECK)

def agent(obs_dict:dict)->list[int]:
    # Both deterministic handwritten engines are advanced on every observation;
    # arbitration only selects between their already-legal domain decisions.
    if not obs_dict or obs_dict.get('select') is None:
        try:_v22_agent(obs_dict or {'select':None})
        except Exception:pass
        try:_v26_agent(obs_dict or {'select':None})
        except Exception:pass
        return _read_deck()
    try:a22=_v22_agent(obs_dict)
    except Exception:a22=[]
    try:a26=_v26_agent(obs_dict)
    except Exception:a26=[]
    ok22=legal(obs_dict,a22);ok26=legal(obs_dict,a26)
    if not ok22 and ok26:return a26
    if not ok26 and ok22:return a22
    if not ok22 and not ok26:
        sel=obs_dict.get('select') or {};n=len(sel.get('option') or []);mn=int(sel.get('minCount',0) or 0);mx=int(sel.get('maxCount',n) or 0)
        return list(range(min(n,max(mn,min(mx,n)))))
    ctx=int((obs_dict.get('select') or {}).get('context',-1))
    sem22=action_semantic(obs_dict,a22);sem26=action_semantic(obs_dict,a26)
    l22=action_role_label(ctx,sem22);l26=action_role_label(ctx,sem26)
    conflict=_CONFLICT_RULES.get((ctx,l22,l26))
    if conflict=='v26':return a26
    if conflict=='v22':return a22
    # v26 is descriptively stronger for card search, Punk Up distribution and
    # Punk Up count. Elsewhere v22 supplies the higher-fidelity strategic choice.
    if ctx in _V26_CONTEXTS:return a26
    # When both engines agree semantically, v26's physical-copy/index resolver
    # is more faithful and cannot alter the strategic action.
    if sem22==sem26:return a26
    return a22
