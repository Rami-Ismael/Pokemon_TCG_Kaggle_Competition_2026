from __future__ import annotations
"""Conservative public-state endgame guard.

The baseline remains authoritative unless the visible board proves an exact
prize-taking line. No opponent identity or hidden information is used.
"""
import policy_features as pf
import human_controller as hc

GRIM=648
BOSS=1182
BLOCKING_ACTIVE={117,345}
PRIZE_VALUES={24: 2, 29: 2, 30: 2, 37: 2, 40: 2, 44: 2, 46: 2, 52: 2, 63: 2, 75: 2, 79: 2, 80: 2, 83: 2, 84: 2, 96: 2, 99: 2, 107: 2, 108: 2, 117: 2, 121: 2, 125: 2, 130: 2, 138: 2, 139: 2, 140: 2, 141: 2, 150: 2, 153: 2, 154: 2, 161: 2, 176: 2, 179: 2, 184: 2, 189: 2, 190: 2, 193: 2, 198: 2, 205: 2, 207: 2, 210: 2, 223: 2, 229: 2, 231: 2, 232: 2, 236: 2, 239: 2, 241: 2, 243: 2, 244: 2, 246: 2, 248: 2, 249: 2, 259: 2, 269: 2, 272: 2, 283: 2, 293: 2, 299: 2, 302: 2, 306: 2, 313: 2, 316: 2, 320: 2, 326: 2, 328: 2, 329: 2, 331: 2, 336: 2, 337: 2, 340: 2, 357: 2, 369: 2, 372: 2, 381: 2, 389: 2, 404: 2, 407: 2, 424: 2, 431: 2, 447: 2, 455: 2, 458: 2, 471: 2, 481: 2, 509: 2, 515: 2, 525: 2, 527: 2, 547: 2, 561: 2, 573: 2, 583: 2, 598: 2, 618: 2, 631: 2, 641: 2, 648: 2, 652: 3, 662: 3, 678: 3, 687: 3, 695: 3, 723: 3, 737: 3, 747: 3, 754: 3, 756: 3, 766: 3, 772: 3, 781: 3, 790: 3, 795: 2, 806: 2, 813: 2, 828: 3, 835: 2, 849: 3, 861: 3, 868: 3, 886: 3, 896: 3, 904: 3, 911: 2, 919: 3, 928: 3, 932: 3, 939: 3, 944: 2, 951: 2, 954: 2, 957: 2, 962: 2, 968: 2, 969: 2, 975: 2, 979: 2, 984: 2, 988: 2, 990: 2, 993: 2, 997: 2, 1002: 2, 1006: 3, 1022: 2, 1026: 2, 1031: 3, 1040: 3, 1056: 3, 1062: 2, 1064: 3, 1071: 2}
_PENDING_BOSS_SERIAL=None

def reset():
    global _PENDING_BOSS_SERIAL
    _PENDING_BOSS_SERIAL=None

def _ctx(obs):
    value=(obs.get('select') or {}).get('context')
    return int(value if value is not None else -1)

def _effect(obs):
    s=obs.get('select') or {}
    return hc._cid(s.get('effect')) or hc._cid(s.get('contextCard'))

def _active(player):
    cards=(player or {}).get('active') or []
    return cards[0] if cards and isinstance(cards[0],dict) else None

def _attack_ready(card):
    return hc._cid(card)==GRIM and hc._energy(card)>=2

def _targetable(card):
    return card is not None and hc._cid(card) not in BLOCKING_ACTIVE and 0<hc._hp(card)<=180

def choose(obs,baseline):
    global _PENDING_BOSS_SERIAL
    if not hc._legal(obs,baseline):
        return baseline
    select=obs.get('select') or {}
    options=select.get('option') or []
    context=_ctx(obs)
    effect=_effect(obs)

    if _PENDING_BOSS_SERIAL is not None and (context==3 or effect==BOSS):
        for i,option in enumerate(options):
            card=hc._card_at(obs,option)
            if int((card or {}).get('serial',-1) or -1)==_PENDING_BOSS_SERIAL:
                _PENDING_BOSS_SERIAL=None
                action=[i]
                return action if hc._legal(obs,action) else baseline
        _PENDING_BOSS_SERIAL=None

    if context!=0:
        return baseline
    _,me,opponent=hc._players(obs)
    active=_active(me)
    opposing_active=_active(opponent)
    if not _attack_ready(active):
        return baseline
    remaining=len((me or {}).get('prize') or [])
    if remaining<=0 or remaining>3:
        return baseline
    if _targetable(opposing_active) and PRIZE_VALUES.get(hc._cid(opposing_active),1)>=remaining:
        return baseline

    boss_index=None
    for i,option in enumerate(options):
        semantic=pf.semantic(obs,option)
        if int(semantic.get('type',-1))==7 and int(semantic.get('source_id',0) or 0)==BOSS:
            boss_index=i
            break
    if boss_index is None:
        return baseline

    targets=[]
    for card in (opponent or {}).get('bench') or []:
        if not isinstance(card,dict) or not _targetable(card):
            continue
        prize_value=PRIZE_VALUES.get(hc._cid(card),1)
        if prize_value<remaining:
            continue
        serial=int(card.get('serial',-1) or -1)
        if serial<0:
            continue
        targets.append((prize_value,-hc._hp(card),int(card.get('maxHp',0) or 0),-serial,serial))
    if not targets:
        return baseline

    action=[boss_index]
    if not hc._legal(obs,action):
        return baseline
    _PENDING_BOSS_SERIAL=max(targets)[-1]
    return action
