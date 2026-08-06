from __future__ import annotations
from typing import Any

def _cid(x: Any) -> int:
    if isinstance(x, dict):
        try: return int(x.get('id', x.get('cardId', 0)) or 0)
        except Exception: return 0
    try: return int(x or 0)
    except Exception: return 0

def _zone(player: dict, area: int | None) -> list:
    return {2: player.get('hand') or [], 3: player.get('discard') or [],
            4: player.get('active') or [], 5: player.get('bench') or [],
            6: player.get('prize') or []}.get(int(area or 0), [])

def _source(obs: dict, opt: dict):
    cur=obs.get('current') or {}; sel=obs.get('select') or {}
    your=int(cur.get('yourIndex',0) or 0); players=cur.get('players') or [{},{}]
    pidx=int(opt.get('playerIndex',your) if opt.get('playerIndex') is not None else your)
    if not 0 <= pidx < len(players): pidx=your
    area=opt.get('area'); idx=opt.get('index'); obj=None; zone=0
    if area==1:
        arr=sel.get('deck') or []; zone=1
        if isinstance(idx,int) and 0<=idx<len(arr): obj=arr[idx]
    elif area in (2,3,4,5,6):
        arr=_zone(players[pidx],area); zone=int(area)
        if isinstance(idx,int) and 0<=idx<len(arr): obj=arr[idx]
    elif area==7:
        arr=cur.get('stadium') or []; zone=7
        if isinstance(idx,int) and 0<=idx<len(arr): obj=arr[idx]
        elif arr: obj=arr[0]
    elif area==12:
        arr=cur.get('looking') or []; zone=12
        if isinstance(idx,int) and 0<=idx<len(arr): obj=arr[idx]
    elif isinstance(idx,int):
        arr=players[your].get('hand') or []; zone=2
        if 0<=idx<len(arr): obj=arr[idx]
    return obj,zone,0 if pidx==your else 1

def _target(obs: dict,opt: dict):
    cur=obs.get('current') or {}; your=int(cur.get('yourIndex',0) or 0); players=cur.get('players') or [{},{}]
    area=opt.get('inPlayArea'); idx=opt.get('inPlayIndex')
    if area is None and int(opt.get('type',-1) if opt.get('type') is not None else -1)==10:
        area=opt.get('area'); idx=opt.get('index')
    pidx=int(opt.get('playerIndex',your) if opt.get('playerIndex') is not None else your)
    if not 0<=pidx<len(players): pidx=your
    arr=_zone(players[pidx],area) if area in (4,5) else []
    obj=arr[idx] if isinstance(idx,int) and 0<=idx<len(arr) else None
    return obj,int(area or 0),0 if pidx==your else 1

def option_semantic(obs: dict,opt: dict) -> tuple[int,...]:
    src,sz,sr=_source(obs,opt); tgt,ta,tr=_target(obs,opt)
    return (int(opt.get('type',-1) if opt.get('type') is not None else -1),
            _cid(src),sz,sr,_cid(tgt),ta,tr,
            int(opt.get('attackId',0) or 0),int(opt.get('number',0) or 0))

def action_semantic(obs: dict, action: list[int]) -> tuple[tuple[int,...],...]:
    options=((obs.get('select') or {}).get('option') or [])
    return tuple(option_semantic(obs,options[i]) for i in action if isinstance(i,int) and 0<=i<len(options))

def legal(obs: dict, action: list[int]) -> bool:
    if not isinstance(action,list) or any(not isinstance(i,int) for i in action): return False
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    mn=int(sel.get('minCount',0) or 0); mx=int(sel.get('maxCount',len(opts)) or 0)
    return mn<=len(action)<=mx and len(set(action))==len(action) and all(0<=i<len(opts) for i in action)

_CARD_ROLE={7:'DARK',104:'FROSLASS',112:'MUNKIDORI',646:'IMPIDIMP',647:'MORGREM',648:'GRIMMSNARL',860:'SNORUNT',1079:'RARE_CANDY',1080:'UNFAIR_STAMP',1086:'POFFIN',1097:'NIGHT_STRETCHER',1122:'POKEGEAR',1137:'TOOL_SCRAPPER',1152:'POKE_PAD',1182:'BOSS',1219:'PETREL',1227:'LILLIE',1231:'DAWN',1259:'SPIKEMUTH'}
def action_role_label(context:int, sem:tuple[tuple[int,...],...])->str:
    if not sem:return 'NONE'
    if len(sem)>1:return '+'.join(action_role_label(context,(x,)) for x in sem)
    t,sid,sz,sr,tid,ta,tr,atk,num=sem[0]
    if context==0:
        if t==14:return 'END'
        if t==13:return 'ATTACK_'+str(atk)
        if t==12:return 'RETREAT'
        if t==10:return 'ABILITY_'+_CARD_ROLE.get(sid,str(sid))
        if t==8:return 'ATTACH_'+_CARD_ROLE.get(tid,str(tid))
        if t==9:return 'EVOLVE_'+_CARD_ROLE.get(sid,str(sid))
        if t==7:return 'PLAY_'+_CARD_ROLE.get(sid,str(sid))
    return f'{t}:{sid}:{tid}:{atk}:{num}:n{len(sem)}'
