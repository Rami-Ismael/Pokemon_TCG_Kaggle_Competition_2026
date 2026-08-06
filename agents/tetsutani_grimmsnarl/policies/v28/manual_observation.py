from __future__ import annotations
import csv
from pathlib import Path

BASE=Path(__file__).resolve().parent
_cards=None
_stage={};_ctype={};_ccat={}

def get_card_id(x):
    return int(x.get('id',0) or 0) if isinstance(x,dict) else 0

def _zone_list(player,area):
    if not isinstance(player,dict):return []
    return {2:player.get('hand') or [],3:player.get('discard') or [],4:player.get('active') or [],5:player.get('bench') or [],6:player.get('prize') or []}.get(area,[])

def _rel_idx(index,your):
    try:return 0 if int(index)==int(your) else 1
    except Exception:return -1

def resolve_source(obs,opt):
    cur=obs.get('current') or {};sel=obs.get('select') or {};your=int(cur.get('yourIndex',0) or 0)
    players=cur.get('players') or [{},{}]
    pidx=int(opt.get('playerIndex',your) if opt.get('playerIndex') is not None else your)
    if not 0<=pidx<len(players):pidx=your
    area=opt.get('area');idx=opt.get('index');obj=None;zone=0
    if area==1:
        arr=sel.get('deck') or [];zone=1
        if isinstance(idx,int) and 0<=idx<len(arr):obj=arr[idx]
    elif area in (2,3,4,5,6,7):
        arr=_zone_list(players[pidx],int(area));zone=int(area)
        if isinstance(idx,int) and 0<=idx<len(arr):obj=arr[idx]
    elif isinstance(idx,int):
        arr=players[your].get('hand') or []
        if 0<=idx<len(arr):obj=arr[idx];zone=2
    return obj,zone,_rel_idx(pidx,your)

def _resolve_target(obs,opt):
    cur=obs.get('current') or {};your=int(cur.get('yourIndex',0) or 0);players=cur.get('players') or [{},{}]
    area=opt.get('inPlayArea');idx=opt.get('inPlayIndex')
    if area is None and opt.get('type')==10:area=opt.get('area');idx=opt.get('index')
    pidx=int(opt.get('playerIndex',your) if opt.get('playerIndex') is not None else your)
    if not 0<=pidx<len(players):pidx=your
    arr=_zone_list(players[pidx],int(area)) if area in (4,5) else []
    obj=arr[idx] if isinstance(idx,int) and 0<=idx<len(arr) else None
    return obj,int(area or 0),_rel_idx(pidx,your)

def semantic(obs,opt):
    src,zone,srel=resolve_source(obs,opt);tgt,tarea,trel=_resolve_target(obs,opt)
    return {
        'type':int(opt.get('type',-1) if opt.get('type') is not None else -1),
        'source_id':get_card_id(src),'source_zone':zone,'source_rel':srel,
        'target_id':get_card_id(tgt),'target_area':tarea,'target_rel':trel,
        'attack_id':int(opt.get('attackId',0) or 0),
        'area':int(opt.get('area',0) or 0),'index':int(opt.get('index',-1) if opt.get('index') is not None else -1),
        'inplay_area':int(opt.get('inPlayArea',0) or 0),'inplay_index':int(opt.get('inPlayIndex',-1) if opt.get('inPlayIndex') is not None else -1),
    }

def _load_cards():
    global _cards,_stage,_ctype,_ccat
    if _cards is not None:return
    with open(BASE.parent.parent/'EN_Card_Data.csv',encoding='utf-8-sig',newline='') as f:
        _cards={int(r['Card ID']):r for r in csv.DictReader(f)}
    _stage={v:i+1 for i,v in enumerate(sorted({r['Stage (Pokémon)/Type (Energy and Trainer)'] for r in _cards.values()}))}
    _ctype={v:i+1 for i,v in enumerate(sorted({r['Type'] for r in _cards.values()}))}
    _ccat={v:i+1 for i,v in enumerate(sorted({r['Category'] for r in _cards.values()}))}

def card_meta(cid):
    _load_cards();r=_cards.get(int(cid),{})
    def num(v):
        try:return int(str(v).replace('n/a','0').strip() or 0)
        except Exception:return 0
    return (_stage.get(r.get('Stage (Pokémon)/Type (Energy and Trainer)'),0),
            _ctype.get(r.get('Type'),0),_ccat.get(r.get('Category'),0),
            num(r.get('HP')),num(r.get('Retreat')))
