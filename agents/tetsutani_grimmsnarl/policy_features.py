from __future__ import annotations
import csv, json, hashlib, collections
from pathlib import Path

BASE=Path(__file__).resolve().parent
DECK=[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1122, 1137, 1152, 1152, 1152, 1152, 1182, 1182, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259]
DECK_IDS=sorted(set(DECK))
cards={}
with open(BASE/'EN_Card_Data.csv',encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f): cards[int(r['Card ID'])]=r
stage_vals=sorted({r['Stage (Pokémon)/Type (Energy and Trainer)'] for r in cards.values()})
type_vals=sorted({r['Type'] for r in cards.values()})
cat_vals=sorted({r['Category'] for r in cards.values()})
STAGE={v:i+1 for i,v in enumerate(stage_vals)}; CTYPE={v:i+1 for i,v in enumerate(type_vals)}; CCAT={v:i+1 for i,v in enumerate(cat_vals)}

def rel_idx(x,your):
    try:return 0 if int(x)==int(your) else 1
    except:return -1

def get_card_id(x):
    return int(x.get('id',0) or 0) if isinstance(x,dict) else 0

def get_serial(x):
    return int(x.get('serial',-1) if isinstance(x,dict) else -1)

def zone_list(player, area):
    if not isinstance(player,dict): return []
    return {2:player.get('hand') or [],3:player.get('discard') or [],4:player.get('active') or [],5:player.get('bench') or [],6:player.get('prize') or []}.get(area,[])

def resolve_source(obs, opt):
    cur=obs.get('current') or {}; sel=obs.get('select') or {}; your=int(cur.get('yourIndex',0) or 0)
    players=cur.get('players') or [{},{}]
    pidx=int(opt.get('playerIndex',your) if opt.get('playerIndex') is not None else your)
    if not (0<=pidx<len(players)):pidx=your
    area=opt.get('area')
    idx=opt.get('index')
    obj=None; zone=0
    if area==1:
        arr=sel.get('deck') or []; zone=1
        if isinstance(idx,int) and 0<=idx<len(arr): obj=arr[idx]
    elif area in (2,3,4,5,6,7):
        arr=zone_list(players[pidx],int(area)); zone=int(area)
        if isinstance(idx,int) and 0<=idx<len(arr): obj=arr[idx]
    elif isinstance(idx,int):
        # MAIN options with only an index normally point into the acting player's hand.
        arr=(players[your].get('hand') or []) if 0<=your<len(players) else []
        if 0<=idx<len(arr): obj=arr[idx]; zone=2
    return obj,zone,rel_idx(pidx,your)

def resolve_target(obs,opt):
    cur=obs.get('current') or {}; your=int(cur.get('yourIndex',0) or 0); players=cur.get('players') or [{},{}]
    area=opt.get('inPlayArea')
    idx=opt.get('inPlayIndex')
    if area is None and opt.get('type')==10: area=opt.get('area'); idx=opt.get('index')
    pidx=int(opt.get('playerIndex',your) if opt.get('playerIndex') is not None else your)
    if not (0<=pidx<len(players)):pidx=your
    arr=zone_list(players[pidx],int(area)) if area in (4,5) else []
    obj=arr[idx] if isinstance(idx,int) and 0<=idx<len(arr) else None
    return obj,int(area or 0),rel_idx(pidx,your)

def semantic(obs,opt):
    src,zone,src_rel=resolve_source(obs,opt); tgt,tarea,tgt_rel=resolve_target(obs,opt)
    return {
      'type':int(opt.get('type',-1) if opt.get('type') is not None else -1),
      'source_id':get_card_id(src),'source_serial':get_serial(src),'source_zone':zone,'source_rel':src_rel,
      'target_id':get_card_id(tgt),'target_serial':get_serial(tgt),'target_area':tarea,'target_rel':tgt_rel,
      'attack_id':int(opt.get('attackId',0) or 0),
      'area':int(opt.get('area',0) or 0),'index':int(opt.get('index',-1) if opt.get('index') is not None else -1),
      'inplay_area':int(opt.get('inPlayArea',0) or 0),'inplay_index':int(opt.get('inPlayIndex',-1) if opt.get('inPlayIndex') is not None else -1),
    }

def intent_text(s):
    if s['attack_id']: return f"t{s['type']}:a{s['attack_id']}"
    if s['source_id']: return f"t{s['type']}:c{s['source_id']}"
    return f"t{s['type']}"

def card_meta(cid):
    r=cards.get(int(cid),{})
    def num(v,default=0):
        try:return int(str(v).replace('n/a','0').strip() or default)
        except:return default
    return STAGE.get(r.get('Stage (Pokémon)/Type (Energy and Trainer)'),0),CTYPE.get(r.get('Type'),0),CCAT.get(r.get('Category'),0),num(r.get('HP')),num(r.get('Retreat'))

def normalize_obj(x,your,drop_serial=False):
    if isinstance(x,dict):
        out={}
        for k,v in x.items():
            if k in ('logs','remainingOverageTime','step'): continue
            if drop_serial and k=='serial': continue
            if k=='playerIndex': out[k]=rel_idx(v,your)
            elif k=='yourIndex': out[k]=0
            elif k=='firstPlayer': out[k]=(-1 if v==-1 else rel_idx(v,your))
            else: out[k]=normalize_obj(v,your,drop_serial)
        return out
    if isinstance(x,list): return [normalize_obj(v,your,drop_serial) for v in x]
    return x

def hash_key(obs,prev,drop_serial=False,coarse=False):
    cur=obs.get('current') or {}; your=int(cur.get('yourIndex',0) or 0)
    if coarse:
        p=cur.get('players') or [{},{}]; me=p[your] if your<len(p) else {}; opp=p[1-your] if len(p)>1 else {}
        sel=obs.get('select') or {}
        obj={
          'turn':cur.get('turn'),'tac':cur.get('turnActionCount'),'fp':rel_idx(cur.get('firstPlayer',-1),your) if cur.get('firstPlayer',-1)!=-1 else -1,
          'flags':[cur.get(k) for k in ('energyAttached','supporterPlayed','stadiumPlayed','retreated')],
          'me':{'deck':me.get('deckCount'),'hand':[get_card_id(c) for c in (me.get('hand') or [])], 'discard':sorted(get_card_id(c) for c in (me.get('discard') or [])),
                'active':[get_card_id(c) for c in (me.get('active') or [])],'bench':[get_card_id(c) for c in (me.get('bench') or [])]},
          'opp':{'handCount':opp.get('handCount'),'active':[get_card_id(c) for c in (opp.get('active') or [])],'bench':[get_card_id(c) for c in (opp.get('bench') or [])]},
          'sel':{'context':sel.get('context'),'type':sel.get('type'),'min':sel.get('minCount'),'max':sel.get('maxCount'),'effect':get_card_id(sel.get('effect')),'cc':get_card_id(sel.get('contextCard')),
                 'opts':[semantic(obs,o) for o in (sel.get('option') or [])]},
          'prev':prev[-2:],
        }
    else:
        obj={'current':normalize_obj(cur,your,drop_serial),'select':normalize_obj(obs.get('select'),your,drop_serial),'prev':prev[-3:]}
    raw=json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.blake2b(raw,digest_size=16).hexdigest()

def pokemon_stats(obj):
    if not isinstance(obj,dict): return (0,0,0,0,0,0)
    hp=int(obj.get('hp',0) or 0); mh=int(obj.get('maxHp',0) or 0)
    return (get_card_id(obj),hp,mh,max(0,mh-hp),len(obj.get('energyCards') or []),1 if obj.get('appearThisTurn') else 0)

def base_state(obs,history):
    cur=obs.get('current') or {}; sel=obs.get('select') or {}; your=int(cur.get('yourIndex',0) or 0)
    ps=cur.get('players') or [{},{}]; me=ps[your] if your<len(ps) else {}; opp=ps[1-your] if len(ps)>1 else {}
    row={}
    row.update({
      'turn':int(cur.get('turn',0) or 0),'turn_action_count':int(cur.get('turnActionCount',0) or 0),
      'first_player_rel':(-1 if cur.get('firstPlayer',-1)==-1 else rel_idx(cur.get('firstPlayer'),your)),
      'energy_attached':int(bool(cur.get('energyAttached'))),'supporter_played':int(bool(cur.get('supporterPlayed'))),
      'stadium_played':int(bool(cur.get('stadiumPlayed'))),'retreated':int(bool(cur.get('retreated'))),
      'context':int(sel.get('context',-1) if sel.get('context') is not None else -1),'select_type':int(sel.get('type',-1) if sel.get('type') is not None else -1),
      'min_count':int(sel.get('minCount',0) or 0),'max_count':int(sel.get('maxCount',0) or 0),'n_options':len(sel.get('option') or []),
      'remain_damage':int(sel.get('remainDamageCounter',0) or 0),'remain_energy':int(sel.get('remainEnergyCost',0) or 0),
      'effect_id':get_card_id(sel.get('effect')),'context_card_id':get_card_id(sel.get('contextCard')),
      'stadium_id':get_card_id((cur.get('stadium') or [None])[0] if (cur.get('stadium') or []) else None),
    })
    for prefix,p in [('own',me),('opp',opp)]:
        row[f'{prefix}_deck_count']=int(p.get('deckCount',0) or 0); row[f'{prefix}_hand_count']=int(p.get('handCount',0) or 0)
        row[f'{prefix}_prize_count']=len(p.get('prize') or []); row[f'{prefix}_bench_n']=len(p.get('bench') or [])
        for st in ('asleep','burned','confused','paralyzed','poisoned'): row[f'{prefix}_{st}']=int(bool(p.get(st)))
        active=(p.get('active') or [None])[0] if (p.get('active') or []) else None
        aid,ahp,amh,admg,aen,aapp=pokemon_stats(active)
        for n,v in zip(('active_id','active_hp','active_maxhp','active_damage','active_energy_n','active_appear'),(aid,ahp,amh,admg,aen,aapp)):row[f'{prefix}_{n}']=v
        bench=p.get('bench') or []
        row[f'{prefix}_total_inplay_hp']=ahp+sum(int(c.get('hp',0) or 0) for c in bench)
        row[f'{prefix}_total_inplay_damage']=admg+sum(max(0,int(c.get('maxHp',0) or 0)-int(c.get('hp',0) or 0)) for c in bench)
        row[f'{prefix}_total_energy']=aen+sum(len(c.get('energyCards') or []) for c in bench)
        for j in range(5):
            vals=pokemon_stats(bench[j] if j<len(bench) else None)
            for n,v in zip(('id','hp','maxhp','damage','energy_n','appear'),vals): row[f'{prefix}_bench{j}_{n}']=v
    hand=me.get('hand') or []; disc=me.get('discard') or []; inplay=(me.get('active') or [])+(me.get('bench') or [])
    hc=collections.Counter(get_card_id(c) for c in hand); dc=collections.Counter(get_card_id(c) for c in disc); ic=collections.Counter(get_card_id(c) for c in inplay)
    for cid in DECK_IDS:
        row[f'hand_c{cid}']=hc[cid]; row[f'discard_c{cid}']=dc[cid]; row[f'inplay_c{cid}']=ic[cid]
    for j in range(20): row[f'hand_pos{j}_id']=get_card_id(hand[j]) if j<len(hand) else 0
    otype=collections.Counter(int(o.get('type',-1) if o.get('type') is not None else -1) for o in (sel.get('option') or []))
    for t in range(16): row[f'opt_type_count_{t}']=otype[t]
    for hidx in range(3):
        h=history[-1-hidx] if len(history)>hidx else None
        for k in ('type','source_id','target_id','attack_id','source_zone','target_area'):
            row[f'prev{hidx+1}_{k}']=int(h.get(k,0) if h else 0)
    return row

def option_row(obs,state,opt,pos,s,dup_count,dup_rank):
    src,zone,_=resolve_source(obs,opt); tgt,_,_=resolve_target(obs,opt)
    row=dict(state)
    row.update({
      'opt_pos':pos,'opt_pos_norm':pos/max(1,len((obs.get('select') or {}).get('option') or [])-1),
      'opt_type':s['type'],'opt_area':s['area'],'opt_index':s['index'],'opt_player_rel':s['source_rel'],
      'opt_inplay_area':s['inplay_area'],'opt_inplay_index':s['inplay_index'],'opt_attack_id':s['attack_id'],
      'opt_source_id':s['source_id'],'opt_source_serial':s['source_serial'],'opt_source_zone':s['source_zone'],
      'opt_target_id':s['target_id'],'opt_target_serial':s['target_serial'],'opt_target_area':s['target_area'],'opt_target_rel':s['target_rel'],
    })
    st,ct,cc,hp,ret=card_meta(s['source_id']); row.update({'source_stage':st,'source_card_type':ct,'source_category':cc,'source_meta_hp':hp,'source_retreat':ret})
    tt,tc,tcat,thp,tret=card_meta(s['target_id']); row.update({'target_stage':tt,'target_card_type':tc,'target_category':tcat,'target_meta_hp':thp,'target_retreat':tret})
    if isinstance(tgt,dict):
        mh=int(tgt.get('maxHp',0) or 0); ch=int(tgt.get('hp',0) or 0)
        row.update({'target_current_hp':ch,'target_damage':max(0,mh-ch),'target_energy_n':len(tgt.get('energyCards') or []),'target_appear':int(bool(tgt.get('appearThisTurn')))})
    else: row.update({'target_current_hp':0,'target_damage':0,'target_energy_n':0,'target_appear':0})
    row['semantic_dup_count']=dup_count; row['semantic_dup_rank']=dup_rank
    return row,s
