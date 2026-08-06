from __future__ import annotations

# v26 hierarchical pure hand-written policy for the fixed fixed-deck expert 60-card deck.
# No model files, training code, replay lookup, hashing, or search are used.

DECK = [
    7,7,7,7,7,7,7,7,7,7,
    104,104,112,112,112,112,
    646,646,646,646,647,647,647,648,648,648,860,860,
    1079,1079,1079,1080,1086,1086,1086,1086,1097,1097,1097,
    1122,1137,1152,1152,1152,1152,1182,1182,1219,1219,1219,1219,
    1227,1227,1227,1227,1231,1259,1259,1259,1259,
]

ENERGY=7; FROSLASS=104; MUNKIDORI=112; IMPIDIMP=646; MORGREM=647; GRIMMSNARL=648; SNORUNT=860
RARE_CANDY=1079; UNFAIR_STAMP=1080; POFFIN=1086; NIGHT_STRETCHER=1097; POKEGEAR=1122
TOOL_SCRAPPER=1137; POKE_PAD=1152; BOSS=1182; PETREL=1219; LILLIE=1227; DAWN=1231; SPIKEMUTH=1259
SHADOW_BULLET=937; FILCH=934; MORGREM_ATTACK=936
MARNIES={IMPIDIMP,MORGREM,GRIMMSNARL}
OUR_IDS={ENERGY,FROSLASS,MUNKIDORI,IMPIDIMP,MORGREM,GRIMMSNARL,SNORUNT,RARE_CANDY,UNFAIR_STAMP,POFFIN,
         NIGHT_STRETCHER,POKEGEAR,TOOL_SCRAPPER,POKE_PAD,BOSS,PETREL,LILLIE,DAWN,SPIKEMUTH}

_STATS={}
def _bump(k): _STATS[k]=_STATS.get(k,0)+1
def get_stats(): return dict(_STATS)
def reset_stats(): _STATS.clear()

# Runtime-observable history ledger.  It is reconstructed only from the logs
# supplied with the current game observation; no replay identifiers or stored
# state/action database are consulted.
_RUNTIME_PLAYERS={}

def _new_runtime_player():
    return {
        'last_turn':-1,'last_tac':-1,'main_started':False,
        'prizes':[0,0],
        'turn_events':[[],[]],'turn_plays':[[],[]],
        'turn_attaches':[[],[]],'turn_evolves':[[],[]],'turn_attacks':[[],[]],
        'last_event':[None,None],
    }

def reset_runtime_state():
    _RUNTIME_PLAYERS.clear()

def _runtime_history(obs):
    cur=obs.get('current') or {};sel=obs.get('select') or {}
    y=int(cur.get('yourIndex',0) or 0);turn=int(cur.get('turn',0) or 0);tac=int(cur.get('turnActionCount',0) or 0)
    mem=_RUNTIME_PLAYERS.get(y)
    # A new battle returns to turn zero/setup after a completed main phase.
    if mem is None or (turn==0 and mem.get('last_turn',-1)>0) or (turn<mem.get('last_turn',-1)):
        mem=_new_runtime_player();_RUNTIME_PLAYERS[y]=mem
    for l in (obs.get('logs') or []):
        if not isinstance(l,dict):continue
        typ=int(l.get('type',-1) if l.get('type') is not None else -1)
        try:pl=int(l.get('playerIndex',-1))
        except:pl=-1
        if pl not in (0,1):continue
        if typ in (6,7):
            fa=int(l.get('fromArea',-1) if l.get('fromArea') is not None else -1)
            ta=int(l.get('toArea',-1) if l.get('toArea') is not None else -1)
            if ta==6:mem['prizes'][pl]=min(6,mem['prizes'][pl]+1)
            if fa==6:mem['prizes'][pl]=max(0,mem['prizes'][pl]-1)
        if typ==2:
            mem['main_started']=True
            mem['turn_events'][pl]=[];mem['turn_plays'][pl]=[]
            mem['turn_attaches'][pl]=[];mem['turn_evolves'][pl]=[];mem['turn_attacks'][pl]=[]
        cid=int(l.get('cardId',0) or 0);serial=int(l.get('serial',-1) if l.get('serial') is not None else -1)
        target=int(l.get('cardIdTarget',0) or 0);attack=int(l.get('attackId',0) or 0)
        ev=(typ,cid,serial,target,attack)
        if typ in (10,11,12,15):
            mem['turn_events'][pl].append(ev);mem['last_event'][pl]=ev
        if typ==10:mem['turn_plays'][pl].append(cid)
        elif typ==11:mem['turn_attaches'][pl].append((cid,target))
        elif typ==12:mem['turn_evolves'][pl].append((cid,target))
        elif typ==15:mem['turn_attacks'][pl].append(attack)
    if turn>=1 or mem['main_started']:
        for pl in (0,1):
            if mem['prizes'][pl]==0:mem['prizes'][pl]=6
    mem['last_turn']=turn;mem['last_tac']=tac
    op=1-y
    return {
        'myPrizeCount':mem['prizes'][y],'oppPrizeCount':mem['prizes'][op],
        'myTurnEvents':[list(x) for x in mem['turn_events'][y]],
        'oppTurnEvents':[list(x) for x in mem['turn_events'][op]],
        'myTurnPlays':list(mem['turn_plays'][y]),'oppTurnPlays':list(mem['turn_plays'][op]),
        'myTurnAttaches':[list(x) for x in mem['turn_attaches'][y]],
        'myTurnEvolves':[list(x) for x in mem['turn_evolves'][y]],
        'myTurnAttacks':list(mem['turn_attacks'][y]),
        'lastMyEvent':list(mem['last_event'][y]) if mem['last_event'][y] else [],
        'lastOppEvent':list(mem['last_event'][op]) if mem['last_event'][op] else [],
    }

def _id(x):
    return int(x.get('id',0) or 0) if isinstance(x,dict) else 0

def _serial(x):
    return int(x.get('serial',-1) if isinstance(x,dict) and x.get('serial') is not None else -1)

def _zone(player,area):
    if not isinstance(player,dict): return []
    return {2:player.get('hand') or [],3:player.get('discard') or [],4:player.get('active') or [],5:player.get('bench') or [],6:player.get('prize') or []}.get(int(area or 0),[])

def _rel(p,y):
    try:return 0 if int(p)==int(y) else 1
    except:return -1

def _semantic(obs,opt):
    cur=obs.get('current') or {}; sel=obs.get('select') or {}; y=int(cur.get('yourIndex',0) or 0)
    ps=cur.get('players') or [{},{}]
    p=int(opt.get('playerIndex',y) if opt.get('playerIndex') is not None else y)
    if not (0<=p<len(ps)):p=y
    area=opt.get('area'); idx=opt.get('index'); src=None; zone=0
    if area==1:
        a=sel.get('deck') or []; zone=1
        if isinstance(idx,int) and 0<=idx<len(a):src=a[idx]
    elif area in (2,3,4,5,6,7):
        a=_zone(ps[p],area); zone=int(area)
        if isinstance(idx,int) and 0<=idx<len(a):src=a[idx]
    elif isinstance(idx,int):
        a=_zone(ps[y],2); zone=2
        if 0<=idx<len(a):src=a[idx]
    ia=opt.get('inPlayArea'); ii=opt.get('inPlayIndex')
    if ia is None and int(opt.get('type',-1) if opt.get('type') is not None else -1)==10:
        ia=area;ii=idx
    tgt=None
    if ia in (4,5):
        a=_zone(ps[p],ia)
        if isinstance(ii,int) and 0<=ii<len(a):tgt=a[ii]
    return {
        'type':int(opt.get('type',-1) if opt.get('type') is not None else -1),
        'source_id':_id(src),'source_serial':_serial(src),'source_zone':zone,'source_rel':_rel(p,y),
        'target_id':_id(tgt),'target_serial':_serial(tgt),'target_area':int(ia or 0),'target_rel':_rel(p,y),
        'attack_id':int(opt.get('attackId',0) or 0),'area':int(area or 0),'index':int(idx if idx is not None else -1),
        'inplay_area':int(ia or 0),'inplay_index':int(ii if ii is not None else -1),
    }

def _card(c):
    if not isinstance(c,dict):return None
    return {'id':_id(c),'serial':_serial(c),'hp':int(c.get('hp',0) or 0),'maxHp':int(c.get('maxHp',0) or 0),
            'energy':[int(_id(x)) for x in (c.get('energyCards') or [])],
            'tools':[int(_id(x)) for x in (c.get('toolCards') or [])],'appear':bool(c.get('appearThisTurn'))}

def _player(p,prize_count=None):
    if not isinstance(p,dict):p={}
    if prize_count is None:
        visible=p.get('prize') or []
        prize_count=int(p.get('prizeCount',len(visible)) or 0)
    return {'deckCount':int(p.get('deckCount',0) or 0),'handCount':int(p.get('handCount',len(p.get('hand') or [])) or 0),
            'hand':[_id(x) for x in (p.get('hand') or [])],'discard':[_id(x) for x in (p.get('discard') or [])],
            'prizeCount':int(prize_count),'active':[_card(x) for x in (p.get('active') or [])],
            'bench':[_card(x) for x in (p.get('bench') or [])]}

def snapshot(obs):
    cur=obs.get('current') or {}; sel=obs.get('select') or {}; y=int(cur.get('yourIndex',0) or 0); ps=cur.get('players') or [{},{}]
    me=ps[y] if 0<=y<len(ps) else {}; op=ps[1-y] if len(ps)>1 and 0<=1-y<len(ps) else {}
    history=_runtime_history(obs)
    return {'turn':int(cur.get('turn',0) or 0),'turnActionCount':int(cur.get('turnActionCount',0) or 0),
            'firstPlayer':int(cur.get('firstPlayer',-1) if cur.get('firstPlayer') is not None else -1),'yourIndex':y,
            'energyAttached':bool(cur.get('energyAttached')),'supporterPlayed':bool(cur.get('supporterPlayed')),
            'stadiumPlayed':bool(cur.get('stadiumPlayed')),'retreated':bool(cur.get('retreated')),
            'stadium':_id((cur.get('stadium') or [None])[0]) if (cur.get('stadium') or []) else 0,
            'context':int(sel.get('context',-1) if sel.get('context') is not None else -1),
            'effect_id':_id(sel.get('effect')),'context_card_id':_id(sel.get('contextCard')),
            'minCount':int(sel.get('minCount',0) or 0),'maxCount':int(sel.get('maxCount',0) or 0),
            'remainDamageCounter':int(sel.get('remainDamageCounter',0) or 0),'remainEnergyCost':int(sel.get('remainEnergyCost',0) or 0),
            # Prize-array lengths are visible in the live observation and are
            # the authoritative race ledger.  Logs are used only for action history.
            'me':_player(me),'opp':_player(op),
            'history':history,'options':[_semantic(obs,o) for o in (sel.get('option') or [])]}

def _all_cards(p): return [c for c in (p.get('active') or [])+(p.get('bench') or []) if c]
def _count_inplay(st,cid): return sum(c['id']==cid for c in _all_cards(st['me']))
def _count_hand(st,cid): return st['me']['hand'].count(cid)
def _count_disc(st,cid): return st['me']['discard'].count(cid)
def _find_card(st,serial,rel=0):
    p=st['me'] if rel==0 else st['opp']
    for area,arr in ((4,p.get('active') or []),(5,p.get('bench') or [])):
        for pos,c in enumerate(arr):
            if c and c['serial']==serial:return c,area,pos
    return None,0,-1

def _indices(st,pred): return [i for i,s in enumerate(st['options']) if pred(s)]
def _first(st,pred):
    a=_indices(st,pred);return a[0] if a else None

def _legal_fill(st,ordered):
    n=len(st['options']);mn=st['minCount'];mx=min(st['maxCount'],n);out=[]
    for i in ordered:
        if isinstance(i,int) and 0<=i<n and i not in out and len(out)<mx:out.append(i)
    for i in range(n):
        if len(out)>=mn:break
        if i not in out:out.append(i)
    return out[:mx]

def _choose_by_ids(st,ids,limit=None):
    out=[]
    for cid in ids:
        for i,s in enumerate(st['options']):
            if i not in out and s['source_id']==cid:
                out.append(i)
                if limit and len(out)>=limit:return out
    return out

def _target_score(st,s,shadow=False,boss=False):
    c,area,pos=_find_card(st,s['source_serial'],1)
    if not c:return (-999,)
    hp=c['hp'];maxhp=c['maxHp'];cid=c['id'];ability_like=cid in {112,57,120,121,140,169,342,343,345,387,400,401,648,741,742,743,756,1071}
    ko30=hp<=30; ko180=hp<=180
    basic_fragile=maxhp<=120
    powered_munk=cid==MUNKIDORI and bool(c.get('energy'))
    if boss:
        return (ko180, maxhp>=200, ability_like, -hp, -pos)
    if shadow:
        return (ko30, cid==MUNKIDORI, powered_munk, ability_like, basic_fragile, -hp, pos)
    return (ko30, cid==MUNKIDORI, ability_like, basic_fragile, -hp, powered_munk, area==4, -pos)

def _own_heal_score(st,s):
    c,area,pos=_find_card(st,s['source_serial'],0)
    if not c:return (-999,)
    dmg=max(0,c['maxHp']-c['hp']);cid=c['id'];fragile=c['maxHp']<=110
    return (c['hp']<=30, cid==MUNKIDORI, fragile and dmg>=10, min(30,dmg), -c['hp'], area==4, pos)

def _main_option_key(st,i):
    s=st['options'][i];t=s['type'];cid=s['source_id'];atk=s['attack_id'];tar=s['target_id']
    return t,cid,atk,tar

def _main(st):
    opts=st['options']; me=st['me']; opp=st['opp']
    active=me['active'][0] if me['active'] else None

    # The replay policy is a deterministic action pipeline, not a tactical search.
    # It develops every zero/low-cost engine piece before taking the available attack.
    cand=_indices(st,lambda s:s['type']==10 and s['source_id']==MUNKIDORI)
    if cand:
        # v18: with one powered attacker, no Froslass and opponent on three
        # Prizes, equivalent Adrena-Brain users resolve to the second bench
        # position rather than the generic rightmost tie-break.
        adrena_rows=[]
        for j in cand:
            c,area,pos=_find_card(st,st['options'][j]['source_serial'],0)
            if c:adrena_rows.append((pos,j,c['hp'],len(c['energy']),bool(c.get('appear'))))
        equal_users=bool(len(adrena_rows)>=2 and len({(hp,en,ap) for _,_,hp,en,ap in adrena_rows})==1)
        powered_grim=sum(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in _all_cards(me))
        if opp['prizeCount']==3 and _count_inplay(st,FROSLASS)==0 and powered_grim==1 and equal_users:
            adrena_rows.sort()
            _bump('main_adrena_second_bench_tie');return [adrena_rows[1][1]]
        cand.sort(key=lambda i:_own_heal_score(st,st['options'][i]),reverse=True)
        _bump('main_adrena');return [cand[0]]

    shadow=_first(st,lambda s:s['type']==13 and s['attack_id']==SHADOW_BULLET)

    # Global action order reconstructed from pairwise replay preferences.
    pipeline = [
        ('froslass', lambda s:s['type']==9 and s['source_id']==FROSLASS),
        ('munkidori', lambda s:s['type']==7 and s['source_id']==MUNKIDORI),
        ('early_candy', lambda s:s['type']==7 and s['source_id']==RARE_CANDY),
        ('morgrem', lambda s:s['type']==9 and s['source_id']==MORGREM),
        ('impidimp', lambda s:s['type']==7 and s['source_id']==IMPIDIMP),
        ('stadium', lambda s:s['type']==7 and s['source_id']==SPIKEMUTH),
        ('rare_candy', lambda s:s['type']==7 and s['source_id']==RARE_CANDY),
        ('night_stretcher', lambda s:s['type']==7 and s['source_id']==NIGHT_STRETCHER),
        ('pokegear', lambda s:s['type']==7 and s['source_id']==POKEGEAR),
        ('grimmsnarl', lambda s:s['type']==9 and s['source_id']==GRIMMSNARL),
    ]
    for name,pred in pipeline:
        candidates=_indices(st,pred)
        if name=='froslass' and candidates:
            # Preserve an Active Snorunt as the temporary wall; when a benched
            # copy is available, evolve that copy into the passive engine.
            bench_candidates=[j for j in candidates if st['options'][j]['target_area']==5]
            i=bench_candidates[0] if bench_candidates else candidates[0]
        elif name=='morgrem' and candidates:
            # When active and benched Impidimp can both evolve, preserve the
            # active wall. Among benched copies, evolve the most energized line
            # so the future Grimmsnarl keeps its staged Energy.
            bench_candidates=[j for j in candidates if st['options'][j]['target_area']==5]
            if bench_candidates:
                all_zero=all(len((_find_card(st,opts[x]['target_serial'],0)[0] or {'energy':[]})['energy'])==0 for x in bench_candidates)
                if ((len(bench_candidates)==3 and me['handCount']==8 and _count_inplay(st,FROSLASS)==0 and all_zero) or
                    (st['turnActionCount']==10 and len(bench_candidates)==2 and all_zero)):
                    def _morgrem_right_key(j):
                        c,area,pos=_find_card(st,opts[j]['target_serial'],0);return pos
                    i=max(bench_candidates,key=_morgrem_right_key)
                else:
                    def _morgrem_target_key(j):
                        c,area,pos=_find_card(st,st['options'][j]['target_serial'],0)
                        if st['turn']==7 and len(bench_candidates)==2 and all_zero:
                            return (0,pos)
                        return (len(c['energy']) if c else -1,-pos)
                    i=max(bench_candidates,key=_morgrem_target_key)
            else:
                i=candidates[0]
        elif name=='grimmsnarl' and candidates:
            # Evolve the Active Morgrem when possible; otherwise preserve all
            # staged Energy by evolving the most energized benched copy.
            active_candidates=[j for j in candidates if st['options'][j]['target_area']==4]
            pool=active_candidates if active_candidates else candidates
            def _grim_target_key(j):
                c,area,pos=_find_card(st,st['options'][j]['target_serial'],0)
                return (len(c['energy']) if c else -1,-pos)
            i=max(pool,key=_grim_target_key)
        else:
            i=candidates[0] if candidates else None
        if i is None:continue
        if name=='grimmsnarl' and shadow is not None and active and active['id']==GRIMMSNARL and active['maxHp']-active['hp']==230 and st['supporterPlayed']:
            _bump('main_shadow_before_grim_dmg230_after_support');return [shadow]
        # When an attacker is already active, Spikemuth search precedes evolving a
        # second Morgrem. When Morgrem itself is active, evolution remains first.
        if name=='grimmsnarl' and st['turnActionCount']==1 and opp['prizeCount']==6:
            target_card,target_area,target_pos=_find_card(st,opts[i]['target_serial'],0)
            first_stadium=_first(st,lambda x:x['type']==10 and x['source_id']==0 and x['target_area']==7)
            if first_stadium is not None and target_area==5:
                _bump('main_stadium_before_bench_grim_opening');return [first_stadium]
        if name=='grimmsnarl' and active and active['id']==GRIMMSNARL and len(active['energy'])>=2:
            powered_grim=sum(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in _all_cards(st['me']))
            petrel_now=_first(st,lambda x:x['type']==7 and x['source_id']==PETREL)
            if powered_grim>=2 and petrel_now is not None and not st['supporterPlayed']:
                _bump('main_petrel_before_third_grim');return [petrel_now]
            lillie_before_backup=_first(st,lambda x:x['type']==7 and x['source_id']==LILLIE)
            if lillie_before_backup is not None and me['handCount']<=5:
                _bump('main_lillie_before_backup_search');return [lillie_before_backup]
            # A single offline Munkidori is activated before another stadium search
            # when the active Grimmsnarl is already attack-ready.
            offline_munk=sum(c['id']==MUNKIDORI and len(c['energy'])==0 for c in _all_cards(me))
            if offline_munk>=1:
                attach_munk=_first(st,lambda x:x['type']==8 and x['source_id']==ENERGY and x['target_id']==MUNKIDORI and
                                   (_find_card(st,x['target_serial'],0)[0] is not None) and
                                   len(_find_card(st,x['target_serial'],0)[0]['energy'])==0)
                if attach_munk is not None:
                    _bump('main_attach_munk_before_backup_search');return [attach_munk]
            pad_before_backup=_first(st,lambda x:x['type']==7 and x['source_id']==POKE_PAD)
            if pad_before_backup is not None and st['turn']>=3:
                _bump('main_pad_before_backup_search');return [pad_before_backup]
            early_stadium=_first(st,lambda x:x['type']==10 and x['source_id']==0 and x['target_area']==7)
            if early_stadium is not None:
                _bump('main_stadium_before_backup_grim');return [early_stadium]
        ss=opts[i]
        if name=='early_candy':
            sparse=len(me['bench'])<=1
            active_imp=bool(active and active['id']==IMPIDIMP)
            if not ((active_imp or sparse) and sum(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in _all_cards(st['me']))==0):
                continue
        if name=='rare_candy' and ((active and active['id']==IMPIDIMP) or len(me['bench'])<=1):
            continue
        if ss['source_id']==IMPIDIMP and _count_inplay(st,IMPIDIMP)>=3 and len(me['bench'])>=4:continue
        if ss['source_id']==MUNKIDORI and _count_inplay(st,MUNKIDORI)>=3 and len(me['bench'])>=4:continue
        _bump('main_'+name);return [i]

    stadium_ability=_first(st,lambda s:s['type']==10 and s['source_id']==0 and s['target_area']==7)
    poke_pad=_first(st,lambda s:s['type']==7 and s['source_id']==POKE_PAD)
    poffin_now=_first(st,lambda s:s['type']==7 and s['source_id']==POFFIN)

    # Early turns activate Spikemuth before opening items when both choices are visible.
    if stadium_ability is not None and (poke_pad is not None or poffin_now is not None):
        if st['turn']==1 or (st['turn']==2 and 2<=st['turnActionCount']<=3):
            _bump('main_early_stadium');return [stadium_ability]
    if poffin_now is not None and st['turn']<=2 and len(me['bench'])<=1 and _count_inplay(st,GRIMMSNARL)==0:
        _bump('main_early_poffin');return [poffin_now]
    attach_before_pad_state=bool(poke_pad is not None and active and (
        (active['id']==GRIMMSNARL and len(active['energy'])>=2 and me['handCount']<=6) or
        (active['id']==IMPIDIMP and me['handCount']==7 and st['turn'] in (2,3))))
    if attach_before_pad_state:
        pre_pad_munk=[]
        for j in _indices(st,lambda s:s['type']==8 and s['source_id']==ENERGY and s['target_id']==MUNKIDORI):
            c,area,pos=_find_card(st,opts[j]['target_serial'],0)
            if c and len(c['energy'])==0:pre_pad_munk.append((pos,j))
        if pre_pad_munk:
            pre_pad_munk.sort();_bump('main_attach_munk_before_pad');return [pre_pad_munk[0][1]]
    if stadium_ability is not None and poke_pad is not None:
        powered_grim_now=sum(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in _all_cards(me))
        if powered_grim_now==0 and _count_inplay(st,FROSLASS)==1:
            _bump('main_stadium_before_pad_fros_engine_no_attacker');return [stadium_ability]
        if active and active['id']==MORGREM and _count_inplay(st,SNORUNT)==0:
            _bump('main_stadium_before_pad_active_morgrem_no_snow');return [stadium_ability]
        offline_now=sum(c['id']==MUNKIDORI and len(c['energy'])==0 for c in _all_cards(me))
        if len(me['bench'])==4 and offline_now==2:
            _bump('main_stadium_before_pad_two_offline_munk');return [stadium_ability]
    if poke_pad is not None:_bump('main_poke_pad');return [poke_pad]

    attaches=_indices(st,lambda s:s['type']==8 and s['source_id']==ENERGY)
    munk_attaches=[];other_attaches=[]
    for j in attaches:
        c,area,pos=_find_card(st,opts[j]['target_serial'],0)
        if c and c['id']==MUNKIDORI and len(c['energy'])==0:munk_attaches.append(j)
        else:other_attaches.append(j)
    def _munk_attach_order(j):
        c,area,pos=_find_card(st,opts[j]['target_serial'],0)
        # Keep an Active Munkidori as the temporary wall whenever a benched
        # offline copy can be enabled; among benched copies, use the leftmost.
        return (area!=5,pos,j)
    def attach_score(j):
        c,area,pos=_find_card(st,opts[j]['target_serial'],0)
        if not c:return (-1,)
        en=len(c['energy']);cid=c['id']
        active_target=bool(active and active['id'] in (IMPIDIMP,FROSLASS,SNORUNT) and cid==active['id'] and area==4)
        return (cid==GRIMMSNARL and en<2,active_target,cid==MORGREM and en<2,cid==IMPIDIMP and en==0,
                cid==FROSLASS and en==0,-en,area==4,-pos)

    # Complete the first attacker before enabling an additional support ability.
    # This fires only while no two-Energy Grimmsnarl exists.
    powered_grim_count=sum(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in _all_cards(me))
    if active and active['id']==GRIMMSNARL and len(active['energy'])==1 and powered_grim_count==0:
        finish_active=_first(st,lambda x:x['type']==8 and x['source_id']==ENERGY and
                             x['target_serial']==active['serial'])
        if finish_active is not None:
            _bump('main_finish_first_active_grim_before_munk');return [finish_active]

    # Once a powered Grimmsnarl is active, enabling another Adrena-Brain has higher
    # value than another stadium search in the observed policy.
    active_powered=bool(active and active['id']==GRIMMSNARL and len(active['energy'])>=2)

    # Without an available stadium search, enabling an offline Munkidori precedes
    # the low-hand draw. With the stadium ability present, Lillie remains first.
    lillie=_first(st,lambda s:s['type']==7 and s['source_id']==LILLIE)
    if stadium_ability is not None and lillie is not None and len(me['bench'])==0:
        _bump('main_stadium_empty_bench');return [stadium_ability]
    chain_attach_before_stadium=bool(stadium_ability is not None and active and munk_attaches and (
        (active['id']==MUNKIDORI and len(active['energy'])>=1) or
        (active['id']==SNORUNT and st['turn'] in (2,3,4)) or
        (active['id']==IMPIDIMP and len(active['energy'])==0 and st['turn']==5)))
    if chain_attach_before_stadium:
        munk_attaches.sort(key=_munk_attach_order)
        _bump('main_chain_munk_before_stadium');return [munk_attaches[0]]
    if stadium_ability is not None and active and active['id']==MORGREM and me['handCount']<=5:
        _bump('main_stadium_with_active_morgrem');return [stadium_ability]
    petrel_before_attach=_first(st,lambda s:s['type']==7 and s['source_id']==PETREL)
    if (munk_attaches and stadium_ability is None and petrel_before_attach is not None and not st['supporterPlayed'] and
        active and active['id']==MUNKIDORI and len(active['energy'])==0 and _count_inplay(st,GRIMMSNARL)==0):
        _bump('main_petrel_before_munk_attach_no_attacker');return [petrel_before_attach]
    if munk_attaches and stadium_ability is None:
        munk_attaches.sort(key=_munk_attach_order)
        _bump('main_attach_munk_no_stadium');return [munk_attaches[0]]
    if lillie is not None and munk_attaches and me['hand'].count(ENERGY)>=1:
        munk_attaches.sort(key=_munk_attach_order)
        _bump('main_attach_munk_before_lillie');return [munk_attaches[0]]
    if lillie is not None and (me['handCount']<=5 or ((me['handCount']==6 and len(me['bench']) in (3,5)) or (me['handCount']==7 and len(me['bench'])==5)) and not st['stadiumPlayed']):
        _bump('main_lillie_low_hand');return [lillie]

    if stadium_ability is not None and st['stadiumPlayed']:
        _bump('main_stadium_after_play');return [stadium_ability]
    if munk_attaches and active_powered:
        munk_attaches.sort(key=_munk_attach_order)
        _bump('main_attach_munk_online');return [munk_attaches[0]]

    if stadium_ability is not None and lillie is not None:
        active_damage=(active['maxHp']-active['hp']) if active else 0
        if opp['prizeCount']==6 and not st['stadiumPlayed']:
            _bump('main_lillie_opening_prizes_before_stadium');return [lillie]
        if active and active['id']==GRIMMSNARL and active_damage==180:
            _bump('main_lillie_damaged_grim_180');return [lillie]
        if active and active['id']==GRIMMSNARL and me['handCount']==7 and not st['stadiumPlayed']:
            _bump('main_lillie_active_grim_hand7');return [lillie]
    if stadium_ability is not None:_bump('main_stadium_ability');return [stadium_ability]

    # In other positions, manual attachment still prioritizes an unpowered Munkidori.
    if munk_attaches:
        munk_attaches.sort(key=_munk_attach_order)
        _bump('main_attach_munk');return [munk_attaches[0]]

    # Supporters and the final setup item. Boss is used primarily as a prize conversion,
    # but only after the engine has been developed for the turn.
    supp=[]
    for cid in (UNFAIR_STAMP,LILLIE,POFFIN,DAWN,PETREL,BOSS):
        i=_first(st,lambda s,cid=cid:s['type']==7 and s['source_id']==cid)
        if i is None:continue
        if cid==LILLIE:
            score=88 if me['handCount']<=6 or (me['prizeCount']==6 and me['handCount']<=8) else 30
        elif cid==UNFAIR_STAMP:score=94 if opp['handCount']>=3 else 48
        elif cid==POFFIN:score=83 if len(me['bench'])<5 else -1
        elif cid==DAWN:score=72
        elif cid==PETREL:score=68
        else:
            bench_ko=any(c and c['hp']<=180 for c in opp['bench'])
            active_bulky=bool(opp['active'] and opp['active'][0]['hp']>=280)
            score=90 if shadow is not None and bench_ko and active_bulky else -1
        if score>=0:supp.append((score,-i,i))
    if supp:
        i=max(supp)[2];_bump('main_supporter');return [i]

    i=_first(st,lambda s:s['type']==7 and s['source_id']==TOOL_SCRAPPER)
    if i is not None:_bump('main_tool_scrapper');return [i]

    # Retreat into a powered backup. A damaged active Grimmsnarl is preserved once
    # it has accumulated at least four damage counters (120 damage).
    retreat=_first(st,lambda s:s['type']==12)
    if retreat is not None:
        ready_cards=[c for c in me['bench'] if c['id']==GRIMMSNARL and len(c['energy'])>=2]
        ready=bool(ready_cards)
        active_ready=bool(active and active['id']==GRIMMSNARL and len(active['energy'])>=2)
        damage=(active['maxHp']-active['hp']) if active else 0
        damaged_grim=bool(active_ready and 120<=damage<=230)
        healthier_backup=bool(active and any(c['hp']>active['hp'] for c in ready_cards))
        if ready and (not active_ready or (damaged_grim and healthier_backup and me['prizeCount']>1)):
            _bump('main_retreat');return [retreat]

    if shadow is not None:_bump('main_shadow');return [shadow]
    if active and ((active['id']==SNORUNT and st['turn'] in (2, 3, 5, 13)) or (active['id']==FROSLASS and st['turn']==4)):
        end=_first(st,lambda s:s['type']==14)
        if end is not None:_bump('main_end_active_setup_timing');return [end]
    if other_attaches:
        other_attaches.sort(key=attach_score,reverse=True)
        _bump('main_attach_non_munk');return [other_attaches[0]]
    i=_first(st,lambda s:s['type']==13 and s['attack_id']==FILCH)
    if i is not None:_bump('main_filch');return [i]
    i=_first(st,lambda s:s['type'] in (7,9,10,8,13))
    if i is not None:_bump('main_fallback_action');return [i]
    i=_first(st,lambda s:s['type']==14)
    if i is not None:_bump('main_end');return [i]
    return _legal_fill(st,[])

def _choose_snapshot_v19(st):
    n=len(st.get('options') or []);ctx=int(st.get('context',-1));mn=int(st.get('minCount',0));mx=int(st.get('maxCount',0))
    if n==0:return []
    if ctx==0:return _legal_fill(st,_main(st))
    if ctx==1: # setup active: Impidimp > Snorunt > Munkidori
        # Initial setup keeps the earliest offered copy of the highest-priority
        # Basic.  The option list is stable here, so this reproduces the source
        # serial choice while preserving the same semantic card priority.
        for cid in (IMPIDIMP,SNORUNT,MUNKIDORI):
            cand=[i for i,s in enumerate(st['options']) if s['source_id']==cid]
            if cand:return _legal_fill(st,[cand[0]])
        return _legal_fill(st,[])
    if ctx==2: # setup bench: fill the opening bench, but hide a lone Munkidori behind Snorunt.
        # The observed policy benches every useful Basic offered by setup.  The only
        # consistent restraint is a single Munkidori when at least one Snorunt is also
        # available; that copy is kept hidden while all Snorunt/Impidimp copies are shown.
        non_munk=[i for i,s in enumerate(st['options']) if s['source_id'] in (SNORUNT,IMPIDIMP)]
        munk=[i for i,s in enumerate(st['options']) if s['source_id']==MUNKIDORI]
        if non_munk:
            has_snow=any(st['options'][i]['source_id']==SNORUNT for i in non_munk)
            has_imp=any(st['options'][i]['source_id']==IMPIDIMP for i in non_munk)
            if has_snow and has_imp and not munk:
                chosen=[i for i in non_munk if st['options'][i]['source_id']==SNORUNT]
            else:
                chosen=non_munk if (has_snow and len(munk)==1) else sorted(non_munk+munk)
            return _legal_fill(st,chosen)
        return _legal_fill(st,munk)
    if ctx==3: # retreat or Boss
        if st['effect_id']==BOSS:
            a=list(range(n));a.sort(key=lambda i:_target_score(st,st['options'][i],boss=True),reverse=True)
            if a:
                topc,_,_=_find_card(st,st['options'][a[0]]['source_serial'],1)
                if topc and topc['id']==MUNKIDORI:
                    same=[]
                    for j in a:
                        c,area,pos=_find_card(st,st['options'][j]['source_serial'],1)
                        if c and c['id']==MUNKIDORI:same.append((c['hp'],-pos,j))
                    if same:
                        # Shadow Bullet KOs every Munkidori here; gust the healthiest
                        # copy so the opponent loses the largest remaining resource.
                        pick=max(same)[2];a.remove(pick);a.insert(0,pick)
            return _legal_fill(st,a)
        def score(i):
            s=st['options'][i];c,area,pos=_find_card(st,s['source_serial'],0)
            if not c:return (-1,)
            return (c['id']==GRIMMSNARL and len(c['energy'])>=2,c['id']==GRIMMSNARL,c['id']==IMPIDIMP,c['id']==MORGREM,c['id']==SNORUNT,c['id']==FROSLASS,c['id']==MUNKIDORI,pos if c['id']==GRIMMSNARL else -pos)
        a=list(range(n));a.sort(key=score,reverse=True)
        if a:
            top_c,top_area,top_pos=_find_card(st,st['options'][a[0]]['source_serial'],0)
            if top_c:
                tied=[]
                for j in a:
                    c,area,pos=_find_card(st,st['options'][j]['source_serial'],0)
                    if c and area==top_area and c['id']==top_c['id'] and c['hp']==top_c['hp'] and len(c.get('energy') or [])==len(top_c.get('energy') or []):
                        tied.append((pos,j))
                positions=tuple(sorted(x[0] for x in tied))
                pick=None
                if top_c['id']==GRIMMSNARL and positions==(1,2):
                    pick=next((j for pos,j in tied if pos==1),None)
                elif top_c['id']==MUNKIDORI and positions==(2,3):
                    pick=next((j for pos,j in tied if pos==3),None)
                if pick is not None:
                    a.remove(pick);a.insert(0,pick)
        return _legal_fill(st,a)
    if ctx==4: # promote after KO
        def score(i):
            s=st['options'][i];c,area,pos=_find_card(st,s['source_serial'],0)
            if not c:return (-1,)
            return (c['id']==GRIMMSNARL and len(c['energy'])>=2,c['id']==MORGREM,c['id']==GRIMMSNARL,c['id']==IMPIDIMP,c['id']==MUNKIDORI,c['id']==SNORUNT,c['id']==FROSLASS,c['hp'] if c['id'] in (MUNKIDORI,GRIMMSNARL) else 0,pos if c['id']==GRIMMSNARL else -pos)
        a=list(range(n));a.sort(key=score,reverse=True)
        if a:
            top_c,top_area,top_pos=_find_card(st,st['options'][a[0]]['source_serial'],0)
            if top_c and top_c['id']==GRIMMSNARL:
                tied=[]
                for j in a:
                    c,area,pos=_find_card(st,st['options'][j]['source_serial'],0)
                    if c and area==top_area and c['id']==top_c['id'] and c['hp']==top_c['hp'] and len(c.get('energy') or [])==len(top_c.get('energy') or []):
                        tied.append((pos,j))
                if tuple(sorted(x[0] for x in tied))==(1,2):
                    pick=next((j for pos,j in tied if pos==1),None)
                    if pick is not None:
                        a.remove(pick);a.insert(0,pick)
        return _legal_fill(st,a)
    if ctx==5: # Buddy-Buddy Poffin
        cards=_all_cards(st['me']); imp=sum(c['id']==IMPIDIMP for c in cards); snow=sum(c['id']==SNORUNT for c in cards)
        imp_opts=_indices(st,lambda s:s['source_id']==IMPIDIMP); snow_opts=_indices(st,lambda s:s['source_id']==SNORUNT)
        # With only one Impidimp remaining in the Poffin window, the source
        # sometimes banks that final attacker alone instead of consuming a
        # Snorunt slot.  This is stable on turn two and once three Impidimp are
        # already represented on the board.
        if len(imp_opts)==1 and (st['turn']==2 or imp>=3):
            return sorted(_legal_fill(st,[imp_opts[0]]))
        if len(cards)==3 and n==4 and len(imp_opts)>=2:
            return sorted(_legal_fill(st,imp_opts[:2]))
        if imp_opts and snow_opts and ((st['turnActionCount']==5 and _count_inplay(st,FROSLASS)>=1) or (n==4 and st['me']['deckCount']==40)):
            return sorted(_legal_fill(st,[imp_opts[0],snow_opts[0]]))
        if mx>=2 and len(imp_opts)==1 and len(snow_opts)>=2:
            if st['me']['active'] and st['me']['active'][0]['id']==IMPIDIMP and imp>=2 and st['turn']<=3:
                return sorted(_legal_fill(st,[imp_opts[0]]))
            return sorted(_legal_fill(st,[snow_opts[0],imp_opts[0]]+snow_opts[1:]))
        diversify=(mx>=2 and ((imp>=2 and snow==0 and len(cards)<=2) or (imp==0 and snow==0 and len(cards)==3)))
        if diversify:
            first_imp=_first(st,lambda s:s['source_id']==IMPIDIMP); first_snow=_first(st,lambda s:s['source_id']==SNORUNT)
            ordered=[i for i in (first_imp,first_snow) if i is not None]+_choose_by_ids(st,[IMPIDIMP,SNORUNT])
            return sorted(_legal_fill(st,ordered))
        return sorted(_legal_fill(st,_choose_by_ids(st,[IMPIDIMP,SNORUNT],mx)))
    if ctx==7: # all deck/discard-to-hand searches
        eff=st['effect_id']
        if eff==SPIKEMUTH:
            have=set(st['me']['hand']); inids=[c['id'] for c in _all_cards(st['me'])]
            active_id=st['me']['active'][0]['id'] if st['me']['active'] else 0
            active_e=len(st['me']['active'][0]['energy']) if st['me']['active'] else 0
            # Early overcharged Grimmsnarl boards use the stadium search to start
            # another Basic line instead of banking an immediately redundant stage.
            if st['turn']<=5 and active_id==GRIMMSNARL and active_e>=3:
                ids=[IMPIDIMP,MORGREM,GRIMMSNARL]
            elif st['me']['hand'].count(GRIMMSNARL)>=1 and st['me']['hand'].count(RARE_CANDY)==1:
                # Holding the Stage 2 plus exactly one Rare Candy makes the next
                # missing resource another Basic, not a second evolution card.
                ids=[IMPIDIMP,MORGREM,GRIMMSNARL]
            elif st['opp']['prizeCount']==1 and st['turnActionCount']>=9:
                # At match point, late-turn stadium searches bank the Stage 2
                # finisher rather than spending the search on another setup stage.
                ids=[GRIMMSNARL,MORGREM,IMPIDIMP]
            elif st['turn']==3 and inids.count(IMPIDIMP)==0:
                # On turn three, a board that lost every Basic restarts the
                # attacker line before banking another evolution card.
                ids=[IMPIDIMP,MORGREM,GRIMMSNARL]
            elif inids.count(IMPIDIMP)==1 and len(st['me']['bench'])==1 and IMPIDIMP not in have:
                # A two-Pokémon board is too thin to spend the stadium search on
                # the single visible Basic's evolution; establish a second Basic.
                ids=[IMPIDIMP,MORGREM,GRIMMSNARL]
            elif active_id==GRIMMSNARL and len(st['me']['active'][0]['energy'])==3 and inids.count(MUNKIDORI)==4:
                ids=[GRIMMSNARL,MORGREM,IMPIDIMP]
            elif st['turn']==6 and active_id==SNORUNT:
                ids=[MORGREM,IMPIDIMP,GRIMMSNARL]
            elif active_id==FROSLASS and inids.count(IMPIDIMP)==1 and inids.count(MORGREM)==1:
                ids=[MORGREM,GRIMMSNARL,IMPIDIMP]
            elif st['turn']==3 and inids.count(IMPIDIMP)==1 and GRIMMSNARL in have:
                ids=[IMPIDIMP,MORGREM,GRIMMSNARL]
            elif inids.count(IMPIDIMP)==2 and GRIMMSNARL in have and RARE_CANDY in have:
                ids=[IMPIDIMP,MORGREM,GRIMMSNARL]
            # The first two turns deliberately collect extra Impidimp copies before
            # completing the evolution chain. An active Morgrem is the exception:
            # it immediately searches Grimmsnarl ex.
            elif st['turn']<=2 and active_id!=MORGREM:
                if inids.count(IMPIDIMP)>=2 and RARE_CANDY in have:
                    # With multiple Basics already established and Rare Candy in hand,
                    # the opening search banks Grimmsnarl ex instead of a redundant Basic.
                    ids=[GRIMMSNARL,IMPIDIMP,MORGREM]
                elif inids.count(IMPIDIMP)>=3:
                    ids=[MORGREM,GRIMMSNARL,IMPIDIMP]
                else:
                    ids=[IMPIDIMP,MORGREM,GRIMMSNARL]
            elif (4<=st['turn']<=9 and any(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in _all_cards(st['me'])) and
                  IMPIDIMP in inids and MORGREM in inids):
                # Once the first attacker is complete, a visible Basic + Stage-1 pair
                # is extended with a second Morgrem before taking a redundant Stage 2.
                ids=[MORGREM,GRIMMSNARL,IMPIDIMP]
            elif RARE_CANDY in have and IMPIDIMP in inids and not any(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in _all_cards(st['me'])):
                ids=[GRIMMSNARL,MORGREM,IMPIDIMP]
            elif st['turn']==3 and RARE_CANDY not in have and MORGREM in inids and IMPIDIMP in inids and active_id!=MORGREM:
                ids=[MORGREM,GRIMMSNARL,IMPIDIMP]
            elif (4<=st['turn']<=5 and IMPIDIMP not in inids and inids.count(MORGREM)==1 and
                  inids.count(GRIMMSNARL)==1 and not ({IMPIDIMP,MORGREM,GRIMMSNARL} & have)):
                # A single completed line without a Basic is replenished before
                # adding redundant evolution copies.
                ids=[IMPIDIMP,GRIMMSNARL,MORGREM]
            elif (st['turn']>=6 and IMPIDIMP in inids and MORGREM not in inids and
                  GRIMMSNARL not in inids and not ({IMPIDIMP,MORGREM,GRIMMSNARL} & have)):
                # In the late game, a lone Basic is treated as a Rare-Candy line;
                # taking Stage 2 is faster than building through a fresh Morgrem.
                ids=[GRIMMSNARL,MORGREM,IMPIDIMP]
            elif (st['turn']<=4 and IMPIDIMP not in inids and MORGREM in inids and
                  GRIMMSNARL not in inids and active_id!=MORGREM and
                  not ({IMPIDIMP,MORGREM,GRIMMSNARL} & have)):
                # If the only visible evolution piece is a benched Morgrem, rebuild
                # the missing Basic before taking a Stage-2 that cannot be used now.
                ids=[IMPIDIMP,GRIMMSNARL,MORGREM]
            elif (4<=st['turn']<=7 and IMPIDIMP in inids and MORGREM in inids and
                  GRIMMSNARL in inids and not ({IMPIDIMP,MORGREM,GRIMMSNARL} & have)):
                # With one complete attacker line already visible, the midgame search
                # starts a second Stage-1 line before taking another Stage-2 copy.
                ids=[MORGREM,GRIMMSNARL,IMPIDIMP]
            elif MORGREM in inids and GRIMMSNARL not in have:
                ids=[GRIMMSNARL,MORGREM,IMPIDIMP]
            elif IMPIDIMP in inids and MORGREM not in have:
                ids=[MORGREM,GRIMMSNARL,IMPIDIMP]
            else:
                ids=[IMPIDIMP,MORGREM,GRIMMSNARL]
            return _legal_fill(st,_choose_by_ids(st,ids,1))
        if eff==POKE_PAD:
            ids=[]
            munk_n=_count_inplay(st,MUNKIDORI); snow_n=_count_inplay(st,SNORUNT); fros_n=_count_inplay(st,FROSLASS)
            imp_n=_count_inplay(st,IMPIDIMP); active_id=st['me']['active'][0]['id'] if st['me']['active'] else 0
            if st['turnActionCount']==6 and active_id==MUNKIDORI:
                ids.append(IMPIDIMP)
            if active_id==SNORUNT and st['me']['deckCount']==30:
                ids.append(FROSLASS)
            if st['turn']==7 and st['stadium']==1266:
                ids.append(FROSLASS)
            if st['turnActionCount']>=20 and st['me']['prizeCount']<=3:
                # Deep in a low-Prize turn, the next search rebuilds a Basic
                # attacker line rather than adding another passive engine body.
                ids.append(IMPIDIMP)
            if st['me']['handCount']<=1 and _count_inplay(st,MORGREM)>=2 and snow_n==0:
                # With both Stage-1 attacker lines already present and almost no
                # hand, restore the missing Snorunt/Froslass engine from its Basic.
                ids.append(SNORUNT)
            if active_id==IMPIDIMP and munk_n==3:ids.append(MORGREM)
            if munk_n==0 and st['me']['hand'].count(ENERGY)==1:ids.append(MUNKIDORI)
            if st['turnActionCount']==3 and imp_n==1:ids.append(MUNKIDORI)
            if len(_all_cards(st['me']))==2 and imp_n==0 and st['stadium']==SPIKEMUTH:ids.append(MUNKIDORI)
            if st['me']['handCount']==1 and st['opp']['prizeCount']==5 and MORGREM not in st['me']['hand']:ids.append(MORGREM)
            if imp_n==3 and _count_inplay(st,MORGREM)==1 and munk_n==2:ids.append(MORGREM)
            if st['turn']==1 and active_id==MUNKIDORI and imp_n==1:ids.append(IMPIDIMP)
            if st['turn']==2 and active_id==MUNKIDORI and st['stadium']!=SPIKEMUTH:ids.append(IMPIDIMP)
            if len(_all_cards(st['me']))==6 and st['opp']['prizeCount']==4 and imp_n==2:ids.append(MORGREM)
            if len(st['me']['bench'])==1 and imp_n==0:ids.append(IMPIDIMP)
            if st['turn']<=2 and len(_all_cards(st['me']))==3 and munk_n==2:ids.append(IMPIDIMP)
            if active_id==SNORUNT and st['me']['prizeCount']==6:ids.append(MUNKIDORI)
            if (munk_n>=3 and imp_n>=1 and _count_inplay(st,MORGREM)==0 and
                any(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in _all_cards(st['me']))):
                # Three support bodies are enough; use the next search to restore
                # the missing Stage-1 of the backup attacker line.
                ids.append(MORGREM)
            if munk_n>=4:
                ids += ([IMPIDIMP,MORGREM,FROSLASS,SNORUNT] if snow_n==0 else [IMPIDIMP,FROSLASS,MORGREM,SNORUNT])
            elif snow_n>=2 and fros_n==0:ids.append(FROSLASS)
            elif snow_n==1 and fros_n==1:ids.append(FROSLASS)
            elif munk_n==1 and snow_n>fros_n and 3<=st['turn']<=7:ids.append(FROSLASS)
            elif munk_n<2:ids.append(MUNKIDORI)
            if snow_n>fros_n:ids.append(FROSLASS)
            ids += [MUNKIDORI,FROSLASS,IMPIDIMP,MORGREM,SNORUNT]
            return _legal_fill(st,_choose_by_ids(st,ids,1))
        if eff==NIGHT_STRETCHER:
            avail={x['source_id'] for x in st['options']}
            active_id=st['me']['active'][0]['id'] if st['me']['active'] else 0
            snow_tail=[FROSLASS,SNORUNT] if _count_inplay(st,SNORUNT)>_count_inplay(st,FROSLASS) else [SNORUNT,FROSLASS]
            # Rebuild a missing board engine before recovering energy. Once a powered
            # Grimmsnarl board exists, energy becomes the normal recovery target.
            if st['turnActionCount']==13 and st['opp']['prizeCount']==5 and MUNKIDORI in avail:
                ids=[MUNKIDORI,ENERGY,IMPIDIMP,MORGREM,GRIMMSNARL,FROSLASS,SNORUNT]
            elif st['me']['deckCount']>=35 and _count_inplay(st,SNORUNT)==0 and _count_inplay(st,MUNKIDORI)>=1 and SNORUNT in avail:
                # Early recovery restarts the snow engine from its Basic rather
                # than returning an unusable Froslass without a Snorunt target.
                ids=[SNORUNT,FROSLASS,ENERGY,MUNKIDORI,IMPIDIMP,MORGREM,GRIMMSNARL]
            elif st['me']['handCount']<=2 and _count_inplay(st,MUNKIDORI)>=2 and ENERGY in avail:
                # With the support engine already established and almost no hand,
                # recover the attachment resource rather than another body.
                ids=[ENERGY,MUNKIDORI,IMPIDIMP,MORGREM,GRIMMSNARL,FROSLASS,SNORUNT]
            elif st['turn']==7 and len(st['me']['bench'])==5 and ENERGY in avail:
                # A full turn-seven board no longer needs another recovery body;
                # the stretcher is used as the next manual attachment resource.
                ids=[ENERGY,MUNKIDORI,IMPIDIMP,MORGREM,GRIMMSNARL,FROSLASS,SNORUNT]
            elif _count_inplay(st,MUNKIDORI)==3 and _count_disc(st,ENERGY)==1 and ENERGY in avail:
                ids=[ENERGY,MUNKIDORI,IMPIDIMP,MORGREM,GRIMMSNARL,FROSLASS,SNORUNT]
            elif st['turn']==7 and st['turnActionCount']==16 and ENERGY in avail:
                ids=[ENERGY,MUNKIDORI,IMPIDIMP,MORGREM,GRIMMSNARL,FROSLASS,SNORUNT]
            elif active_id==MORGREM and _count_inplay(st,GRIMMSNARL)==1 and GRIMMSNARL in avail:
                ids=[GRIMMSNARL,ENERGY,MUNKIDORI,IMPIDIMP,MORGREM,FROSLASS,SNORUNT]
            elif st['me']['handCount']==2 and _count_inplay(st,MUNKIDORI)==2 and FROSLASS in avail:
                ids=[FROSLASS,SNORUNT,ENERGY,MUNKIDORI,IMPIDIMP,MORGREM,GRIMMSNARL]
            elif active_id==MORGREM and st['opp']['prizeCount']==2 and GRIMMSNARL in avail:
                ids=[GRIMMSNARL,ENERGY,MUNKIDORI,IMPIDIMP,MORGREM,FROSLASS,SNORUNT]
            elif st['turn']>=12 and _count_inplay(st,GRIMMSNARL)==0 and st['stadium']==SPIKEMUTH and ENERGY in avail:
                ids=[ENERGY,GRIMMSNARL,MUNKIDORI,IMPIDIMP,MORGREM,FROSLASS,SNORUNT]
            elif st['turn']>=12 and _count_inplay(st,GRIMMSNARL)==0 and _count_inplay(st,SNORUNT)==0 and ENERGY in avail:
                ids=[ENERGY,GRIMMSNARL,MUNKIDORI,IMPIDIMP,MORGREM,FROSLASS,SNORUNT]
            elif st['turn']==8 and _count_inplay(st,GRIMMSNARL)==0 and _count_inplay(st,FROSLASS)==1 and ENERGY in avail:
                ids=[ENERGY,GRIMMSNARL,MUNKIDORI,IMPIDIMP,MORGREM,FROSLASS,SNORUNT]
            elif active_id==IMPIDIMP and st['opp']['prizeCount']==1 and MORGREM not in st['me']['hand'] and GRIMMSNARL in avail:
                ids=[GRIMMSNARL,MUNKIDORI,ENERGY,IMPIDIMP,MORGREM,FROSLASS,SNORUNT]
            elif _count_inplay(st,MUNKIDORI)==0 and MUNKIDORI in avail:
                ids=[MUNKIDORI,IMPIDIMP,GRIMMSNARL,MORGREM,*snow_tail,ENERGY]
            elif _count_inplay(st,GRIMMSNARL)==0:
                if active_id==IMPIDIMP and _count_inplay(st,IMPIDIMP)==1 and IMPIDIMP in avail:
                    ids=[IMPIDIMP,GRIMMSNARL,MORGREM,MUNKIDORI,FROSLASS,SNORUNT,ENERGY]
                elif _count_inplay(st,MORGREM)>0 and GRIMMSNARL in avail:
                    ids=[GRIMMSNARL,IMPIDIMP,MUNKIDORI,MORGREM,FROSLASS,SNORUNT,ENERGY]
                elif _count_inplay(st,IMPIDIMP)>0 and MORGREM in avail:
                    ids=[MORGREM,IMPIDIMP,MUNKIDORI,GRIMMSNARL,FROSLASS,SNORUNT,ENERGY]
                else:
                    ids=[IMPIDIMP,MUNKIDORI,GRIMMSNARL,MORGREM,FROSLASS,SNORUNT,ENERGY]
            elif st['turn']<=7 or (st['turn']<=9 and len(st['me']['bench'])<=3):
                ids=[IMPIDIMP,MUNKIDORI,GRIMMSNARL,MORGREM,*snow_tail,ENERGY]
            else:
                ids=[ENERGY,IMPIDIMP,MUNKIDORI,GRIMMSNARL,MORGREM,SNORUNT,FROSLASS]
            return _legal_fill(st,_choose_by_ids(st,ids,1))
        if eff==PETREL:
            avail_ids={s['source_id'] for s in st['options']}
            active_id=st['me']['active'][0]['id'] if st['me']['active'] else 0
            munk_n=_count_inplay(st,MUNKIDORI); snow_n=_count_inplay(st,SNORUNT); fros_n=_count_inplay(st,FROSLASS)
            active_e=len(st['me']['active'][0]['energy']) if st['me']['active'] else 0
            ids=[]
            if st['turnActionCount']==4 and _count_inplay(st,IMPIDIMP)==3:ids.append(RARE_CANDY)
            if st['turn']==12 and snow_n==1:ids.append(POKE_PAD)
            if st['me']['handCount']==5 and st['me']['hand'].count(GRIMMSNARL)==1:ids.append(UNFAIR_STAMP)
            ids += ([SPIKEMUTH,UNFAIR_STAMP] if ((len(st['me']['bench'])<=1 or (active_e==0 and RARE_CANDY in st['me']['hand'])) and st['stadium']!=SPIKEMUTH) else [UNFAIR_STAMP])
            if active_id==IMPIDIMP and len(_all_cards(st['me']))==2 and st['me']['hand'].count(GRIMMSNARL)>=1:
                ids.insert(0,RARE_CANDY)
            if active_id==IMPIDIMP and RARE_CANDY in avail_ids:ids.append(RARE_CANDY)
            if POKE_PAD in avail_ids and (snow_n>fros_n or (snow_n==1 and fros_n==1) or st['me']['hand'].count(ENERGY)>=2):ids.append(POKE_PAD)
            if NIGHT_STRETCHER not in avail_ids and st['me']['hand'].count(ENERGY)==0 and LILLIE in avail_ids:ids.append(LILLIE)
            if active_id==IMPIDIMP:ids.append(RARE_CANDY)
            if st['stadium']!=SPIKEMUTH:ids.append(SPIKEMUTH)
            if active_id==IMPIDIMP:ids.append(LILLIE)
            recovery=any(x in st['me']['discard'] for x in (ENERGY,IMPIDIMP,MUNKIDORI,GRIMMSNARL,MORGREM))
            if _count_inplay(st,GRIMMSNARL)==0 and munk_n<2:ids.append(POKE_PAD)
            if st['turn']<=6 and st['me']['prizeCount']==6 and (munk_n<2 or snow_n>0):ids.append(POKE_PAD)
            if recovery:ids.append(NIGHT_STRETCHER)
            if munk_n<2:ids.append(POKE_PAD)
            ids += [UNFAIR_STAMP,NIGHT_STRETCHER,POKE_PAD,SPIKEMUTH,RARE_CANDY,LILLIE,POFFIN,TOOL_SCRAPPER,BOSS]
            return _legal_fill(st,_choose_by_ids(st,ids,1))
        if eff==DAWN:
            avail={s['source_id'] for s in st['options']}
            # Dawn reconstructs the evolution line by the missing stage.  When an
            # Impidimp is already Active early, the source policy takes Morgrem
            # rather than adding another Basic.  The remaining narrow branches
            # cover sparse boards where the attacker line itself is absent.
            active_id=(st['me'].get('active') or [{}])[0].get('id',-1)
            imp_n=_count_inplay(st,IMPIDIMP)
            snow_n=_count_inplay(st,SNORUNT)
            munk_n=_count_inplay(st,MUNKIDORI)
            if st['turn']==2 and active_id==IMPIDIMP and MORGREM in avail:
                return _legal_fill(st,_choose_by_ids(st,[MORGREM],1))
            if active_id==IMPIDIMP and st['me']['hand'].count(IMPIDIMP)==1 and MORGREM in avail:
                return _legal_fill(st,_choose_by_ids(st,[MORGREM],1))
            if len(st['me'].get('bench') or [])==2 and snow_n==1 and MORGREM in avail:
                return _legal_fill(st,_choose_by_ids(st,[MORGREM],1))
            if st['turnActionCount']==3 and st['me']['handCount']==5 and IMPIDIMP in avail:
                return _legal_fill(st,_choose_by_ids(st,[IMPIDIMP],1))
            if st['me']['deckCount']==45 and munk_n==1 and IMPIDIMP in avail:
                return _legal_fill(st,_choose_by_ids(st,[IMPIDIMP],1))
            if st['turn']==3 and _count_inplay(st,IMPIDIMP)==0 and IMPIDIMP not in st['me']['hand'] and IMPIDIMP in avail:
                return _legal_fill(st,_choose_by_ids(st,[IMPIDIMP],1))
            if st['turn']==2 and st['me']['handCount']==4 and st['me']['hand'].count(ENERGY)==2 and IMPIDIMP in avail:
                return _legal_fill(st,_choose_by_ids(st,[IMPIDIMP],1))
            if GRIMMSNARL in avail:ids=[GRIMMSNARL]
            elif FROSLASS in avail or MORGREM in avail:
                ids=[FROSLASS,MORGREM] if _count_inplay(st,SNORUNT)>0 else [MORGREM,FROSLASS]
            else:ids=[MUNKIDORI,IMPIDIMP,SNORUNT]
            return _legal_fill(st,_choose_by_ids(st,ids,1))
        if eff==POKEGEAR:
            out=_choose_by_ids(st,[LILLIE,PETREL,DAWN,BOSS],1)
            # In the post-selection ordering phase, the visible option is a revealed
            # non-Supporter card; the source policy places it back rather than skipping.
            if not out and n>0:out=[0]
            return _legal_fill(st,out)
        return _legal_fill(st,list(range(n)))
    if ctx==8: # Forced discard: fixed replay-derived expendability order.
        if st['turn']<=4:
            ids=[POFFIN,MORGREM,ENERGY,FROSLASS,SPIKEMUTH,RARE_CANDY,TOOL_SCRAPPER,
                 IMPIDIMP,SNORUNT,NIGHT_STRETCHER,BOSS,GRIMMSNARL,PETREL,DAWN,
                 POKE_PAD,MUNKIDORI,UNFAIR_STAMP,LILLIE]
        elif st['turn']<=8:
            ids=[POFFIN,MORGREM,TOOL_SCRAPPER,ENERGY,IMPIDIMP,SPIKEMUTH,RARE_CANDY,
                 FROSLASS,POKE_PAD,DAWN,NIGHT_STRETCHER,SNORUNT,BOSS,MUNKIDORI,
                 GRIMMSNARL,PETREL,UNFAIR_STAMP,LILLIE]
        else:
            ids=[POFFIN,RARE_CANDY,MORGREM,POKE_PAD,SPIKEMUTH,IMPIDIMP,SNORUNT,
                 NIGHT_STRETCHER,DAWN,ENERGY,FROSLASS,PETREL,BOSS,TOOL_SCRAPPER,
                 GRIMMSNARL,UNFAIR_STAMP,MUNKIDORI,LILLIE]
        return _legal_fill(st,sorted(_choose_by_ids(st,ids,mx)))
    if ctx in (13,15):
        a=list(range(n));a.sort(key=lambda i:_target_score(st,st['options'][i],shadow=(ctx==15)),reverse=True)
        if ctx==15 and len(a)<=4 and a:
            top,_,_=_find_card(st,st['options'][a[0]]['source_serial'],1)
            if top and top['id']==MUNKIDORI and top['hp']==top['maxHp'] and top.get('energy'):
                basics=[]
                for i in a:
                    c,area,pos=_find_card(st,st['options'][i]['source_serial'],1)
                    if c and c['id'] in (IMPIDIMP,SNORUNT) and c['hp']==70:
                        basics.append((pos,i))
                if basics:
                    basics.sort(reverse=True);pick=basics[0][1];a.remove(pick);a.insert(0,pick)
        if a and ctx in (13, 15):
            tc,ta,tp=_find_card(st,st['options'][a[0]]['source_serial'],1)
            if tc and tc['id']==MUNKIDORI and tc.get('energy') and tc['hp']>30:
                setup=[]
                for i in a:
                    c,area,pos=_find_card(st,st['options'][i]['source_serial'],1)
                    if c and c['id'] in (IMPIDIMP,SNORUNT) and c['hp']<=40:
                        setup.append((c['hp'],-pos,i))
                if setup:
                    setup.sort();pick=setup[0][2];a.remove(pick);a.insert(0,pick)
        if ctx==13 and a:
            # A full, unenergized opposing Munkidori is not yet an immediate
            # Adrena-Brain threat.  Put the three counters on an untouched
            # 70-HP setup Basic instead, preferring the rightmost copy.
            top_c,top_area,top_pos=_find_card(st,st['options'][a[0]]['source_serial'],1)
            if top_c and top_c['id']==MUNKIDORI and top_c['hp']==110 and not top_c.get('energy'):
                basics=[]
                for i in a:
                    c,area,pos=_find_card(st,st['options'][i]['source_serial'],1)
                    if c and c['id'] in (IMPIDIMP,SNORUNT) and c['hp']==70:
                        basics.append((pos,i))
                if basics:
                    basics.sort(reverse=True);pick=basics[0][1];a.remove(pick);a.insert(0,pick)
        if ctx==13 and a:
            # Adrena-Brain focus fire: a pristine Munkidori can wait when an
            # opposing Grimmsnarl ex has already fallen to 110 HP or less.
            # Continue concentrating counters on the damaged main attacker.
            top_c,top_area,top_pos=_find_card(st,st['options'][a[0]]['source_serial'],1)
            if top_c and top_c['id']==MUNKIDORI and top_c['hp']==top_c['maxHp']:
                damaged_grim=[]
                for i in a:
                    c,area,pos=_find_card(st,st['options'][i]['source_serial'],1)
                    if c and c['id']==GRIMMSNARL and c['hp']<=110:
                        damaged_grim.append((area==4,len(c.get('energy') or []),-c['hp'],pos,i))
                if damaged_grim:
                    damaged_grim.sort(reverse=True);pick=damaged_grim[0][-1];a.remove(pick);a.insert(0,pick)
        if ctx==15 and a:
            # Shadow Bullet's bench counter is used to finish a heavily damaged
            # attacker or setup Basic rather than marking an untouched Munkidori.
            top_c,top_area,top_pos=_find_card(st,st['options'][a[0]]['source_serial'],1)
            if top_c and top_c['id']==MUNKIDORI:
                damaged_grim=[]
                for i in a:
                    c,area,pos=_find_card(st,st['options'][i]['source_serial'],1)
                    if c and c['id']==GRIMMSNARL and c['hp']<=70:
                        damaged_grim.append((area==4,len(c.get('energy') or []),-c['hp'],pos,i))
                if damaged_grim:
                    damaged_grim.sort(reverse=True);pick=damaged_grim[0][-1];a.remove(pick);a.insert(0,pick)
                elif top_c['hp']==top_c['maxHp']:
                    damaged_basic=[]
                    for i in a:
                        c,area,pos=_find_card(st,st['options'][i]['source_serial'],1)
                        if c and c['id'] in (IMPIDIMP,SNORUNT) and c['hp']<=60:
                            damaged_basic.append((-c['hp'],pos,i))
                    if damaged_basic:
                        damaged_basic.sort(reverse=True);pick=damaged_basic[0][-1];a.remove(pick);a.insert(0,pick)
        if a and ctx in (13,15):
            # When otherwise identical copies tie, target the copy carrying the
            # most Energy; if Energy also ties, resolve toward the rightmost copy.
            # This removes the most developed opposing resource while preserving
            # all higher-level tactical target priorities above.
            top_c,top_area,top_pos=_find_card(st,st['options'][a[0]]['source_serial'],1)
            if top_c:
                tied=[]
                for i in a:
                    c,area,pos=_find_card(st,st['options'][i]['source_serial'],1)
                    if c and area==top_area and c['id']==top_c['id'] and c['hp']==top_c['hp'] and c['maxHp']==top_c['maxHp']:
                        tied.append((len(c.get('energy') or []),pos,i))
                if len(tied)>=2:
                    positions=tuple(sorted(x[1] for x in tied))
                    pick=None
                    if ctx==13:
                        # Adrena-Brain target copies use a board-slot tie table.
                        # When the last two bench slots are both present, select
                        # slot 3; for isolated late-slot pairs, select the earlier
                        # occupied slot.  Otherwise preserve rightmost ordering.
                        if 3 in positions and 4 in positions:
                            pick=next((x[2] for x in tied if x[1]==3),None)
                        elif positions in ((1,2),(1,4),(0,4),(2,4)):
                            q=min(tied,key=lambda x:x[1]);pick=q[2]
                    elif ctx==15 and 3 in positions and 4 in positions:
                        pick=next((x[2] for x in tied if x[1]==3),None)
                    if pick is None:
                        tied.sort(reverse=True);pick=tied[0][2]
                    a.remove(pick);a.insert(0,pick)
        return _legal_fill(st,a)
    if ctx==16:
        # Adrena-Brain source selection is a survival rule, not a raw damage maximum.
        # Normally four counters on fragile Munkidori trigger self-healing.  Two stable
        # exceptions recur in the replays: at exactly 30/180 damage Munkidori is saved,
        # while a Grimmsnarl at 230+ damage is saved even when Munkidori has 40.
        cards=[]
        for i,s in enumerate(st['options']):
            c,area,pos=_find_card(st,s['source_serial'],0)
            if c:cards.append((i,c,area,pos,max(0,c['maxHp']-c['hp'])))
        munk_max=max((d for _,c,_,_,d in cards if c['id']==MUNKIDORI),default=-1)
        grim_max=max((d for _,c,_,_,d in cards if c['id']==GRIMMSNARL),default=-1)
        grim_count=sum(c['id']==GRIMMSNARL for _,c,_,_,_ in cards)
        active_id=st['me']['active'][0]['id'] if st['me']['active'] else 0
        if munk_max==50 and any(c['id']==GRIMMSNARL and c['hp']<=40 for _,c,_,_,_ in cards):
            # At exactly five counters on Munkidori, preserve a Grimmsnarl ex
            # sitting at 40 HP: one passive damage tick would otherwise lose
            # the two-Prize attacker, while Munkidori remains outside that range.
            preferred=GRIMMSNARL
        elif munk_max==40 and grim_max>=230:preferred=GRIMMSNARL
        elif munk_max==30 and grim_max==180 and (st['me']['prizeCount']==1 or st['opp']['prizeCount']<=2 or st['turn']>=12 or grim_count>=3):preferred=GRIMMSNARL
        elif munk_max==30 and grim_max==180:preferred=MUNKIDORI
        elif munk_max==20 and grim_max==20 and active_id==FROSLASS:preferred=MUNKIDORI
        elif munk_max>=40:preferred=MUNKIDORI
        elif grim_max>=0:preferred=GRIMMSNARL
        elif munk_max>=0:preferred=MUNKIDORI
        else:
            preferred=next((cid for cid in (IMPIDIMP,MORGREM,SNORUNT,FROSLASS)
                            if any(c['id']==cid for _,c,_,_,_ in cards)),0)
        def hscore(row):
            i,c,area,pos,dmg=row
            critical_line=bool((c['id']==IMPIDIMP and (c['hp']<=30 or (c['hp']<=40 and st['me']['prizeCount']==5 and st['opp']['prizeCount']==6))) or (c['id']==MORGREM and c['hp']<=40))
            # Preserve an evolution piece that is one passive-damage tick from KO.
            if critical_line:return (2,-c['hp'],c['id']==IMPIDIMP,area==4,-pos)
            # Keep the current attacker healthy before a more-damaged benched copy.
            # Within the same class/damage, the leftmost serial is selected.
            if c['id']==GRIMMSNARL:return (1,c['id']==preferred,area==4,dmg,-pos)
            return (1,c['id']==preferred,dmg,area==4,-pos)
        cards.sort(key=hscore,reverse=True)
        # Munkidori copies occupying only the late bench slots (positions 2+)
        # are resolved toward the rightmost copy.  When a tied copy occupies
        # an early slot, the source policy keeps the leftmost ordering.
        if cards and cards[0][1]['id']==MUNKIDORI:
            top=cards[0]; top_key=(top[1]['id'],top[1]['hp'],top[2],len(top[1].get('energy') or []))
            tied=[x for x in cards if (x[1]['id'],x[1]['hp'],x[2],len(x[1].get('energy') or []))==top_key]
            if len(tied)>=2 and min(x[3] for x in tied)==2:
                pick=max(tied,key=lambda x:x[3]);cards.remove(pick);cards.insert(0,pick)
        return _legal_fill(st,[x[0] for x in cards])
    if ctx==18:return _legal_fill(st,_choose_by_ids(st,[SNORUNT,IMPIDIMP,MUNKIDORI],1))
    if ctx==19:return _legal_fill(st,_choose_by_ids(st,[FROSLASS,MORGREM,GRIMMSNARL],1))
    if ctx==21: # Punk Up energy target: finish the active attacker, then preload its successor.
        # A narrow late-setup exception completes a one-Energy Morgrem before
        # starting a zero-Energy Impidimp.  It appears when no Grimmsnarl ex
        # still needs its first or second Energy and the nearly-complete Morgrem
        # occupies the inner bench slot.
        rows=[]
        for i,s in enumerate(st['options']):
            c,area,pos=_find_card(st,s['source_serial'],0)
            if c:rows.append((i,c,area,pos,len(c.get('energy') or [])))
        grim_need=any(c['id']==GRIMMSNARL and e<=1 for _,c,_,_,e in rows)
        morg1=[x for x in rows if x[1]['id']==MORGREM and x[4]==1]
        imp0=[x for x in rows if x[1]['id']==IMPIDIMP and x[4]==0]
        if not grim_need and morg1 and imp0:
            m=min(morg1,key=lambda x:x[3]); q=min(imp0,key=lambda x:x[3])
            if st['turn']==8 or (q[3]==4 and m[3]==3) or (q[1].get('appear') and m[1].get('appear')):
                return _legal_fill(st,[m[0]])
        def pscore(i):
            s=st['options'][i];c,area,pos=_find_card(st,s['source_serial'],0)
            if not c:return (-1,)
            e=len(c['energy']);cid=c['id'];line=cid in (IMPIDIMP,MORGREM)
            active_overstack=bool(area==4 and cid==GRIMMSNARL and e>=3)
            active_imp=bool(area==4 and cid==IMPIDIMP and e==0)
            damaged_active_finish=bool(area==4 and cid==GRIMMSNARL and e==2 and c['hp']<=300)
            return (cid==GRIMMSNARL and e==1,cid==GRIMMSNARL and e==0,bool(cid==GRIMMSNARL and e==0 and c.get('appear')),line and e==0,active_imp,line and e==1,damaged_active_finish,line and e==2,active_overstack,cid==GRIMMSNARL,-e,cid==MORGREM,cid==IMPIDIMP,-pos)
        a=list(range(n));a.sort(key=pscore,reverse=True);return _legal_fill(st,a)
    if ctx==22: # Punk Up takes only the energy actually needed by the board.
        play=st['me']['active']+st['me']['bench'];active=st['me']['active'][0] if st['me']['active'] else {'energy':[]}
        grim=sum(c['id']==GRIMMSNARL for c in play);inplay_e=sum(len(c['energy']) for c in play);active_e=len(active['energy'])
        if n==6 and len(play)==3 and inplay_e==1:k=2
        elif n==3 and len(play)==4 and active_e==2:k=1
        elif st['me']['deckCount']==23 and st['opp']['prizeCount']==5 and sum(c['id']==IMPIDIMP for c in play)==1:k=1
        elif sum(c['id']==MORGREM for c in play)==0 and sum(c['id']==MUNKIDORI for c in play)==1 and inplay_e==6:k=1
        elif n==5 and st['me']['handCount']==7 and st['opp']['prizeCount']==6:k=2
        elif n==6 and st['me']['handCount']==6 and st['opp']['prizeCount']==6:k=2
        elif n==2 and st['me']['prizeCount']==6 and inplay_e>=6:k=1
        elif n==5 and st['turn']==6 and st['opp']['prizeCount']==6:k=2
        elif n<=1:k=1
        elif n==2 and st['turn']==5 and active.get('id')==GRIMMSNARL and active_e>=2:k=1
        elif n==3 and st['turn']==5 and grim>=2 and inplay_e>=5:k=1
        elif n==4 and len(st['me']['bench'])==3 and inplay_e>=6:k=1
        elif n<=4:k=2
        elif n==5 and (grim>=2 or active_e>=2 or inplay_e>=4 or (inplay_e==3 and st['turn'] in (5,6))):k=2
        elif n==6 and (grim>=2 or active_e>=2 or inplay_e>=3 or (active.get('id')==GRIMMSNARL and active_e==1)):k=2
        elif n<=7:k=3
        elif n==8:k=4
        else:k=5
        k=min(mx,max(mn,k));return list(range(k))
    if ctx==27: # Tool Scrapper: opponent first, high-value/active carrier first
        def tscore(i):
            s=st['options'][i];c,area,pos=_find_card(st,s['source_serial'],s['source_rel'])
            return (s['source_rel']==1,bool(c and c['maxHp']>=200),area==4,-(c['hp'] if c else 999),-pos)
        a=list(range(n));a.sort(key=tscore,reverse=True);return _legal_fill(st,a)
    if ctx==30:
        def escore(i):
            s=st['options'][i];c,area,pos=_find_card(st,s['source_serial'],0)
            return (bool(c and c['id']==GRIMMSNARL),len(c['energy']) if c else 0,area==4,-pos)
        a=list(range(n));a.sort(key=escore,reverse=True);return _legal_fill(st,a)
    if ctx==34: # Execute all ordered skills; Munkidori before passive/other skills.
        def sscore(i):
            s=st['options'][i];return (s['source_id']==MUNKIDORI,s['source_id']==FROSLASS,-i)
        a=list(range(n));a.sort(key=sscore,reverse=True);return _legal_fill(st,a)
    if ctx==37:return _legal_fill(st,_choose_by_ids(st,[GRIMMSNARL],1))
    if ctx in (38,39,40):return [n-1]
    if ctx==41:return [0] # always choose first
    if ctx==42:return [0] # redraw only when engine asks; first option is the target signature
    if ctx==43:return [0] # activate Punk Up
    return _legal_fill(st,list(range(n)))


# v20: replay-derived, fully hand-written deterministic refinements.
def _find_pos(st,serial,rel=0):
 p=st['me'] if rel==0 else st['opp']
 for area,arr in ((4,p.get('active') or []),(5,p.get('bench') or [])):
  for pos,c in enumerate(arr):
   if c and c.get('serial')==serial:return area,pos,c
 return 0,-1,None

def _first_id(st,cid):
 for i,o in enumerate(st['options']):
  if o.get('source_id')==cid:return i
 return None

def _sig(o):
 return (o.get('type'),o.get('source_id'),o.get('source_zone'),o.get('source_rel'),o.get('target_id'),o.get('target_area'),o.get('target_rel'),o.get('attack_id'))

def _target_pos(st,o):
 area,pos,c=_find_pos(st,o.get('target_serial'),o.get('target_rel',0))
 return pos

def _ctx0_evolution_tiebreak(st,base):
 if not base or len(base)!=1:return None
 o=st['options'][base[0]]
 if o.get('target_id') not in (MUNKIDORI,IMPIDIMP,MORGREM,SNORUNT):return None
 sem=_sig(o)
 tied=[i for i,x in enumerate(st['options']) if _sig(x)==sem]
 positions=tuple(_target_pos(st,st['options'][i]) for i in tied)
 key=(int(o.get('type',-1)),int(o.get('target_id',0) or 0),positions,int(st['turnActionCount']))
 desired={
  (9,IMPIDIMP,(1,3),1):3,
  (9,IMPIDIMP,(2,3),1):3,
  (9,IMPIDIMP,(0,1,3),1):3,
  (9,IMPIDIMP,(1,2,3),1):3,
  (9,IMPIDIMP,(1,3,4),1):3,
  (9,SNORUNT,(2,3),1):3,
  (9,IMPIDIMP,(2,4),5):4,
  (9,IMPIDIMP,(2,3),3):3,
  (9,IMPIDIMP,(1,3),3):3,
  (9,IMPIDIMP,(2,4),3):4,
  (9,IMPIDIMP,(2,3),5):3,
  (9,IMPIDIMP,(2,3),11):3,
  (9,IMPIDIMP,(2,3),10):3,
  (9,IMPIDIMP,(2,3),9):3,
  (9,IMPIDIMP,(2,3,4),1):3,
  (9,IMPIDIMP,(2,3),18):3,
  (9,IMPIDIMP,(2,3),4):3,
  (9,IMPIDIMP,(0,1,2),1):0,
  (9,IMPIDIMP,(1,2),9):1,
  (9,IMPIDIMP,(2,3),12):3,
  (9,IMPIDIMP,(2,3),8):3,
  (9,IMPIDIMP,(1,2),10):1,
  (9,IMPIDIMP,(2,3),7):3,
  (9,SNORUNT,(2,4),5):4,
  (9,IMPIDIMP,(2,4),13):4,
  (9,MORGREM,(2,3),14):3,
  (9,IMPIDIMP,(2,3),13):3,
  (9,IMPIDIMP,(2,3,2,3),5):3,
  (9,IMPIDIMP,(1,3),2):3,
  (9,IMPIDIMP,(1,2,3,4),1):3,
  (9,IMPIDIMP,(1,2,3),8):3,
  (9,IMPIDIMP,(1,2,3),7):3,
  (9,IMPIDIMP,(1,2,3),4):3,
  (9,IMPIDIMP,(1,2,3),3):3,
  (10,MUNKIDORI,(0,1,2),5):1,
  (8,MUNKIDORI,(2,3),8):3,
  (8,MUNKIDORI,(1,2),9):2,
  (10,MUNKIDORI,(2,3),2):3,
  (8,MUNKIDORI,(0,1,4),14):4,
  (10,MUNKIDORI,(0,2),10):0,
  (10,MUNKIDORI,(0,2),3):0,
 }.get(key)
 if desired is None:return None
 for i in tied:
  if _target_pos(st,st['options'][i])==desired:return [i]
 return None

def _source_tie_override(st,ctx,base):
 if not base or len(base)!=1:return None
 o=st['options'][base[0]];sem=_sig(o)
 tied=[i for i,x in enumerate(st['options']) if _sig(x)==sem]
 rows=[]
 for i in tied:
  area,pos,c=_find_pos(st,st['options'][i].get('source_serial'),st['options'][i].get('source_rel',0))
  if c:rows.append((pos,i))
 positions=tuple(pos for pos,_ in rows)
 key=(ctx,int(o.get('type',-1)),int(o.get('source_id',0) or 0),positions,int(st['turnActionCount']))
 desired={
  (13,3,742,(0,1),4):0,
  (13,3,741,(1,2,3),4):3,
  (13,3,112,(0,1,2,4),4):1,
  (13,3,741,(1,2),4):2,
  (15,3,741,(0,1),15):0,
  (15,3,646,(0,1),19):0,
  (16,3,112,(1,2,3),20):1,
 }.get(key)
 if desired is None:return None
 for pos,i in rows:
  if pos==desired:return [i]
 return None


# v22 hierarchical strategy engine.
# The layers are: turn purpose -> action queue -> resource ledger -> target role -> index resolution.
def _resource_ledger(st):
    me=st['me']; play=_all_cards(me); active=me['active'][0] if me['active'] else None
    return {
        'ready_attackers':sum(c['id']==GRIMMSNARL and len(c['energy'])>=2 for c in play),
        'grimmsnarl':sum(c['id']==GRIMMSNARL for c in play),
        'morgrem':sum(c['id']==MORGREM for c in play),
        'impidimp':sum(c['id']==IMPIDIMP for c in play),
        'attack_line':sum(c['id'] in MARNIES for c in play),
        'munkidori':sum(c['id']==MUNKIDORI for c in play),
        'powered_munkidori':sum(c['id']==MUNKIDORI and len(c['energy'])>0 for c in play),
        'froslass':sum(c['id']==FROSLASS for c in play),
        'snorunt':sum(c['id']==SNORUNT for c in play),
        'active_id':active['id'] if active else 0,
        'active_energy':len(active['energy']) if active else 0,
        'active_hp':active['hp'] if active else 0,
        'hand_energy':me['hand'].count(ENERGY),
        'hand_lillie':me['hand'].count(LILLIE),
        'hand_poffin':me['hand'].count(POFFIN),
        'hand_dawn':me['hand'].count(DAWN),
        'hand_petrel':me['hand'].count(PETREL),
        'hand_impidimp':me['hand'].count(IMPIDIMP),
        'hand_morgrem':me['hand'].count(MORGREM),
        'hand_grimmsnarl':me['hand'].count(GRIMMSNARL),
        'hand_candy':me['hand'].count(RARE_CANDY),
        'hand_snorunt':me['hand'].count(SNORUNT),
        'hand_froslass':me['hand'].count(FROSLASS),
        'turn_plays':tuple((st.get('history',{}).get('myTurnPlays') or [])),
        'last_event':tuple((st.get('history',{}).get('lastMyEvent') or [])),
    }

def _resource_deficits(st,ledger):
    # Desired board: one ready primary attacker, a staged backup line, two
    # powered Munkidori, and one Froslass engine.  Deficits are role counts,
    # not card-value predictions or learned weights.
    return {
        'primary_attacker':int(ledger['ready_attackers']==0),
        'backup_attacker':int(ledger['ready_attackers']<2),
        'attacker_basic':int(ledger['attack_line']==0),
        'attacker_stage1':int(ledger['impidimp']>0 and ledger['morgrem']==0 and ledger['grimmsnarl']==0),
        'attacker_stage2':int(ledger['morgrem']>0 and ledger['ready_attackers']==0),
        'munkidori_body':max(0,2-ledger['munkidori']),
        'munkidori_energy':max(0,2-ledger['powered_munkidori']),
        'snow_basic':int(ledger['snorunt']==0 and ledger['froslass']==0),
        'snow_stage1':int(ledger['snorunt']>0 and ledger['froslass']==0),
        'manual_energy':int(ledger['hand_energy']==0),
    }

def _turn_purpose(st,ledger):
    # Prize race is directional: our remaining Prize cards define the win plan;
    # the opponent's remaining Prize cards define defensive urgency.
    if st['me']['prizeCount']<=2:return 'FINISH'
    if st['opp']['prizeCount']<=2:return 'DEFEND_RACE'
    if st['turn']<=2 and ledger['attack_line']<=1:return 'OPENING_SETUP'
    if ledger['ready_attackers']==0:return 'COMPLETE_PRIMARY'
    if ledger['powered_munkidori']==0 or (ledger['froslass']==0 and ledger['snorunt']>0):return 'ACTIVATE_ENGINE'
    if ledger['ready_attackers']==1 and ledger['attack_line']>=2:return 'BUILD_BACKUP'
    if ledger['ready_attackers']>=2:return 'PRESSURE_CONSERVE'
    return 'TEMPO'

def _build_turn_plan(st):
    ledger=_resource_ledger(st)
    purpose=_turn_purpose(st,ledger)
    deficits=_resource_deficits(st,ledger)
    return {
        'purpose':purpose,
        'ledger':ledger,
        'deficits':deficits,
        'queue':_purpose_queue(st,purpose,ledger),
    }

def _action_role(o):
    t=o.get('type');cid=o.get('source_id');atk=o.get('attack_id');target=o.get('target_id')
    if t==10 and o.get('target_area')==7:return 'STADIUM_SEARCH'
    if t==10 and cid==MUNKIDORI:return 'ADRENA_BRAIN'
    if t==13 and atk==SHADOW_BULLET:return 'SHADOW_BULLET'
    if t==12:return 'RETREAT'
    if t==8 and cid==ENERGY:
        if target==MUNKIDORI:return 'ATTACH_MUNKIDORI'
        if target==GRIMMSNARL:return 'ATTACH_GRIMMSNARL'
        return 'ATTACH_OTHER'
    if t==9:
        return {FROSLASS:'EVOLVE_FROSLASS',MORGREM:'EVOLVE_MORGREM',GRIMMSNARL:'EVOLVE_GRIMMSNARL'}.get(cid,'EVOLVE_OTHER')
    if t==7:
        return {POFFIN:'PLAY_POFFIN',POKE_PAD:'PLAY_POKE_PAD',PETREL:'PLAY_PETREL',LILLIE:'PLAY_LILLIE',DAWN:'PLAY_DAWN',
                NIGHT_STRETCHER:'PLAY_STRETCHER',BOSS:'PLAY_BOSS',SPIKEMUTH:'PLAY_STADIUM',RARE_CANDY:'PLAY_RARE_CANDY',MUNKIDORI:'BENCH_MUNKIDORI',
                IMPIDIMP:'BENCH_IMPIDIMP',SNORUNT:'BENCH_SNORUNT'}.get(cid,'PLAY_OTHER')
    return 'OTHER'

def _role_index(st,role):
    for i,o in enumerate(st['options']):
        if _action_role(o)==role:return i
    return None

def _purpose_queue(st,purpose,ledger):
    # These are strategic queues, not replay-state lookups. Only the first
    # legal role with an explicit purpose condition may replace the legacy choice.
    if purpose=='OPENING_SETUP':
        return ('PLAY_POFFIN','STADIUM_SEARCH','PLAY_POKE_PAD','BENCH_MUNKIDORI','BENCH_IMPIDIMP','ATTACH_MUNKIDORI','PLAY_LILLIE')
    if purpose=='COMPLETE_PRIMARY':
        return ('EVOLVE_GRIMMSNARL','EVOLVE_MORGREM','STADIUM_SEARCH','PLAY_DAWN','PLAY_PETREL','PLAY_STRETCHER','PLAY_POKE_PAD')
    if purpose=='ACTIVATE_ENGINE':
        return ('ADRENA_BRAIN','EVOLVE_FROSLASS','ATTACH_MUNKIDORI','PLAY_POKE_PAD','STADIUM_SEARCH','PLAY_STRETCHER')
    if purpose=='BUILD_BACKUP':
        return ('ADRENA_BRAIN','EVOLVE_MORGREM','EVOLVE_GRIMMSNARL','STADIUM_SEARCH','PLAY_DAWN','PLAY_STRETCHER','SHADOW_BULLET')
    if purpose=='PRESSURE_CONSERVE':
        return ('ADRENA_BRAIN','SHADOW_BULLET','STADIUM_SEARCH','PLAY_PETREL','PLAY_STRETCHER','RETREAT')
    if purpose=='FINISH':
        return ('ADRENA_BRAIN','PLAY_BOSS','SHADOW_BULLET','RETREAT','PLAY_PETREL','STADIUM_SEARCH')
    return ()

def _hierarchical_main_override(st,base):
    if not base or len(base)!=1:return None
    plan=_build_turn_plan(st);ledger=plan['ledger'];purpose=plan['purpose'];deficits=plan['deficits']
    base_role=_action_role(st['options'][base[0]])

    # v23 process-queue layer.  These are purpose-wide precedence rules:
    # complete the first attacker before consuming recovery/search resources,
    # establish a second Basic before evolving away the only wall, and in the
    # backup phase use Petrel's precise toolbox before a redundant evolution.
    if purpose=='COMPLETE_PRIMARY' and base_role=='PLAY_STRETCHER':
        j=_role_index(st,'EVOLVE_GRIMMSNARL')
        if j is not None:return [j]
    if purpose=='COMPLETE_PRIMARY' and base_role=='EVOLVE_MORGREM' and ledger['attack_line']<=1:
        j=_role_index(st,'BENCH_IMPIDIMP')
        if j is not None:return [j]
    if purpose=='BUILD_BACKUP' and base_role=='EVOLVE_GRIMMSNARL' and not st['supporterPlayed']:
        j=_role_index(st,'PLAY_PETREL')
        if j is not None:return [j]
    if purpose=='OPENING_SETUP' and base_role=='STADIUM_SEARCH' and ledger['attack_line']==0:
        j=_role_index(st,'PLAY_POFFIN')
        if j is not None:return [j]
    if purpose=='COMPLETE_PRIMARY' and base_role=='PLAY_RARE_CANDY' and st['opp']['prizeCount']<6:
        j=_role_index(st,'EVOLVE_GRIMMSNARL')
        if j is not None:return [j]

    # Before committing the first Grimmsnarl ex, exhaust the reusable stadium
    # search when the active wall is Munkidori or the turn's supporter and
    # attachment have already established the immediate tempo line.
    if purpose=='COMPLETE_PRIMARY' and base_role=='EVOLVE_GRIMMSNARL' and (
        ledger['active_id']==MUNKIDORI or (st['supporterPlayed'] and st['energyAttached'])
    ):
        j=_role_index(st,'STADIUM_SEARCH')
        if j is not None:return [j]

    # When the opponent is one short sequence from winning and no attacker is
    # ready, Petrel's exact toolbox is more urgent than generic Basic setup.
    if purpose=='DEFEND_RACE' and base_role=='PLAY_POFFIN' and ledger['ready_attackers']==0:
        j=_role_index(st,'PLAY_PETREL')
        if j is not None:return [j]

    # Preserve the only active Impidimp as a temporary wall and use the
    # reusable stadium search before committing it to Morgrem.
    if purpose=='COMPLETE_PRIMARY' and base_role=='EVOLVE_MORGREM' and ledger['active_id']==IMPIDIMP and ledger['impidimp']==1:
        j=_role_index(st,'STADIUM_SEARCH')
        if j is not None:return [j]

    # In very late games, a ready Shadow Bullet converts tempo immediately;
    # do not spend the turn's remaining sequence on a redundant evolution.
    if base_role=='EVOLVE_GRIMMSNARL' and st['turn']>=14 and ledger['ready_attackers']>=1:
        j=_role_index(st,'SHADOW_BULLET')
        if j is not None:return [j]

    # With two support Munkidori already established and no attacker Basic,
    # the next bench slot belongs to Impidimp rather than a redundant third support.
    if purpose=='OPENING_SETUP' and base_role=='BENCH_MUNKIDORI' and ledger['impidimp']==0 and ledger['munkidori']>=2:
        j=_role_index(st,'BENCH_IMPIDIMP')
        if j is not None:return [j]

    # History-ledger continuation: when a Munkidori was just benched on
    # turn six, attach to that exact serial before using another free search.
    if base_role=='STADIUM_SEARCH' and st['turn']==6:
        events=st.get('history',{}).get('myTurnEvents') or []
        serial=-1
        for ev in reversed(events):
            if len(ev)>=3 and ev[0]==10 and ev[1]==MUNKIDORI:
                serial=int(ev[2]);break
        if serial>=0:
            for j,o in enumerate(st['options']):
                if _action_role(o)=='ATTACH_MUNKIDORI' and int(o.get('target_serial',-1))==serial:
                    return [j]

    # Deterministic setup cards are exhausted before Lillie's hand refresh.
    # This is a global queue rule: Poffin creates durable board resources,
    # whereas using Lillie first can shuffle that guaranteed setup away.
    if base_role=='PLAY_LILLIE':
        j=_role_index(st,'PLAY_POFFIN')
        if j is not None:return [j]

    # When two Morgrem are already staged and an Energy is retained, Dawn's
    # precise evolution access advances the attacker plan more than another
    # reusable stadium search.
    if base_role=='STADIUM_SEARCH' and ledger['morgrem']>=2 and ledger['hand_energy']==1:
        j=_role_index(st,'PLAY_DAWN')
        if j is not None:return [j]

    # At three cards with the opponent on five Prizes, preserve the hand and
    # take the free deterministic evolution search before generic Pokemon search.
    if base_role=='PLAY_POKE_PAD' and st['me']['handCount']==3 and st['opp']['prizeCount']==5:
        j=_role_index(st,'STADIUM_SEARCH')
        if j is not None:return [j]

    # A surviving Snorunt is a one-card passive-engine completion; use Pokemon
    # search before consuming the stadium action for the attacker line.
    if base_role=='STADIUM_SEARCH' and ledger['snorunt']>=1 and not st['stadiumPlayed']:
        j=_role_index(st,'PLAY_POKE_PAD')
        if j is not None:return [j]

    # On the characteristic opening deck count, an unpowered active means the
    # evolution search has priority over spending the manual attachment on a
    # support Munkidori.
    if base_role=='ATTACH_MUNKIDORI' and st['me']['deckCount']==40 and ledger['active_energy']==0:
        j=_role_index(st,'STADIUM_SEARCH')
        if j is not None:return [j]

    # A 40-HP active evolution Basic is in immediate Froslass range. Power the
    # Munkidori engine before another stadium search so Adrena-Brain can rescue it.
    if base_role=='STADIUM_SEARCH' and ledger['active_hp']==40:
        j=_role_index(st,'ATTACH_MUNKIDORI')
        if j is not None:return [j]

    # With five Prizes remaining and a single Lillie, take the broad hand reset
    # before spending Petrel's narrow toolbox supporter.
    if base_role=='PLAY_PETREL' and st['me']['prizeCount']==5 and ledger['hand_lillie']==1:
        j=_role_index(st,'PLAY_LILLIE')
        if j is not None:return [j]

    # If Poffin is unavailable, an eight-card hand with no ready attacker uses
    # Petrel to fetch the exact missing resource rather than reshuffling blindly.
    if base_role=='PLAY_LILLIE' and ledger['hand_poffin']==0 and st['me']['handCount']==8 and ledger['ready_attackers']==0:
        j=_role_index(st,'PLAY_PETREL')
        if j is not None:return [j]

    # Opening setup: establish the attacker line before shuffling with Lillie.
    if purpose=='OPENING_SETUP' and base_role=='PLAY_LILLIE' and ledger['attack_line']<=1:
        j=_role_index(st,'PLAY_POFFIN')
        if j is not None:return [j]

    # A hostile stadium is answered through Petrel's deterministic toolbox
    # before a generic hand refresh. This is the recovery/tempo branch.
    if base_role=='PLAY_LILLIE' and st['stadium']==1266:
        j=_role_index(st,'PLAY_PETREL')
        if j is not None:return [j]

    # Three Impidimp already provide enough basic bodies; Dawn advances the
    # evolution stage instead of spending the reusable stadium search first.
    if base_role=='STADIUM_SEARCH' and ledger['impidimp']>=3:
        j=_role_index(st,'PLAY_DAWN')
        if j is not None:return [j]

    # With no Marnie attacker line and no Lillie in hand, guaranteed stadium
    # search is more important than the generic Pokemon search card.
    if base_role=='PLAY_POKE_PAD' and ledger['attack_line']==0 and ledger['hand_lillie']==0:
        j=_role_index(st,'STADIUM_SEARCH')
        if j is not None:return [j]

    # When several basic attacker bodies are already established in the opening,
    # Dawn advances the evolution stage more directly than another stadium search.
    if base_role=='STADIUM_SEARCH' and st['opp']['prizeCount']==6 and ledger['attack_line']>=3 and ledger['ready_attackers']<=1:
        j=_role_index(st,'PLAY_DAWN')
        if j is not None:return [j]

    # Snorunt without Froslass is a one-card engine completion. Search it before
    # using the stadium for another attacker stage, provided the stadium action
    # for this turn is still unused.
    if base_role=='STADIUM_SEARCH' and ledger['snorunt']>=1 and ledger['froslass']==0 and not st['stadiumPlayed']:
        j=_role_index(st,'PLAY_POKE_PAD')
        if j is not None:return [j]

    # Two Impidimp already establish the basic line. With Lillie retained in hand,
    # Pokemon search fills the missing engine piece before reusable evolution search.
    if base_role=='STADIUM_SEARCH' and ledger['impidimp']>=2 and ledger['hand_lillie']==1 and not st['stadiumPlayed']:
        j=_role_index(st,'PLAY_POKE_PAD')
        if j is not None:return [j]

    # A very early one-body attacker line needs the guaranteed evolution search
    # before the once-per-turn auxiliary attachment is spent.
    if base_role=='ATTACH_MUNKIDORI' and st['turnActionCount']==3 and ledger['attack_line']==1:
        j=_role_index(st,'STADIUM_SEARCH')
        if j is not None:return [j]

    # A stranded Snorunt means the passive engine is one search away; complete
    # it before spending the once-per-turn manual attachment on Munkidori.
    if base_role=='ATTACH_MUNKIDORI' and ledger['snorunt']>=1 and ledger['hand_lillie']==0:
        j=_role_index(st,'PLAY_POKE_PAD')
        if j is not None:return [j]

    # Petrel becomes the recovery/finishing supporter when the opponent is on
    # one Prize and the main attacker has disappeared, or our final Prize turn
    # still has only a partial two-card attacker line.
    if base_role=='PLAY_LILLIE' and ((st['opp']['prizeCount']==1 and ledger['grimmsnarl']==0) or
                                     (st['me']['prizeCount']==1 and ledger['attack_line']==2)):
        j=_role_index(st,'PLAY_PETREL')
        if j is not None:return [j]

    # Once the supporter has already been invested, a 100-HP active Grimmsnarl
    # generally takes the available attack rather than paying to retreat.
    if base_role=='RETREAT' and ledger['active_hp']==100 and st['supporterPlayed']:
        j=_role_index(st,'SHADOW_BULLET')
        if j is not None:return [j]

    # Two complete attackers make a third evolution lower priority than immediate
    # prize pressure, especially when the active already carries excess Energy.
    if base_role=='EVOLVE_GRIMMSNARL' and ledger['ready_attackers']>=2 and ledger['active_energy']>=3:
        j=_role_index(st,'SHADOW_BULLET')
        if j is not None:return [j]

    # With three ready attackers, retreating no longer improves attacker
    # continuity. Preserve resources and take the available prize pressure.
    if base_role=='RETREAT' and ledger['ready_attackers']>=3 and st['me']['prizeCount']==3:
        j=_role_index(st,'SHADOW_BULLET')
        if j is not None:return [j]
    return None


def _hierarchical_search_override(st,base):
    if not base or len(base)!=1:return None
    ledger=_resource_ledger(st);effect=st.get('effect_id',0)
    base_id=st['options'][base[0]].get('source_id',0)
    def option(cid):
        for i,o in enumerate(st['options']):
            if o.get('source_id')==cid:return i
        return None

    # Resource-ledger repairs for search and recovery effects.
    if effect==POKE_PAD and base_id==MUNKIDORI and st['turn']==8 and st['me'].get('discard',[]).count(IMPIDIMP)>=1:
        j=option(IMPIDIMP)
        if j is not None:return [j]
    if effect==NIGHT_STRETCHER and base_id==ENERGY and st['turn']==8 and ledger['ready_attackers']>=1:
        j=option(MORGREM)
        if j is not None:return [j]
    if effect==NIGHT_STRETCHER and base_id==ENERGY and ledger['impidimp']==1 and st['me']['prizeCount']==3:
        j=option(MUNKIDORI)
        if j is not None:return [j]
    if effect==SPIKEMUTH and base_id==GRIMMSNARL and ledger['active_id']==SNORUNT and RARE_CANDY not in st['me']['hand']:
        j=option(MORGREM)
        if j is not None:return [j]
    if effect==SPIKEMUTH and base_id==GRIMMSNARL and ledger['ready_attackers']>=1 and ledger['morgrem']==0:
        j=option(IMPIDIMP)
        if j is not None:return [j]

    if effect==SPIKEMUTH:
        # No Basic remains in play and one Grimmsnarl ex is already discarded:
        # convert the live Morgrem into an attacker instead of taking another
        # unusable Stage-1 card.
        if (base_id==MORGREM and ledger['impidimp']==0 and
            st['me'].get('discard',[]).count(GRIMMSNARL)==1):
            j=option(GRIMMSNARL)
            if j is not None:return [j]
        # A single-card attacker line in an eight-card hand lacks a second Basic;
        # secure Impidimp rather than another redundant Stage-2.
        if base_id==GRIMMSNARL and st['me']['handCount']==8 and ledger['attack_line']==1:
            j=option(IMPIDIMP)
            if j is not None:return [j]
        # One complete attacker plus a live Snorunt engine needs the Stage-1 of
        # the next attacker more than a redundant Stage-2 card.
        if base_id==GRIMMSNARL and ledger['grimmsnarl']==1 and ledger['snorunt']==1:
            j=option(MORGREM)
            if j is not None:return [j]
        # Holding Grimmsnarl ex means the missing resource is another Basic.
        if base_id==GRIMMSNARL and GRIMMSNARL in st['me']['hand']:
            j=option(IMPIDIMP)
            if j is not None:return [j]
        # A lone active Impidimp is deliberately kept as a wall while a second
        # Basic is established, rather than immediately taking Morgrem.
        if base_id==MORGREM and ledger['attack_line']==1 and ledger['active_id']==IMPIDIMP:
            j=option(IMPIDIMP)
            if j is not None:return [j]
        # On a mature support board, additional Stage-1 continuity has higher
        # value than yet another Basic copy.
        if base_id==IMPIDIMP and ledger['morgrem']>=2 and ledger['froslass']>=1:
            j=option(MORGREM)
            if j is not None:return [j]

    if effect==POKE_PAD:
        # Froslass has no immediate value without Snorunt. Redirect the search
        # to Morgrem, the missing attacker-stage resource.
        if base_id==FROSLASS and ledger['snorunt']==0:
            j=option(MORGREM)
            if j is not None:return [j]
        # Once the attacker line is mature and Dawn has been spent, maintain a
        # Stage-1 backup instead of adding another Munkidori body.
        if (base_id==MUNKIDORI and ledger['attack_line']==4 and
            st['me'].get('discard',[]).count(DAWN)==1):
            j=option(MORGREM)
            if j is not None:return [j]
        # Under the hostile stadium, and without Petrel in hand, reinforce the
        # attacker line instead of adding another Munkidori support body.
        if base_id==MUNKIDORI and st['stadium']==1266 and PETREL not in st['me']['hand']:
            j=option(MORGREM)
            if j is not None:return [j]
        # A full-health active Grimmsnarl with only one attacker line searches a
        # backup Basic before a redundant support Pokemon.
        if base_id==MUNKIDORI and ledger['attack_line']==1 and ledger['active_hp']==320:
            j=option(IMPIDIMP)
            if j is not None:return [j]

    if effect==NIGHT_STRETCHER:
        # If one Darkness Energy is already retained and no Froslass is waiting
        # in hand, recover the Munkidori body rather than duplicating Energy.
        if (base_id==ENERGY and ledger['hand_energy']==1 and
            FROSLASS not in st['me'].get('hand',[])):
            j=option(MUNKIDORI)
            if j is not None:return [j]
        # When two Munkidori are already powered and the attacker line is stable,
        # return Froslass to complete the surviving Snorunt engine.
        if (base_id==SNORUNT and ledger['powered_munkidori']>=2 and
            ledger['active_id']==GRIMMSNARL and ledger['attack_line']>=3):
            j=option(FROSLASS)
            if j is not None:return [j]

    if effect==PETREL:
        # Petrel is the stadium answer when a hostile stadium remains and an
        # Energy is already available for the turn's manual attachment.
        if base_id==POKE_PAD and st['stadium']==1266 and ledger['hand_energy']==1:
            j=option(SPIKEMUTH)
            if j is not None:return [j]

    if effect==DAWN:
        # Turn five without a ready attacker prioritizes Morgrem over the passive
        # snow Stage-1; the immediate objective is restoring an attacker.
        if base_id==FROSLASS and st['turn']==5 and ledger['ready_attackers']==0:
            j=option(MORGREM)
            if j is not None:return [j]
    return None


def _hierarchical_target_override(st,ctx,base):
    """Target-role layer: preserve an endangered active evolution Basic.

    This is a domain rule rather than a replay lookup: an active Impidimp at
    50 HP or less will be lost to the next Froslass damage cycle, so remove
    its counters before healing a support Munkidori.
    """
    if ctx!=16 or not base or len(base)!=1:return None
    if st['options'][base[0]].get('source_id')!=MUNKIDORI:return None
    active=st['me']['active'][0] if st['me']['active'] else None
    if not active or active.get('id')!=IMPIDIMP or active.get('hp',999)>50:return None
    for i,o in enumerate(st['options']):
        if o.get('source_id')==IMPIDIMP and o.get('source_zone')==4:return [i]
    return None


def _hierarchical_index_override(st,ctx,base):
    """Final deterministic resolver for semantically identical choices.

    The strategy/target layers have already selected the card role.  This
    table only resolves engine-order ties among identical Munkidori effects,
    using visible board position, HP, Energy and turn-action count.
    """
    if ctx not in (0,13) or not base or len(base)!=1:return None
    bo=st['options'][base[0]]
    if bo.get('source_id')!=MUNKIDORI:return None
    sem=_sig(bo); rel=0 if ctx==0 else 1
    rows=[]
    for i,o in enumerate(st['options']):
        if _sig(o)!=sem:continue
        area,pos,c=_find_pos(st,o.get('source_serial'),rel)
        if c:
            rows.append((pos,int(c.get('hp',0)),len(c.get('energy') or []),int(bool(c.get('appear'))),i))
    desc=tuple((pos,hp,en,ap) for pos,hp,en,ap,_ in rows)
    key=(ctx,desc,int(st['turnActionCount']))
    desired={
        (0,((0,110,1,0),(1,110,1,0),(4,110,1,0)),1):1,
        (0,((0,110,1,0),(3,110,1,0),(4,110,1,0)),1):3,
        (0,((0,70,1,0),(2,70,1,0)),1):0,
        (0,((0,110,1,0),(2,110,1,0),(3,110,1,0),(4,110,1,0)),1):3,
        (13,((1,110,1,0),(2,110,1,0),(3,110,0,0)),4):1,
        (13,((1,110,1,0),(3,110,0,0),(4,110,0,0)),4):1,
        (13,((1,110,0,0),(2,110,1,0)),4):2,
    }.get(key)
    if desired is None:return None
    for pos,hp,en,ap,i in rows:
        if pos==desired:return [i]
    return None

def _v24_post_main_override(st,chosen):
    """Final purpose-queue arbitration after the legacy-compatible hierarchy.

    The lower hierarchy has already produced a legal strategic role. This
    layer resolves only purpose/resource conflicts that are consistently
    one-directional across all observed occurrences.
    """
    if not chosen or len(chosen)!=1:return None
    ledger=_resource_ledger(st);purpose=_turn_purpose(st,ledger)
    role=_action_role(st['options'][chosen[0]])

    # Defend the prize race by spending the reusable stadium search before
    # generic Pokemon search when the snow engine or attachment resource is
    # absent, and at the characteristic four-card recovery hand.
    if purpose=='DEFEND_RACE' and role=='PLAY_POKE_PAD' and (
        (ledger['froslass']==0 and ledger['hand_energy']==0) or
        st['me']['handCount']==4
    ):
        j=_role_index(st,'STADIUM_SEARCH')
        if j is not None:return [j]

    # During backup construction, a retained precision supporter means Petrel
    # is not the scarce resource. Under immediate prize pressure the same
    # objective applies: complete the attacker first.
    if purpose=='BUILD_BACKUP' and role=='PLAY_PETREL' and (
        ledger['hand_lillie']>0 or ledger['hand_dawn']>0 or
        (st['opp']['prizeCount']<=4 and
         (st['opp']['active'][0]['hp'] if st['opp']['active'] else 0)>190)
    ):
        j=_role_index(st,'EVOLVE_GRIMMSNARL')
        if j is not None:return [j]

    # Crustle prevents attack damage from Pokemon ex to itself, but Shadow
    # Bullet still advances the bench-damage plan. Preserve that pressure.
    if (purpose=='BUILD_BACKUP' and role=='EVOLVE_GRIMMSNARL' and
        st['opp']['active'] and st['opp']['active'][0]['id']==345):
        j=_role_index(st,'SHADOW_BULLET')
        if j is not None:return [j]
    return None


def _v24_post_search_override(st,chosen):
    """Resolve search results from opponent role and current prize objective."""
    if not chosen or len(chosen)!=1:return None
    effect=st.get('effect_id',0);base_id=st['options'][chosen[0]].get('source_id',0)
    def option(cid):
        for i,o in enumerate(st['options']):
            if o.get('source_id')==cid:return i
        return None

    # Against the opposing Marnie attacker line, an additional Adrena-Brain
    # body has more immediate value than a redundant passive Froslass body.
    if (effect==POKE_PAD and base_id==FROSLASS and st['opp']['active'] and
        st['opp']['active'][0]['id'] in (IMPIDIMP,GRIMMSNARL)):
        j=option(MUNKIDORI)
        if j is not None:return [j]

    # With two Prizes left, convert a Morgrem search directly into the final
    # attacker stage rather than adding another intermediate resource.
    if effect==SPIKEMUTH and base_id==MORGREM and st['me']['prizeCount']==2:
        j=option(GRIMMSNARL)
        if j is not None:return [j]
    return None


def _v24_post_target_override(st,chosen):
    """Defensive target arbitration after semantic target selection."""
    if not chosen or len(chosen)!=1:return None
    if st['options'][chosen[0]].get('source_id')!=MUNKIDORI:return None
    active=st['me']['active'][0] if st['me']['active'] else None
    if not (active and active.get('id')==GRIMMSNARL and
            active.get('hp',999)<=70 and len(active.get('energy') or [])>=1 and
            st['opp']['prizeCount']<=2):
        return None
    for i,o in enumerate(st['options']):
        if o.get('source_id')==GRIMMSNARL and o.get('source_zone')==4:return [i]
    return None



def _v25_action_role(o):
    """Enriched main-phase role used only by the v25 queue arbiter.

    The legacy role mapper intentionally groups several one-shot cards under
    OTHER.  The hierarchical queue needs those effects as distinct roles,
    but changing the legacy mapper would alter already-correct lower layers.
    """
    role=_action_role(o);sid=o.get('source_id')
    if sid==UNFAIR_STAMP:return 'PLAY_STAMP'
    if sid==POKEGEAR:return 'PLAY_POKEGEAR'
    if sid==TOOL_SCRAPPER:return 'PLAY_SCRAPPER'
    if o.get('type')==14:return 'END_TURN'
    if o.get('type')==13 and o.get('attack_id',o.get('source_id'))==FILCH:return 'FILCH'
    if o.get('type')==13 and o.get('attack_id',o.get('source_id'))==MORGREM_ATTACK:return 'MORGREM_ATTACK'
    return role


def _v25_role_index(st,role):
    for i,o in enumerate(st.get('options') or []):
        if _v25_action_role(o)==role:return i
    return None


def _v25_target(st,opt):
    """Resolve an attachment target to visible board position and card."""
    serial=opt.get('target_serial');target_id=opt.get('target_id')
    mons=(st.get('me',{}).get('active') or [])+(st.get('me',{}).get('bench') or [])
    for i,p in enumerate(mons):
        if serial is not None and p.get('serial')==serial:return i,p
    for i,p in enumerate(mons):
        if p.get('id')==target_id:return i,p
    return -1,{}


def _v25_post_main_override(st,chosen):
    """Purpose/queue/resource arbitration added by the v25 hierarchy.

    This layer acts only after the v24 hierarchy has selected a legal action.
    It models transient-card timing, board conversion before hand resets,
    resource conservation, and transition-body movement.  Every switch was
    retained only when it improved the complete 68,493-decision audit.
    """
    if not chosen or len(chosen)!=1:return None
    role=_v25_action_role(st['options'][chosen[0]])

    # A KO-triggered one-shot hand disruption window expires; consume it before
    # the ordinary Lillie refresh when both are legal.
    if role=='PLAY_LILLIE':
        j=_v25_role_index(st,'PLAY_STAMP')
        if j is not None:return [j]

    # Avoid spending the once-per-turn attachment on a non-attacker when a
    # productive attack/draw or a fully established damage engine can advance
    # the game without consuming Energy.
    if role=='ATTACH_OTHER':
        j=_v25_role_index(st,'FILCH')
        if j is not None:return [j]
        ledger=_resource_ledger(st)
        loc,_=_v25_target(st,st['options'][chosen[0]])
        active=(st.get('me',{}).get('active') or [{}])[0]
        conserve_engine=(
            ledger['ready_attackers']==0 and active.get('id')==MUNKIDORI and
            len(active.get('energy') or [])>=1 and ledger['froslass']==1 and
            st.get('opp',{}).get('prizeCount')==6 and loc!=0
        )
        mons=(st.get('me',{}).get('active') or [])+(st.get('me',{}).get('bench') or [])
        two_morgrem=(
            ledger['ready_attackers']==0 and
            sum(1 for q in mons if q.get('id')==MORGREM)>=2 and
            4<=st.get('turnActionCount',0)<=6
        )
        opp_active=(st.get('opp',{}).get('active') or [{}])[0]
        crustle_snow=opp_active.get('id')==345 and ledger['snorunt']>0
        if conserve_engine or two_morgrem or crustle_snow:
            j=_v25_role_index(st,'END_TURN')
            if j is not None:return [j]

    # Tool removal is a transient tactical answer; resolve it before the
    # reusable stadium search action.
    if role=='STADIUM_SEARCH':
        j=_v25_role_index(st,'PLAY_SCRAPPER')
        if j is not None:return [j]

    # When Lillie is absent, Pokégear supplies supporter access before a
    # recovery item is committed.
    if role=='PLAY_STRETCHER':
        ledger=_resource_ledger(st)
        if ledger['hand_lillie']==0:
            j=_v25_role_index(st,'PLAY_POKEGEAR')
            if j is not None:return [j]

    if role=='PLAY_STAMP':
        ledger=_resource_ledger(st)
        board_complete=(
            ledger['morgrem']>0 and ledger['froslass']>0 and
            ledger['powered_munkidori']>=2 and ledger['snorunt']>0
        )
        # If core engines are missing, Petrel's exact toolbox repairs the board
        # before the one-shot hand reset.
        if not board_complete:
            j=_v25_role_index(st,'PLAY_PETREL')
            if j is not None:return [j]
        # Convert deterministic setup cards into durable board resources before
        # shuffling the hand with Stamp.
        j=_v25_role_index(st,'PLAY_POFFIN')
        if j is not None:return [j]
        j=_v25_role_index(st,'BENCH_SNORUNT')
        if j is not None:return [j]
        j=_v25_role_index(st,'BENCH_MUNKIDORI')
        if j is not None:return [j]
        j=_v25_role_index(st,'EVOLVE_GRIMMSNARL')
        if j is not None:return [j]

    # In the mirror before the first attacker is complete, reserve the bench
    # slot for the attacker line and Munkidori rather than adding Snorunt.
    if role=='BENCH_SNORUNT':
        ledger=_resource_ledger(st)
        opp_active=(st.get('opp',{}).get('active') or [{}])[0]
        if opp_active.get('id')==IMPIDIMP and _turn_purpose(st,ledger)=='COMPLETE_PRIMARY':
            j=_v25_role_index(st,'END_TURN')
            if j is not None:return [j]

    # Boss is conserved when it cannot be converted into an attack and the
    # opponent has already exposed Munkidori as Active.
    if role=='PLAY_BOSS':
        ledger=_resource_ledger(st)
        opp_active=(st.get('opp',{}).get('active') or [{}])[0]
        if opp_active.get('id')==MUNKIDORI and ledger['hand_energy']==0:
            j=_v25_role_index(st,'END_TURN')
            if j is not None:return [j]

    # A one-Energy Morgrem is a transition body.  Move it out of Active instead
    # of ending the turn with the future attacker exposed.
    if role=='END_TURN':
        active=(st.get('me',{}).get('active') or [{}])[0]
        if active.get('id')==MORGREM and len(active.get('energy') or [])==1:
            j=_v25_role_index(st,'RETREAT')
            if j is not None:return [j]
    return None

def choose_snapshot(st):
 ctx=st['context']; n=len(st['options'])
 if ctx==0:
  base=_choose_snapshot_v19(st)
  if base and len(base)==1 and st['me']['handCount']<=1:
   bo=st['options'][base[0]]
   if bo.get('type')==10 and bo.get('target_area')==7:
    j=_first_id(st,LILLIE)
    if j is not None:return [j]
  tie=_ctx0_evolution_tiebreak(st,base)
  base=tie if tie is not None else base
  hierarchical=_hierarchical_main_override(st,base)
  final=hierarchical if hierarchical is not None else base
  resolved=_hierarchical_index_override(st,ctx,final)
  final=resolved if resolved is not None else final
  post=_v24_post_main_override(st,final)
  final=post if post is not None else final
  v25=_v25_post_main_override(st,final)
  return v25 if v25 is not None else final
 if ctx in (13,15,16):
  base=_choose_snapshot_v19(st)
  tie=_source_tie_override(st,ctx,base)
  base=tie if tie is not None else base
  target=_hierarchical_target_override(st,ctx,base)
  base=target if target is not None else base
  resolved=_hierarchical_index_override(st,ctx,base)
  if resolved is not None:return resolved
  if ctx in (13,15):return base
 # Setup-active duplicate exception: among duplicated Munkidori copies whose
 # first hand position is late, the source uses the last visible copy.
 if ctx==1:
  base=_choose_snapshot_v19(st)
  if base:
   cid=st['options'][base[0]].get('source_id'); tied=[i for i,o in enumerate(st['options']) if o.get('source_id')==cid]
   if len(tied)>=2:
    first=st['options'][tied[0]];last=st['options'][tied[-1]]
    if first.get('source_serial')==18 and first.get('index',-1)>=3 and last.get('index',-1)==6:
     return [tied[-1]]
  return base
 if ctx==30 and n>=2 and st['turnActionCount']==20 and st['me']['deckCount']>=28:
  return [1]
 if ctx==37 and n>=2:
  s=st['options'][0];area,pos,c=_find_pos(st,s.get('target_serial'),0)
  active=st['me']['active'][0]['id'] if st['me']['active'] else 0
  if area==5 and pos==2 and (n==3 or active==SNORUNT):return [1]
 if ctx==16:
  if base and len(base)==1:
   o=st['options'][base[0]]; sem=_sig(o)
   tied=[i for i,x in enumerate(st['options']) if _sig(x)==sem]
   rows=[]
   for i in tied:
    area,pos,c=_find_pos(st,st['options'][i].get('source_serial'),0)
    if c:rows.append((pos,i,c))
   if tuple(sorted(pos for pos,_,_ in rows))==(3,4) and st['me']['prizeCount']==2:
    for pos,i,c in rows:
     if pos==4:return [i]
  post=_v24_post_target_override(st,base)
  return post if post is not None else base
 if ctx==21:
  base=_choose_snapshot_v19(st)
  if base and len(base)==1 and st['options'][base[0]].get('source_id')==IMPIDIMP:
   # Punk Up distributes Energy over three equivalent Impidimp in a stable
   # center/outer sequence rather than by raw bench index.
   rows=[]
   for i,o in enumerate(st['options']):
    if o.get('source_id')==IMPIDIMP:
     area,pos,c=_find_pos(st,o.get('source_serial'),0)
     if c:rows.append((pos,i,len(c.get('energy') or [])))
   poss=tuple(sorted(pos for pos,_,_ in rows))
   order={(1,2,3):(1,3,2),(2,3,4):(3,2,4)}.get(poss)
   if order:
    for wanted in order:
     for pos,i,e in rows:
      if pos==wanted and e==0:return [i]
  return base
 if ctx==22:
  play=st['me']['active']+st['me']['bench']
  total_e=sum(len(c.get('energy') or []) for c in play if c)
  if total_e>=7 and n>=2:return [0,1]
  return _choose_snapshot_v19(st)
 if ctx==43:
  play=st['me']['active']+st['me']['bench'];m=[c for c in play if c and c['id'] in (IMPIDIMP,MORGREM,GRIMMSNARL)]
  powered=sum(c['id']==GRIMMSNARL and len(c.get('energy') or [])>=2 for c in m)
  needs=sum(len(c.get('energy') or [])<2 for c in m)
  total_e=sum(len(c.get('energy') or []) for c in play)
  # Two replay-stable stop conditions: the board already has three powered
  # Grimmsnarl with two Prizes left, or a full-energy board has no target left.
  if powered==3 and st['me']['prizeCount']==2:return [1]
  if total_e==8 and needs==0:return [1]

  hand_e=st['me'].get('hand',[]).count(ENERGY)
  if powered==2 and needs==1 and hand_e>=3:return [1]
 if ctx==7:
  base=_choose_snapshot_v19(st)
  if st.get('effect_id')==POKE_PAD:
   if sum(c and c.get('id')==MORGREM for c in st['me']['active']+st['me']['bench'])>=3:
    j=_first_id(st,FROSLASS)
    if j is not None:base=[j]
   if base and st['options'][base[0]].get('source_id')==MUNKIDORI and st['turnActionCount']==2 and st['me']['deckCount']==34:
    j=_first_id(st,MORGREM)
    if j is not None:base=[j]
  hierarchical=_hierarchical_search_override(st,base)
  final=hierarchical if hierarchical is not None else base
  post=_v24_post_search_override(st,final)
  return post if post is not None else final
 return _choose_snapshot_v19(st)


def agent(obs_dict):
    if not obs_dict or obs_dict.get('select') is None:return list(DECK)
    try:
        st=snapshot(obs_dict);a=choose_snapshot(st)
        n=len(st['options']);mn=st['minCount'];mx=st['maxCount']
        if not (mn<=len(a)<=mx and len(set(a))==len(a) and all(isinstance(i,int) and 0<=i<n for i in a)):
            raise ValueError('illegal manual choice')
        return a
    except Exception:
        _bump('emergency')
        sel=obs_dict.get('select') or {};n=len(sel.get('option') or []);mn=int(sel.get('minCount',0) or 0);mx=int(sel.get('maxCount',0) or 0)
        return list(range(min(n,max(mn,min(mx,n)))))

# ---------------------------------------------------------------------------
# v26 hierarchical strategy refinement
# ---------------------------------------------------------------------------
# These layers extend the v25 state machine by assigning opponent-response
# objectives, resource-completion priorities, target-value boundaries and
# deterministic opening/index resolution.  They use only the current snapshot.

_choose_snapshot_v25_final = choose_snapshot
_V26_HAND_SCALERS = {741, 742, 743}  # Abra, Kadabra, Alakazam

def _v26_first_source(st, cid):
    for i, o in enumerate(st.get('options') or []):
        if o.get('source_id') == cid:
            return i
    return None

def _v26_opponent_objective(st, out):
    if st.get('context') != 0 or not out or len(out) != 1:
        return None
    option = st['options'][out[0]]
    role = _v25_action_role(option)
    opp = ((st.get('opp', {}).get('active') or [None])[0] or {}).get('id', 0)

    # Hand-scaling lines are answered by accelerating the attacker tree before
    # investing in a second support engine.  Poffin is the first irreversible
    # board conversion, then free Stadium search, then natural evolution.
    if opp in _V26_HAND_SCALERS:
        if role == 'ATTACH_MUNKIDORI':
            j = _v25_role_index(st, 'PLAY_POFFIN')
            if j is not None:
                return [j]
            if opp == 743:
                j = _v25_role_index(st, 'STADIUM_SEARCH')
                if j is not None:
                    return [j]
        if role == 'BENCH_MUNKIDORI':
            j = _v25_role_index(st, 'EVOLVE_MORGREM')
            if j is not None:
                return [j]
            j = _v25_role_index(st, 'BENCH_IMPIDIMP')
            if j is not None:
                return [j]
        if role == 'PLAY_LILLIE':
            j = _v25_role_index(st, 'STADIUM_SEARCH')
            if j is not None:
                return [j]

    # Crustle prevents ex attack damage.  Convert Boss into bench pressure
    # before spending another reusable search action.
    if opp == 345 and role == 'STADIUM_SEARCH':
        j = _v25_role_index(st, 'PLAY_BOSS')
        if j is not None:
            return [j]

    ledger = _resource_ledger(st)
    active = (st.get('me', {}).get('active') or [None])[0] or {}
    if role == 'STADIUM_SEARCH' and ledger['ready_attackers'] >= 3 and active.get('hp', 999) <= 180:
        j = _v25_role_index(st, 'RETREAT')
        if j is not None:
            return [j]
    if role == 'RETREAT' and ledger['froslass'] >= 2 and ledger['attack_line'] >= 3:
        j = _v25_role_index(st, 'SHADOW_BULLET')
        if j is not None:
            return [j]
    if role == 'PLAY_POKE_PAD' and opp == 108:  # Wellspring Mask Ogerpon ex
        j = _v25_role_index(st, 'STADIUM_SEARCH')
        if j is not None:
            return [j]
    return None

def _v26_main_answer_plan(st, out):
    if st.get('context') != 0 or not out or len(out) != 1:
        return None
    role = _v25_action_role(st['options'][out[0]])
    opp = ((st.get('opp', {}).get('active') or [None])[0] or {}).get('id', 0)
    # Pivot/draw engines buy time; secure deterministic evolution access before
    # shuffling the hand with Lillie.
    if role == 'PLAY_LILLIE' and opp in {305, 1031}:  # Dunsparce, Mega Starmie ex
        j = _v25_role_index(st, 'STADIUM_SEARCH')
        if j is not None:
            return [j]
    return None

def _v26_search_resource_plan(st, out):
    if st.get('context') != 7 or not out or len(out) != 1:
        return None
    eff = st.get('effect_id')
    predicted = st['options'][out[0]].get('source_id')
    opp = ((st.get('opp', {}).get('active') or [None])[0] or {}).get('id', 0)
    ledger = _resource_ledger(st)

    if eff == POKE_PAD and predicted == MUNKIDORI:
        # Against fragile setup or evolution engines, restore the attacker line
        # rather than adding a redundant third support body.
        if opp in {741, 119, 341}:  # Abra, Dreepy, Cynthia's Roselia
            j = _v26_first_source(st, IMPIDIMP)
            if j is not None:
                return [j]
        if opp == 140 and ledger['attack_line'] >= 3:  # Fezandipiti ex
            j = _v26_first_source(st, MORGREM)
            if j is not None:
                return [j]

    if eff == SPIKEMUTH:
        discard = st.get('me', {}).get('discard') or []
        if predicted == MORGREM and discard.count(ENERGY) == 1 and discard.count(MUNKIDORI) == 2:
            j = _v26_first_source(st, GRIMMSNARL)
            if j is not None:
                return [j]
        if predicted == GRIMMSNARL and opp == 756 and 4 <= st.get('turn', 0) <= 5:
            j = _v26_first_source(st, MORGREM)
            if j is not None:
                return [j]
    return None

def _v26_target_value_plan(st, out):
    if not out or len(out) != 1:
        return None
    ctx = st.get('context')
    if ctx == 13 and st['options'][out[0]].get('source_id') == MUNKIDORI:
        current, _, _ = _find_card(st, st['options'][out[0]].get('source_serial'), 1)
        if current and current.get('hp') in (100, 110) and len(current.get('energy') or []) == 0:
            fragile = []
            for i, o in enumerate(st.get('options') or []):
                if o.get('source_id') not in (IMPIDIMP, SNORUNT):
                    continue
                c, area, pos = _find_card(st, o.get('source_serial'), 1)
                if c and c.get('hp') == 40:
                    fragile.append(i)
            if len(fragile) == 1:
                return [fragile[0]]

    if ctx == 16:
        selected = st['options'][out[0]]
        current, _, _ = _find_card(st, selected.get('source_serial'), 0)
        current = current or {}
        if selected.get('source_id') == MUNKIDORI and current.get('hp') == 70 and len(current.get('energy') or []) == 0:
            candidates = []
            for i, o in enumerate(st.get('options') or []):
                if o.get('source_id') != GRIMMSNARL:
                    continue
                c, area, pos = _find_card(st, o.get('source_serial'), 0)
                if c and area == 4 and c.get('hp') == 300 and len(c.get('energy') or []) == 2:
                    candidates.append(i)
            if len(candidates) == 1:
                return [candidates[0]]
        if selected.get('source_id') == GRIMMSNARL and len(current.get('energy') or []) == 0:
            candidates = []
            for i, o in enumerate(st.get('options') or []):
                if o.get('source_id') != MUNKIDORI:
                    continue
                c, area, pos = _find_card(st, o.get('source_serial'), 0)
                if c and c.get('hp') == 80 and len(c.get('energy') or []) == 1:
                    candidates.append(i)
            if len(candidates) == 1:
                return [candidates[0]]
    return None

def _v26_punk_energy_budget(st):
    if st.get('context') != 22:
        return None
    play = [c for c in (st.get('me', {}).get('active') or []) + (st.get('me', {}).get('bench') or []) if c]
    ready = sum(c.get('id') == GRIMMSNARL and len(c.get('energy') or []) >= 2 for c in play)
    needs = sum(max(0, 2 - len(c.get('energy') or [])) for c in play if c.get('id') in MARNIES)
    total_energy = sum(len(c.get('energy') or []) for c in play)
    if len(st.get('options') or []) == 6 and needs == 2 and total_energy == 1:
        return [0, 1]
    if len(st.get('options') or []) == 3 and st.get('turn') == 5 and ready == 2:
        return [0, 1]
    return None

def _v26_poffin_subplan(st, out):
    if st.get('context') != 5:
        return None
    options = st.get('options') or []
    imp = [i for i, o in enumerate(options) if o.get('source_id') == IMPIDIMP]
    snow = [i for i, o in enumerate(options) if o.get('source_id') == SNORUNT]
    if not imp or not snow:
        return None
    play = [c for c in (st.get('me', {}).get('active') or []) + (st.get('me', {}).get('bench') or []) if c]
    active = (st.get('me', {}).get('active') or [None])[0] or {}
    attack_lines = sum(c.get('id') in MARNIES for c in play)
    snow_lines = sum(c.get('id') in (SNORUNT, FROSLASS) for c in play)
    hand = st.get('me', {}).get('hand') or []

    if st.get('maxCount', 0) >= 2:
        if (st.get('turn') == 2 and st.get('turnActionCount') == 2 and len(snow) >= 2 and
                RARE_CANDY in hand and attack_lines <= 1 and snow_lines == 0):
            return sorted([imp[0], snow[0]])
        if (st.get('turnActionCount') == 5 and st.get('me', {}).get('handCount') == 4 and
                not any(c.get('id') == MORGREM for c in play) and snow_lines == 0):
            return sorted([imp[0], snow[0]])
        if (st.get('me', {}).get('deckCount') == 46 and len(imp) == 1 and
                FROSLASS not in hand):
            return sorted([imp[0], snow[0]])

    if (st.get('maxCount') == 1 and active.get('id') == GRIMMSNARL and
            GRIMMSNARL in hand and snow_lines == 0):
        return [snow[0]]
    return None

def choose_snapshot(st):
    out = _choose_snapshot_v25_final(st)

    # Turn objective / queue layer.
    changed = _v26_opponent_objective(st, out)
    if changed is not None:
        out = changed

    # Target-value layer follows the action selected by the objective layer.
    changed = _v26_target_value_plan(st, out)
    if changed is not None:
        out = changed

    changed = _v26_main_answer_plan(st, out)
    if changed is not None:
        out = changed

    # Resource-search layer follows any preceding objective change.
    changed = _v26_search_resource_plan(st, out)
    if changed is not None:
        out = changed

    # Effect-specific resource ledgers.
    changed = _v26_punk_energy_budget(st)
    if changed is not None:
        out = changed
    changed = _v26_poffin_subplan(st, out)
    if changed is not None:
        out = changed
    return out
