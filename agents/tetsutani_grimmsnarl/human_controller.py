from __future__ import annotations
from collections import Counter
import policy_features as pf
import human_memory as hm

# Fixed-deck IDs
DARK=7; FROSLASS=104; MUNKIDORI=112; IMPIDIMP=646; MORGREM=647; GRIM=648; SNORUNT=860
CANDY=1079; STAMP=1080; POFFIN=1086; STRETCHER=1097; POKEGEAR=1122; SCRAPPER=1137
POKEPAD=1152; BOSS=1182; PETREL=1219; LILLIE=1227; DAWN=1231; STADIUM=1259
SHADOW_BULLET=937
TERA_BENCH={96,117}
HIGH_VALUE={96,117,190,648,743,641,431}
ABILITY_ENGINE={96,104,112,190,648,743,269}


def _cur(obs): return obs.get('current') or {}
def _players(obs):
    c=_cur(obs); ps=c.get('players') or []; y=int(c.get('yourIndex',0) or 0)
    return y,(ps[y] if len(ps)>y else {}),(ps[1-y] if len(ps)>1-y else {})
def _cid(x): return int(x.get('id',0) or 0) if isinstance(x,dict) else 0
def _energy(x): return len((x or {}).get('energyCards') or []) if isinstance(x,dict) else 0
def _hp(x): return int((x or {}).get('hp',0) or 0) if isinstance(x,dict) else 0
def _damage(x): return max(0,int((x or {}).get('maxHp',0) or 0)-_hp(x)) if isinstance(x,dict) else 0
def _active(p):
    a=(p or {}).get('active') or []; return a[0] if a and isinstance(a[0],dict) else None
def _inplay(p): return [x for x in (p.get('active') or [])+(p.get('bench') or []) if isinstance(x,dict)]
def _count(p,cid): return sum(_cid(x)==cid for x in _inplay(p))
def _hand_ids(me): return [_cid(x) for x in (me.get('hand') or []) if isinstance(x,dict)]
def _discard_ids(me): return [_cid(x) for x in (me.get('discard') or []) if isinstance(x,dict)]
def _bench_free(me): return max(0,int(me.get('benchMax',5) or 5)-len(me.get('bench') or []))
def _stadium(obs):
    x=_cur(obs).get('stadium') or []; return _cid(x[0]) if x and isinstance(x[0],dict) else 0

def _legal(obs,a):
    s=obs.get('select') or {}; n=len(s.get('option') or []); mn=int(s.get('minCount',0) or 0); mx=int(s.get('maxCount',0) or 0)
    return isinstance(a,list) and mn<=len(a)<=mx and len(a)==len(set(a)) and all(isinstance(i,int) and 0<=i<n for i in a)

def _card_at(obs,opt):
    src,_,_=pf.resolve_source(obs,opt); return src

def _target_at(obs,opt):
    tgt,_,_=pf.resolve_target(obs,opt); return tgt

def _attached_tool(obs,opt):
    y,me,opp=_players(obs); pidx=int(opt.get('playerIndex',y) if opt.get('playerIndex') is not None else y)
    p=me if pidx==y else opp; area=int(opt.get('area',0) or 0); idx=opt.get('index'); arr=p.get('active') or [] if area==4 else p.get('bench') or []
    if not isinstance(idx,int) or idx<0 or idx>=len(arr) or not isinstance(arr[idx],dict): return None
    tools=arr[idx].get('tools') or []; ti=opt.get('toolIndex')
    return tools[ti] if isinstance(ti,int) and 0<=ti<len(tools) and isinstance(tools[ti],dict) else None

def _opp_tools(opp): return sum(len(x.get('tools') or []) for x in _inplay(opp))
def _ready_grim(me): return any(_cid(x)==GRIM and _energy(x)>=2 for x in _inplay(me))
def _boss_ko(opp): return any(0<_hp(x)<=180 for x in (opp.get('bench') or []) if isinstance(x,dict))
def _attack_available(obs): return any(int(o.get('type',-1))==13 for o in (obs.get('select') or {}).get('option') or [])

def _search_card_score(obs,card,mem,effect_id):
    y,me,opp=_players(obs); cid=_cid(card); hand=Counter(_hand_ids(me)); disc=Counter(_discard_ids(me)); plan=mem['plan']; profile=mem['profile']
    line=sum(_cid(x) in {IMPIDIMP,MORGREM,GRIM} for x in _inplay(me)); fros=_count(me,FROSLASS); munki=_count(me,MUNKIDORI)
    mature_imp=any(_cid(x)==IMPIDIMP and not x.get('appearThisTurn',False) for x in _inplay(me))
    mature_mor=any(_cid(x)==MORGREM and not x.get('appearThisTurn',False) for x in _inplay(me))
    s=0.0
    if cid==GRIM: s=80+(180 if mature_mor else 0)+(150 if mature_imp and hand[CANDY] else 0)-(80 if hand[GRIM] else 0)
    elif cid==MORGREM: s=75+(170 if mature_imp else 0)-(60 if hand[MORGREM] else 0)
    elif cid==IMPIDIMP: s=155 if line<2 and _bench_free(me)>0 else 20
    elif cid==FROSLASS: s=145 if _count(me,SNORUNT)>fros else 35
    elif cid==SNORUNT: s=125 if fros<1 and _bench_free(me)>0 else 25
    elif cid==MUNKIDORI: s=150 if munki<2 and _bench_free(me)>0 else 35
    elif cid==DARK: s=135 if any(_cid(x)==MUNKIDORI and _energy(x)==0 for x in _inplay(me)) else (115 if not _ready_grim(me) else 40)
    elif cid==STAMP: s=170 if mem.get('opp_took_prizes') or (mem['opp_hand']>=5 and mem['opp_prizes']<=4) else 55
    elif cid==CANDY: s=165 if mature_imp and hand[GRIM] else 70
    elif cid==STRETCHER: s=150 if any(disc[x] for x in (DARK,GRIM,MORGREM,IMPIDIMP,MUNKIDORI,FROSLASS)) else 55
    elif cid==SCRAPPER: s=165 if _opp_tools(opp) else 5
    elif cid==BOSS: s=180 if _boss_ko(opp) else (90 if plan=='finish' else 30)
    elif cid==STADIUM: s=135 if _stadium(obs)!=STADIUM else 20
    elif cid==POKEPAD: s=125 if (line<2 or fros<1 or munki<2) else 45
    elif cid==POFFIN: s=120 if _bench_free(me)>0 and (line<2 or fros<1) else 35
    elif cid==PETREL: s=100
    elif cid==LILLIE: s=115 if len(me.get('hand') or [])<=4 else 35
    elif cid==DAWN: s=150 if _cur(obs).get('turn',0)<=4 and line<2 else 50
    elif cid==POKEGEAR: s=75
    if plan=='damage_map' and cid in (FROSLASS,SNORUNT,MUNKIDORI,DARK): s+=45
    if plan=='recover' and cid in (GRIM,MORGREM,IMPIDIMP,CANDY,STRETCHER): s+=55
    if profile.startswith('grass') and cid in (FROSLASS,MUNKIDORI,DARK,SCRAPPER): s+=35
    return s

def _promotion_score(card,mem,me,opp):
    cid=_cid(card); e=_energy(card); hp=_hp(card); dmg=_damage(card); s=0.0
    if cid==GRIM: s=80+(90 if e>=2 else -20)
    elif cid==MORGREM: s=35+(30 if e>=2 else 0)
    elif cid==IMPIDIMP: s=22+(15 if e>=1 else 0)
    elif cid==FROSLASS: s=5
    elif cid==MUNKIDORI: s=-5-(20 if e>=1 else 0)
    if mem['plan']=='survive':
        # Preserve an engine piece or loaded attacker when a one-prize pivot can absorb the hit.
        if cid in (FROSLASS,IMPIDIMP) and e==0: s+=55
        if cid==GRIM and e>=2: s-=35
    if mem['plan']=='finish' and cid==GRIM and e>=2: s+=90
    s += max(-20,min(20,hp/10.0-dmg/15.0))
    return s

def _attach_target_score(card,mem,me,opp,is_active):
    cid=_cid(card); e=_energy(card); s=0.0
    if cid==GRIM:
        s=170 if e<2 else (25 if e==2 else -35)
        if mem['profile'].startswith('grass') and is_active and e>=2: s-=90
    elif cid==MUNKIDORI: s=145 if e==0 else -30
    elif cid==MORGREM: s=100 if e<2 else 20
    elif cid==IMPIDIMP: s=75 if e<1 else 15
    elif cid==FROSLASS: s=-20
    if mem['plan']=='damage_map' and cid==MUNKIDORI and e==0: s+=70
    if mem['plan']=='finish' and cid==GRIM and e<2: s+=80
    return s

def _damage_target_score(card,mem,attack_damage=False,bench=False):
    cid=_cid(card); hp=_hp(card); dmg=_damage(card); s=0.0
    if hp<=0: return -999
    if attack_damage and bench and cid in TERA_BENCH: return -120
    if hp<=30: s+=240
    if cid in HIGH_VALUE: s+=55
    if cid in ABILITY_ENGINE: s+=35
    if cid in (344,646,647,860,25): s+=28
    if 180<hp<=210: s+=50  # one damage-transfer activation creates a Shadow Bullet KO
    if hp<=180: s+=30
    s+=min(45,dmg/3.0)
    if mem['profile'].startswith('grass') and cid==96: s+=85
    if mem['profile']=='wall' and cid in (344,345,117): s+=60
    return s

def _remove_damage_score(card,mem):
    cid=_cid(card); d=_damage(card); s=d
    if d<10: s-=100
    if cid==GRIM: s+=50
    elif cid==MUNKIDORI: s+=25
    elif cid==FROSLASS: s+=15
    if mem['active_ko_risk'] and cid==_cid(card): s+=20
    return s

def _direct_selection(obs,mem):
    sel=obs.get('select') or {}; opts=sel.get('option') or []; ctx=int(sel.get('context',-1) if sel.get('context') is not None else -1)
    mn=int(sel.get('minCount',0) or 0); mx=min(len(opts),int(sel.get('maxCount',0) or 0))
    if not opts or mx<=0: return [] if mn==0 else None
    y,me,opp=_players(obs); effect_id=_cid(sel.get('effect')) or _cid(sel.get('contextCard'))
    scores=None
    if ctx==7: scores=[_search_card_score(obs,_card_at(obs,o),mem,effect_id) for o in opts]
    elif ctx==4: scores=[_promotion_score(_card_at(obs,o),mem,me,opp) for o in opts]
    elif ctx==21:
        scores=[]
        for o in opts:
            c=_card_at(obs,o); area=int(o.get('area',0) or 0)
            scores.append(_attach_target_score(c,mem,me,opp,area==4))
    elif ctx in (13,14): scores=[_damage_target_score(_card_at(obs,o),mem,False,int(o.get('area',0) or 0)==5) for o in opts]
    elif ctx==15: scores=[_damage_target_score(_card_at(obs,o),mem,True,int(o.get('area',0) or 0)==5) for o in opts]
    elif ctx==16: scores=[_remove_damage_score(_card_at(obs,o),mem) for o in opts]
    elif ctx==27:
        scores=[]
        for o in opts:
            t=_attached_tool(obs,o); tid=_cid(t); pidx=int(o.get('playerIndex',y) if o.get('playerIndex') is not None else y)
            scores.append((200 if pidx!=y else -30)+(100 if tid==1159 else 0))
    elif ctx==43:
        # Use beneficial optional effects, but decline empty searches or effects that merely consume a card.
        scores=[50 if int(o.get('type',-1))==1 else 0 for o in opts]
        if effect_id==SCRAPPER and _opp_tools(opp)==0: scores=[0 if int(o.get('type',-1))==2 else -50 for o in opts]
    if scores is None: return None
    order=sorted(range(len(opts)),key=lambda i:(-scores[i],i))
    k=mx
    if mn==0:
        positive=[i for i in order if scores[i]>0]
        k=min(mx,max(mn,len(positive)))
        order=positive+order
    a=sorted(order[:k])
    return a if _legal(obs,a) else None

def _main_option_score(obs,opt,mem):
    y,me,opp=_players(obs); sem=pf.semantic(obs,opt); typ=sem['type']; sid=sem['source_id']; tid=sem['target_id']; plan=mem['plan']; profile=mem['profile']; s=0.0
    if typ==13:
        s+=70
        oa=_active(opp)
        if 0<_hp(oa)<=180: s+=120
        if plan=='finish': s+=80
        if profile=='wall' and _cid(oa) in (345,117): s-=35
    elif typ==14:
        s-=75 if _attack_available(obs) else 0
        if plan in ('engine','recover','damage_map'): s-=25
    elif typ==9:
        s+=35
        if sid==GRIM: s+=100
        elif sid==FROSLASS: s+=70
        elif sid==MORGREM: s+=45
        if plan=='damage_map' and sid==FROSLASS: s+=65
    elif typ==8:
        tgt=_target_at(obs,opt); area=int(opt.get('inPlayArea',0) or 0)
        s+=_attach_target_score(tgt,mem,me,opp,area==4)
    elif typ==10:
        s+=60
        if sid==MUNKIDORI: s+=80
        if sid==GRIM: s+=90
    elif typ==12:
        s+=55 if mem['active_ko_risk'] else -20
    elif typ==7:
        card=_card_at(obs,opt); cid=_cid(card); s+=10
        if cid==STAMP: s+=130 if mem.get('opp_took_prizes') or mem['opp_hand']>=5 else 30
        elif cid==BOSS: s+=130 if _boss_ko(opp) else (-20 if plan!='finish' else 50)
        elif cid==SCRAPPER: s+=120 if _opp_tools(opp) else -100
        elif cid==STADIUM: s+=65 if _stadium(obs)!=STADIUM else -80
        elif cid==LILLIE: s+=80 if len(me.get('hand') or [])<=4 else -10
        elif cid==DAWN: s+=90 if int(_cur(obs).get('turn',0) or 0)<=4 else 10
        elif cid in (POFFIN,POKEPAD): s+=70 if _bench_free(me)>0 and plan in ('engine','damage_map','recover') else 5
        elif cid==CANDY: s+=80
        elif cid==STRETCHER: s+=70 if _discard_ids(me) else -5
        elif cid==PETREL: s+=45
        elif cid in (IMPIDIMP,SNORUNT,MUNKIDORI): s+=70 if _bench_free(me)>0 else -100
    return s


def _visible_opp_has(opp,cid):
    return any(_cid(x)==cid for x in _inplay(opp)) or any(_cid(x)==cid for x in (opp.get('discard') or []) if isinstance(x,dict))

def _poffin_setup_action(obs):
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    y,me,opp=_players(obs); mx=min(len(opts),int(sel.get('maxCount',0) or 0),_bench_free(me)); mn=int(sel.get('minCount',0) or 0)
    if mx<=0: return [] if mn==0 else None
    inplay=Counter(_cid(x) for x in _inplay(me)); hand=Counter(_hand_ids(me))
    need_imp=(inplay[IMPIDIMP]+inplay[MORGREM]+inplay[GRIM]+hand[IMPIDIMP])<2
    need_sno=(inplay[SNORUNT]+inplay[FROSLASS]+hand[SNORUNT])<1
    groups={IMPIDIMP:[],SNORUNT:[]}
    for i,o in enumerate(opts):
        cid=_cid(_card_at(obs,o))
        if cid in groups: groups[cid].append(i)
    chosen=[]
    if need_imp and groups[IMPIDIMP] and len(chosen)<mx: chosen.append(groups[IMPIDIMP][0])
    if need_sno and groups[SNORUNT] and len(chosen)<mx: chosen.append(groups[SNORUNT][0])
    for cid in (IMPIDIMP,SNORUNT):
        if len(chosen)>=mx: break
        for i in groups[cid]:
            if i not in chosen: chosen.append(i); break
    if len(chosen)<mn:return None
    a=sorted(chosen)
    return a if _legal(obs,a) else None


def _finish_active_grim_attachment(obs):
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    y,me,opp=_players(obs); active=_active(me)
    if not active or _cid(active)!=GRIM or _energy(active)!=1: return None
    if any(_cid(x)==GRIM and _energy(x)>=2 for x in _inplay(me)): return None
    for i,o in enumerate(opts):
        c=_card_at(obs,o)
        if _cid(c)==GRIM and _energy(c)==1 and int(o.get('area',0) or 0)==4:
            a=[i]
            if _legal(obs,a): return a
    return None



def _grass_attacker_pipeline_action(obs,baseline,mem):
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    if int(sel.get('context',-1) if sel.get('context') is not None else -1)!=0:return None
    if not mem.get('profile','').startswith('grass'):return None
    y,me,opp=_players(obs); ma=_active(me); oa=_active(opp)
    if int(_cur(obs).get('turn',0) or 0)>3:return None
    if _cid(ma)!=IMPIDIMP or _cid(oa)!=96 or _energy(oa)<2:return None
    bench_imp=[x for x in (me.get('bench') or []) if isinstance(x,dict) and _cid(x)==IMPIDIMP and _energy(x)<2]
    if not bench_imp or _count(me,FROSLASS)>0 or any(_damage(x)>=10 for x in _inplay(me)):return None
    # Only replace low-commitment setup actions that would otherwise lose setup tempo.
    bsem=[pf.semantic(obs,opts[i]) for i in (baseline or []) if 0<=i<len(opts)]
    if not bsem or bsem[0].get('type')!=7 or bsem[0].get('source_id') not in (POKEGEAR,POKEPAD,MUNKIDORI):return None
    cand=[]
    for i,o in enumerate(opts):
        sem=pf.semantic(obs,o); c=_target_at(obs,o)
        if sem.get('type')==8 and int(sem.get('inplay_area',0) or 0)==5 and _cid(c)==IMPIDIMP and _energy(c)<2:
            cand.append((1 if not c.get('appearThisTurn',False) else 0,-_energy(c),i,i))
    if not cand:return None
    a=[max(cand)[-1]]
    return a if _legal(obs,a) else None

SUPPORTERS={BOSS,PETREL,LILLIE,DAWN}

def _blank_or_empty(obs):
    sel=obs.get('select') or {}
    if int(sel.get('minCount',0) or 0)==0 and _legal(obs,[]): return []
    for i,o in enumerate(sel.get('option') or []):
        if _cid(_card_at(obs,o))==0:
            a=[i]
            if _legal(obs,a): return a
    return None

def _decline_unproductive_pokegear(obs,mem):
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    ctx=int(sel.get('context',-1) if sel.get('context') is not None else -1)
    effect=_cid(sel.get('effect')) or _cid(sel.get('contextCard'))
    y,me,opp=_players(obs)
    if ctx!=7 or effect!=POKEGEAR or _cid(_active(opp))!=96:return None
    visible=[_cid(_card_at(obs,o)) for o in opts]
    # A top player never takes a non-Supporter from Pokégear; preserve the whiff.
    if any(c in SUPPORTERS for c in visible):return None
    return _blank_or_empty(obs)

def _petrel_pressure_setup_search(obs,mem):
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    ctx=int(sel.get('context',-1) if sel.get('context') is not None else -1)
    effect=_cid(sel.get('effect')) or _cid(sel.get('contextCard'))
    if ctx!=7 or effect!=PETREL or not mem.get('profile','').startswith('grass'):return None
    y,me,opp=_players(obs); turn=int(_cur(obs).get('turn',0) or 0)
    if turn>3 or _bench_free(me)<2:return None
    if _cid(_active(me))!=IMPIDIMP or _count(me,FROSLASS)>0:return None
    oger=sum(_cid(x)==96 for x in _inplay(opp))
    if oger<2:return None
    # Build two one-prize bodies before exposing a Grass-weak two-prize attacker.
    cand=[i for i,o in enumerate(opts) if _cid(_card_at(obs,o))==POFFIN]
    if not cand:return None
    a=[max(cand)]
    return a if _legal(obs,a) else None

def _turn_one_overbench_guard(obs,baseline,mem):
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    if int(sel.get('context',-1) if sel.get('context') is not None else -1)!=0:return None
    if int(_cur(obs).get('turn',0) or 0)!=1 or not mem.get('profile','').startswith('grass'):return None
    y,me,opp=_players(obs); ma=_active(me); oa=_active(opp)
    if _cid(ma)!=IMPIDIMP or _cid(oa)!=96 or _energy(oa)!=0:return None
    bench=[x for x in (me.get('bench') or []) if isinstance(x,dict)]
    if len(bench)!=1 or _cid(bench[0])!=MUNKIDORI:return None
    if sum(1 for c in _hand_ids(me) if c==DARK)<3:return None
    bsem=[pf.semantic(obs,opts[i]) for i in (baseline or []) if 0<=i<len(opts)]
    if not bsem or bsem[0].get('type')!=7 or bsem[0].get('source_id')!=IMPIDIMP:return None
    for i,o in enumerate(opts):
        if pf.semantic(obs,o).get('type')==14:
            a=[i]
            if _legal(obs,a):return a
    return None

def _wall_energy_before_item(obs,baseline,mem):
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    if int(sel.get('context',-1) if sel.get('context') is not None else -1)!=0:return None
    if mem.get('profile')!='wall' or int(_cur(obs).get('turn',0) or 0)>3:return None
    y,me,opp=_players(obs)
    bsem=[pf.semantic(obs,opts[i]) for i in (baseline or []) if 0<=i<len(opts)]
    if not bsem or bsem[0].get('type')!=7:return None
    if bsem[0].get('source_id') not in (STADIUM,POKEGEAR,POKEPAD,MUNKIDORI):return None
    # Only intervene while the control matchup has not yet attached a Darkness Energy.
    if any(_energy(x)>0 for x in _inplay(me)):return None
    targets=[]
    for i,o in enumerate(opts):
        sem=pf.semantic(obs,o); t=_target_at(obs,o)
        if sem.get('type')!=8 or _cid(t) not in (IMPIDIMP,MUNKIDORI):continue
        priority=2 if _cid(t)==IMPIDIMP else 1
        targets.append((priority,i,i))
    if not targets:return None
    a=[max(targets)[-1]]
    return a if _legal(obs,a) else None

def _wall_engine_search(obs,mem):
    sel=obs.get('select') or {}; opts=sel.get('option') or []
    if int(sel.get('context',-1) if sel.get('context') is not None else -1)!=7:return None
    effect=_cid(sel.get('effect')) or _cid(sel.get('contextCard'))
    if mem.get('profile')!='wall' or effect!=POKEPAD:return None
    y,me,opp=_players(obs)
    priorities=[]
    for i,o in enumerate(opts):
        cid=_cid(_card_at(obs,o)); score=-1
        if cid==MORGREM and _count(me,IMPIDIMP)>0 and _count(me,MORGREM)==0:score=400
        elif cid==FROSLASS and _count(me,SNORUNT)>0 and _count(me,FROSLASS)==0:score=350
        elif cid==SNORUNT and _count(me,FROSLASS)==0:score=300
        elif cid==FROSLASS:score=250
        if score>=0:priorities.append((score,i,i))
    if not priorities:return None
    a=[max(priorities)[-1]]
    return a if _legal(obs,a) else None

def choose(obs,candidates):
    mem=hm.update(obs);baseline=candidates.get('baseline_route',[]);coalition=candidates.get('coalition',[])
    sel=obs.get('select') or {};ctx=int(sel.get('context',-1) if sel.get('context') is not None else -1);y,me,opp=_players(obs)
    a=_decline_unproductive_pokegear(obs,mem)
    if a is not None and _legal(obs,a):return a
    a=_grass_attacker_pipeline_action(obs,baseline,mem)
    if a is not None and _legal(obs,a):return a
    if ctx==21:
        a=_finish_active_grim_attachment(obs)
        if a is not None and _legal(obs,a):return a
    if ctx==5 and (_cid(sel.get('effect')) or _cid(sel.get('contextCard')))==POFFIN and _visible_opp_has(opp,96):
        a=_poffin_setup_action(obs)
        if a is not None and _legal(obs,a):return a
    use_counter=(mem.get('profile')=='grass_fast' and float(mem.get('confidence',0.0))>=0.45 and mem.get('plan') in ('engine','recover','damage_map','survive'))
    if use_counter and _legal(obs,coalition):return coalition
    if _legal(obs,baseline):return baseline
    for name in ('model','strategic','mirror','tempo','coalition'):
        a=candidates.get(name,[])
        if _legal(obs,a):return a
    return None

def diagnostics():return hm.diagnostics()
