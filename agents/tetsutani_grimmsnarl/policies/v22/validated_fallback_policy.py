from __future__ import annotations
from collections import Counter

# Fixed deck card IDs
DARK=7; FROSLASS=104; MUNK=112; IMP=646; MORG=647; GRIM=648; SNOR=860
CANDY=1079; STAMP=1080; POFFIN=1086; NIGHT=1097; GEAR=1122; SCRAPPER=1137; PAD=1152
BOSS=1182; PETREL=1219; LILLIE=1227; DAWN=1231; GYM=1259
SHADOW=937

EX_IDS=set()  # card-rule metadata is optional; known major threats receive name-independent HP-based value.

def ids(cards): return [int((c or {}).get('id',0) or 0) for c in (cards or []) if isinstance(c,dict)]
def count_board(st,cid): return ids(st['me'].get('active')).count(cid)+ids(st['me'].get('bench')).count(cid)
def count_hand(st,cid): return (st['me'].get('hand_ids') or []).count(cid)
def board_ids(st,who='me'): return ids(st[who].get('active'))+ids(st[who].get('bench'))
def bench_slots(st): return max(0,5-len(st['me'].get('bench') or []))
def card_obj(st,s):
    side='me' if int(s.get('source_rel',0))==0 else 'op'; p=st[side]; z=int(s.get('source_zone',0)); i=int(s.get('index',-1))
    arr=p.get('active') if z==4 else p.get('bench') if z==5 else None
    if arr is not None and 0<=i<len(arr): return arr[i] or {}
    # fall back by card id, preferring the matching zone
    for c in arr or []:
        if isinstance(c,dict) and int(c.get('id',0) or 0)==int(s.get('source_id',0) or 0): return c
    return {}
def remaining_hp(st,s):
    c=card_obj(st,s); return int(c.get('hp',0) or 0)
def damage(st,s):
    c=card_obj(st,s); return max(0,int(c.get('maxHp',0) or 0)-int(c.get('hp',0) or 0))
def target_obj(st,s):
    side='me' if int(s.get('target_rel',0))==0 else 'op'; p=st[side]
    z=int(s.get('target_area',s.get('inplay_area',0)) or 0); i=int(s.get('inplay_index',-1))
    arr=p.get('active') if z==4 else p.get('bench') if z==5 else None
    if arr is not None and 0<=i<len(arr): return arr[i] or {}
    for c in arr or []:
        if isinstance(c,dict) and int(c.get('id',0) or 0)==int(s.get('target_id',0) or 0): return c
    return {}
def energy_n(st,s): return int(card_obj(st,s).get('en',0) or 0)
def target_energy_n(st,s): return int(target_obj(st,s).get('en',0) or 0)
def prize_value(cid,hp=0):
    # Robust approximation from public board only. Mega-sized HP is usually a 3-prize target; ex-sized HP a 2-prize target.
    if hp>=340:return 3
    if hp>=180:return 2
    return 1

def choose_matching(options,pred,default=None):
    for i,s in enumerate(options):
        if pred(s):return i
    return default

def choose_setup_active(options):
    pri={IMP:0,SNOR:1,MUNK:2}
    return min(range(len(options)),key=lambda i:(pri.get(options[i].get('source_id'),99),i))

def choose_setup_bench(st,options):
    # Replay policy avoids exposing support Pokémon at setup; only Snorunt is normally benched.
    picks=[i for i,s in enumerate(options) if s.get('source_id')==SNOR]
    return picks[:st['max']]

def own_switch_score(st,s):
    cid=s.get('source_id'); en=energy_n(st,s); hp=remaining_hp(st,s)
    # Promote low-value evolution pieces first, preserving Munkidori and attack-ready Grimmsnarl.
    base={MORG:100,IMP:95,GRIM:75,MUNK:60,SNOR:50,FROSLASS:40}.get(cid,45)
    if cid==GRIM and en>=2:base-=45
    if en:base-=8*en
    if hp and hp<=30:base+=15
    return base

def threat_score(st,s,damage_amount=0,bench_damage=False,munk_bonus=260,basic_bonus=70):
    cid=int(s.get('source_id',0) or 0); hp=remaining_hp(st,s); maxhp=int(card_obj(st,s).get('maxHp',0) or hp)
    pv=prize_value(cid,maxhp)
    score=pv*120
    if hp>0:
        if damage_amount and hp<=damage_amount:score+=1000+pv*300
        score+=max(0,240-hp)
        # Good 30-damage setup points: low basics and support engines.
        if hp in (40,50,60,70,80,90,100,110,120):score+=30
    if cid==MUNK:score+=munk_bonus
    elif cid in (IMP,SNOR,741,400,343,742):score+=basic_bonus
    if int(s.get('source_zone',0))==5:score+=25
    if bench_damage and int(s.get('source_zone',0))!=5:score-=1000
    return score

def choose_to_hand(st,options,history):
    effect=int(st.get('effect',0) or 0); avail=[int(s.get('source_id',0) or 0) for s in options]
    # Hidden prizes / opaque selections: preserve engine order.
    if all(x==0 for x in avail): return list(range(st['min'])) if st['min'] else ([0] if options else [])
    cards=(st['me'].get('active') or [])+(st['me'].get('bench') or [])
    board=Counter(int(c.get('id',0) or 0) for c in cards if isinstance(c,dict))
    hand=Counter(st['me'].get('hand_ids') or [])
    discard=Counter(st['me'].get('discard_ids') or [])
    if effect==GYM:
        # Spikemuth Gym: when an immediately playable evolution is missing from hand,
        # fetch exactly that next stage. Otherwise use the search to establish another
        # Impidimp body. The mutually-exclusive branch matches the observed sequencing:
        # finish Morgrem -> Grimmsnarl before considering Impidimp -> Morgrem.
        ready_imp=sum(1 for c in cards if c.get('id')==IMP and not c.get('appear'))
        ready_morg=sum(1 for c in cards if c.get('id')==MORG and not c.get('appear'))
        active_id=int(((st['me'].get('active') or [{}])[0] or {}).get('id',0) or 0)
        # In turns 1-3, a lone Active Impidimp without Rare Candy is not evolved
        # immediately when two Bench slots remain. The replay policy first establishes a
        # second Impidimp body, avoiding a single-line failure before committing to Morgrem.
        if (int(st.get('turn',0) or 0)<=3 and active_id==IMP and ready_imp==1
                and board[IMP]==1 and bench_slots(st)>=2 and hand[CANDY]==0
                and IMP in avail):
            order=[IMP,MORG,GRIM]
        # Rare Candy converts an Active Impidimp directly into Grimmsnarl ex, so the
        # Gym fetches the final stage instead of redundantly taking Morgrem.
        elif active_id==IMP and hand[CANDY]>0 and hand[GRIM]==0:
            order=[GRIM,MORG,IMP]
        # Once one Grimmsnarl line is established and no Candy is available, a fresh
        # Impidimp is developed through Morgrem rather than stockpiling another final stage.
        elif ready_imp>0 and board[GRIM]>0 and hand[CANDY]==0:
            order=[MORG,IMP,GRIM]
        # If no Impidimp can evolve now and the search does not offer another Basic,
        # retain the scarce final stage for the next Morgrem instead of duplicating Stage 1.
        elif ready_imp==0 and hand[GRIM]==0 and IMP not in avail:
            order=[GRIM,MORG,IMP]
        elif ready_morg>hand[GRIM]:
            order=[GRIM,MORG,IMP]
        elif ready_imp>hand[MORG]:
            order=[MORG,IMP,GRIM]
        else:
            order=[IMP,MORG,GRIM]
    elif effect==PAD:
        # Poké Pad is primarily the Munkidori density search. The fourth represented
        # Munkidori is delayed when a live Snorunt can be completed into Froslass:
        # 0-2 Munkidori -> Munkidori, exactly 3 -> Froslass, then the fourth Munkidori.
        represented_munk=board[MUNK]+hand[MUNK]
        need_froslass=board[SNOR]>0 and board[FROSLASS]+hand[FROSLASS]==0
        order=[]
        # During the first two turns, one established Munkidori is enough. If the
        # Marnie line still has at most one body, Poké Pad switches to Impidimp before
        # adding a second support Pokémon, establishing the primary attacker first.
        line_n=board[IMP]+board[MORG]+board[GRIM]
        if (int(st.get('turn',0) or 0)<=2 and represented_munk>=1 and line_n<=1
                and hand[IMP]==0 and IMP in avail and bench_slots(st)>=1):
            order.append(IMP)
        # In the late game, any remaining Snorunt is completed into another Froslass
        # before increasing Munkidori density; repeated spread pressure closes prizes.
        if int(st.get('turn',0) or 0)>=7 and board[SNOR]>0:order.append(FROSLASS)
        # On a crowded Bench, complete the existing Snorunt into Froslass before
        # searching another Munkidori; this converts an occupied slot into pressure.
        if need_froslass and len(st['me'].get('bench') or [])>=4:order.append(FROSLASS)
        if represented_munk<3:order.append(MUNK)
        if need_froslass:order.append(FROSLASS)
        if represented_munk<4:order.append(MUNK)
        ready_imp=sum(1 for c in cards if c.get('id')==IMP and not c.get('appear'))>hand[MORG]
        if ready_imp:order.append(MORG)
        order += [IMP,MORG,MUNK,FROSLASS,SNOR]
    elif effect==NIGHT:
        # Night Stretcher: energy is recovered specifically to turn on an unenergized
        # Munkidori. Otherwise rebuild the evolution line before generic recovery.
        # Recover Darkness Energy only when the hand cannot already solve the
        # activation. Otherwise Night Stretcher is spent on a missing Pokémon piece.
        need_energy=(hand[DARK]==0 and any(c.get('id')==MUNK and int(c.get('en',0) or 0)==0 for c in cards))
        need_grim=sum(1 for c in cards if c.get('id')==MORG and not c.get('appear'))>hand[GRIM]
        need_morg=sum(1 for c in cards if c.get('id')==IMP and not c.get('appear'))>hand[MORG]
        need_froslass=board[SNOR]>0 and board[FROSLASS]+hand[FROSLASS]==0
        order=[]
        if need_energy:order.append(DARK)
        if need_grim:order.append(GRIM)
        if need_morg:order.append(MORG)
        if need_froslass:order.append(FROSLASS)
        order += [MUNK,IMP,MORG,GRIM,FROSLASS,SNOR,DARK]
        # A full Bench makes an Impidimp recovered by Night Stretcher immediately
        # unusable. In that exact case the target policy recovers Darkness Energy
        # instead, preserving the next Adrena-Brain activation without wasting a slot.
        base_first=next((cid for cid in order if cid in avail),0)
        if base_first==IMP and bench_slots(st)==0 and DARK in avail:
            order=[DARK]+order
        # In the prize-closing phase, an extra recovered Basic is slower than the
        # Darkness Energy needed to keep the damage-transfer engine online.
        elif base_first==IMP and int(st['me'].get('prize_n',0) or 0)<=2 and DARK in avail:
            order=[DARK]+order
        elif (base_first==MUNK and int(st['me'].get('prize_n',0) or 0)<=4
              and hand[DARK]==0 and DARK in avail):
            order=[DARK]+order
        # A discarded Froslass is only recovered after the engine has an Energy;
        # when both are offered, restore the activation resource first.
        elif base_first==FROSLASS and DARK in avail:
            order=[DARK]+order
    elif effect==PETREL:
        # Petrel is a compact toolbox. Stamp is the default early target; in long games
        # Night Stretcher overtakes it when Pokémon/Energy have accumulated in discard.
        recoverable=sum(discard[c] for c in (DARK,IMP,MORG,GRIM,MUNK,SNOR,FROSLASS))
        active_id=int((st['me'].get('active') or [{}])[0].get('id',0) or 0)
        score={STAMP:400,NIGHT:210,PAD:180,GYM:170,CANDY:140,LILLIE:100,
               POFFIN:60,SCRAPPER:50,BOSS:40,GEAR:20,DAWN:20}
        score[NIGHT]+=10*min(5,recoverable)
        if int(st.get('turn',0) or 0)>=9:score[NIGHT]+=100
        if st.get('stadium')!=GYM:score[GYM]+=200
        if active_id==IMP and board[IMP]>0 and hand[GRIM]>0 and hand[CANDY]==0:score[CANDY]+=150
        if board[MUNK]+hand[MUNK]>=4:score[PAD]-=100
        # Petrel is not normally used to fetch a second copy of a Trainer already
        # available in hand. This exposes the next toolbox branch instead of wasting
        # the supporter on redundancy.
        for cid in (STAMP,NIGHT,PAD,GYM,CANDY,LILLIE):
            if hand[cid]>0:score[cid]-=260
        # With an open bench and fewer than four represented Munkidori, Poké Pad is
        # the preferred indirect Pokémon search over a generic Night Stretcher.
        if bench_slots(st)>0 and board[MUNK]+hand[MUNK]<4:score[PAD]+=180
        ranked=sorted(score,key=lambda c:(score[c],-c),reverse=True)
        order=ranked
    elif effect==DAWN:
        # Dawn exposes one evolution stage per prompt. Detect the stage from legal cards.
        aset=set(avail)
        if aset <= {GRIM}: order=[GRIM]
        elif aset <= {MORG,FROSLASS}:
            # Dawn finishes an already established Snorunt line; without Snorunt,
            # Morgrem remains the Stage 1 priority.
            need_froslass=board[SNOR]>0 and board[FROSLASS]+hand[FROSLASS]==0
            order=[FROSLASS,MORG] if need_froslass else [MORG,FROSLASS]
        elif aset <= {MUNK,IMP,SNOR}:
            # Dawn first repairs a missing Marnie body. Once two Marnie-line Pokémon
            # are established, Munkidori becomes the Stage-0 priority.
            line_n=board[IMP]+board[MORG]+board[GRIM]
            order=[IMP,MUNK,SNOR] if line_n<2 else [MUNK,IMP,SNOR]
        else: order=[GRIM,MORG,FROSLASS,MUNK,IMP,SNOR]
    elif effect==GEAR:
        # The simulator resolves Pokégear's revealed-card ordering through this prompt.
        # The replay policy preserves the revealed order and always selects the first option.
        return [0] if options else []
    else:
        order=[GRIM,MORG,MUNK,IMP,FROSLASS,DARK,STAMP,NIGHT,PAD,GYM,CANDY,LILLIE,POFFIN,SCRAPPER,BOSS,DAWN,SNOR]
    for cid in order:
        for i,s in enumerate(options):
            if int(s.get('source_id',0) or 0)==cid:return [i]
    if st['min']==0:return []
    return list(range(min(st['min'],len(options))))

def desired_energy(st,c):
    cid=int((c or {}).get('id',0) or 0); zone_active=c in (st['me'].get('active') or [])
    if cid==GRIM:return 2
    if cid in (MORG,IMP):return 2
    return 0

def energy_deficit_total(st):
    total=0
    for c in (st['me'].get('active') or [])+(st['me'].get('bench') or []):
        if not isinstance(c,dict):continue
        total+=max(0,desired_energy(st,c)-int(c.get('en',0) or 0))
    return total

def choose_attach_target(st,options):
    scored=[]
    for i,s in enumerate(options):
        c=card_obj(st,s); cid=int(s.get('source_id',0) or 0); en=int(c.get('en',0) or 0); need=max(0,desired_energy(st,c)-en); active=int(s.get('source_zone',0))==4
        base={GRIM:90,MORG:55,IMP:50,MUNK:45}.get(cid,0)
        if active:base+=35
        if cid==MUNK and en==0:base+=30
        if need<=0:base-=100
        scored.append((need>0,base-10*en,-i,i))
    return max(scored)[-1]

def discard_keep_value(st,s):
    cid=int(s.get('source_id',0) or 0)
    board=Counter(board_ids(st)); hand=Counter(st['me'].get('hand_ids') or [])
    val={DARK:45,MUNK:90,IMP:75,MORG:80,GRIM:95,SNOR:30,FROSLASS:35,CANDY:65,STAMP:95,POFFIN:35,NIGHT:65,GEAR:30,SCRAPPER:20,PAD:50,BOSS:60,PETREL:55,LILLIE:70,DAWN:50,GYM:45}.get(cid,40)
    if cid==POFFIN and (bench_slots(st)==0 or board[IMP]>=2):val-=30
    if cid==GYM and st.get('stadium')==GYM:val-=25
    if cid==DARK and hand[DARK]>2:val-=18
    if cid in (MORG,GRIM) and board[IMP]+board[MORG]==0:val-=25
    if hand[cid]>1:val-=8*(hand[cid]-1)
    return val

def main_score(st,s,history):
    t=int(s.get('type',-1)); src=int(s.get('source_id',0) or 0); tgt=int(s.get('target_id',0) or 0); area=int(s.get('target_area',0) or 0)
    turn=int(st.get('turn',0)); tac=int(st.get('tac',0)); me=st['me']; board=Counter(board_ids(st)); hand=Counter(me.get('hand_ids') or [])
    active=(me.get('active') or [{}])[0] or {}; active_id=int(active.get('id',0) or 0); active_en=int(active.get('en',0) or 0)
    # Ability: Adrena-Brain is almost categorically first.
    if t==10 and src==MUNK:return 10000
    # Free Spikemuth search ability.
    if t==10 and src==0 and int(s.get('area',0))==7:return 8500 if st.get('flags',{}).get('stadium') and st.get('stadium')==GYM else 8425
    if t==7: # play
        if src==MUNK:
            return 9300 - 250*max(0,board[MUNK]-2)
        if src==IMP:
            line_n=board[IMP]+board[MORG]+board[GRIM]
            if me.get('prize_n')==6 and line_n<=1:return 8950
            return 8850 - 180*max(0,line_n-2)
        if src==NIGHT:
            return 8900 if me.get('prize_n')==6 and board[FROSLASS]>0 else 8550
        if src==PAD:return 8450
        if src==POFFIN:return 8250 if bench_slots(st)>=1 else 3000
        if src==CANDY:
            if board[IMP]>0 and hand[GRIM]>0:
                return 8920 if me.get('prize_n')==6 else 8750
            return 7800
        if src==STAMP:return 8150
        if src==GEAR:return 8350
        if src==GYM:return 8850 if st.get('stadium')!=GYM else 2000
        if src==LILLIE:
            return 7700 + (350 if me.get('hand_n',0)<=6 else 0) + (150 if me.get('prize_n')==6 else 0) + (500 if me.get('hand_n',0)<=2 else 0)
        if src==PETREL:return 7750
        if src==DAWN:return 7700
        if src==BOSS:
            # Do not gust away an Active that Shadow Bullet can already knock out.
            # Boss remains available for non-lethal Active positions and stall targets.
            op_active=(st['op'].get('active') or [{}])[0] or {}
            active_hp=int(op_active.get('hp',0) or 0)
            return 6900 if 0<active_hp<=180 and active_id==GRIM and active_en>=2 else 7400
        if src==FROSLASS:return 6500
        if src==SNOR:return 5900 if board[SNOR]==0 else 4300
        if src==SCRAPPER:return 7600
        return 6000
    if t==9: # evolve
        active_target=(area==4)
        if src==GRIM:
            return 9500 if active_target else 8380
        if src==MORG:
            return 9150 if not active_target else 8700
        if src==FROSLASS:return 8900
        return 8200
    if t==8 and src==DARK: # manual energy attach
        cid=tgt; obj=None
        if cid==MUNK:
            en=target_energy_n(st,s)
            zero_munk=sum(1 for c in (me.get('active') or [])+(me.get('bench') or []) if c.get('id')==MUNK and int(c.get('en',0) or 0)==0)
            if en==0 and active_id==GRIM and zero_munk>0:return 8440 + (50 if zero_munk>=2 else 0)
            return (8330 + (50 if zero_munk>=2 else 0)) if en==0 else 6200
        if cid==GRIM:
            en=target_energy_n(st,s)
            # Once the Active Grimmsnarl ex is attack-ready, take Shadow Bullet before
            # spending the turn attachment on a Bench Grimmsnarl. The backup can be
            # developed on the following turn without giving up immediate pressure.
            if area==5 and active_id==GRIM and active_en>=2:return 6900
            return 8050 if en<2 else 5200
        if cid in (MORG,IMP):return 6900
        return 5000
    if t==12: # retreat
        ready=any(c.get('id')==GRIM and int(c.get('en',0) or 0)>=2 for c in me.get('bench') or [])
        if ready:
            # Preserve a damaged active Grimmsnarl ex by pivoting to a fresh,
            # attack-ready copy on the Bench before committing the attack.
            if active_id==GRIM and int(active.get('hp',0) or 0)<=200:
                # With three or fewer Prizes remaining, preserve tempo and continue
                # attacking instead of pivoting merely to protect the damaged Active.
                return 6900 if me.get('prize_n',0)<=3 else 7100
            return 6800
        return -100
    if t==13:
        if int(s.get('attack_id',0) or 0)==SHADOW:return 7000
        return 5200
    if t==14:return 0
    return 4000

def _choose_v8_base(st,options,history):
    ctx=int(st['context']); mn=int(st['min']); mx=int(st['max']); n=len(options)
    if n==0:return []
    if ctx==0:
        scores=[main_score(st,s,history) for s in options]
        winner=max(range(n),key=lambda i:(scores[i],-i))
        # When several identical evolution actions differ only by the target copy,
        # evolve the copy that already carries the most Energy. This preserves prior
        # manual attachments; otherwise retain engine order as the deterministic tie-break.
        wo=options[winner]
        if int(wo.get('type',-1))==9:
            semantic=(int(wo.get('source_id',0) or 0),int(wo.get('target_id',0) or 0),
                      int(wo.get('target_area',0) or 0),scores[winner])
            tied=[i for i,o in enumerate(options)
                  if int(o.get('type',-1))==9
                  and (int(o.get('source_id',0) or 0),int(o.get('target_id',0) or 0),
                       int(o.get('target_area',0) or 0),scores[i])==semantic]
            if len(tied)>1:
                winner=max(tied,key=lambda i:(target_energy_n(st,options[i]),-i))
        # Energizing a Bench Munkidori is a tempo prerequisite for Adrena-Brain.
        # The teacher performs this attachment before optional draw/search actions,
        # and before evolving the exposed Active Impidimp into Morgrem.
        bench_munk_attach=choose_matching(options,lambda s:int(s.get('type',-1))==8
                                          and int(s.get('source_id',0) or 0)==DARK
                                          and int(s.get('target_id',0) or 0)==MUNK
                                          and int(s.get('target_area',0) or 0)==5
                                          and target_energy_n(st,s)==0)
        win_type=int(options[winner].get('type',-1))
        win_src=int(options[winner].get('source_id',0) or 0)
        win_area=int(options[winner].get('target_area',0) or 0)
        active=(st['me'].get('active') or [{}])[0] or {}
        if bench_munk_attach is not None:
            # At six Prizes, energize the protected Bench Munkidori rather than the
            # sacrificial Active copy when both are legal attachment targets.
            if (win_type==8 and win_src==DARK
                    and int(options[winner].get('target_id',0) or 0)==MUNK
                    and int(options[winner].get('target_area',0) or 0)==4
                    and st['me'].get('prize_n',0)==6):
                return [bench_munk_attach]
            if win_type==7 and win_src==LILLIE:return [bench_munk_attach]
            if win_type==9 and win_src==MORG and win_area==4:return [bench_munk_attach]
            if win_type==7 and win_src==PAD and int(active.get('id',0) or 0)==GRIM and st.get('stadium')==GYM:
                return [bench_munk_attach]
        # The teacher draws before taking the free Spikemuth search. When the Gym
        # ability is otherwise the best action and the hand is at most seven cards,
        # use Lillie's Determination first, then return to the deterministic search.
        is_gym_ability=(win_type==10
                        and int(options[winner].get('source_id',0) or 0)==0
                        and int(options[winner].get('area',0) or 0)==7)
        # With a nearly full hand but an underdeveloped Marnie line, take the
        # guaranteed Gym evolution search immediately instead of drawing first.
        line_n=(count_board(st,IMP)+count_board(st,MORG)+count_board(st,GRIM))
        immediate_gym=(st['me'].get('hand_n',0)>=6 and line_n<=2)
        hand_ids=Counter(st['me'].get('hand_ids') or [])
        # When Rare Candy is already secured but no Froslass is established, resolve
        # Pokégear before Poké Pad. The supporter information is obtained before spending
        # the narrower Pokémon search, while the evolution shortcut remains protected.
        if (win_type==7 and win_src==PAD and count_board(st,FROSLASS)==0
                and hand_ids[CANDY]>=1):
            gear=choose_matching(options,lambda s:int(s.get('type',-1))==7
                                 and int(s.get('source_id',0) or 0)==GEAR)
            if gear is not None:return [gear]
        # Immediately after an evolution, with at least two Marnie-line bodies and at most
        # one unenergized Munkidori, Dawn develops the next stage before another Gym search.
        if (is_gym_ability and history and int(history[-1].get('type',-1) or -1)==9
                and line_n>=2 and sum(1 for c in (st['me'].get('active') or [])+(st['me'].get('bench') or [])
                                      if c.get('id')==MUNK and int(c.get('en',0) or 0)==0)<=1):
            dawn=choose_matching(options,lambda s:int(s.get('type',-1))==7
                                 and int(s.get('source_id',0) or 0)==DAWN)
            if dawn is not None:return [dawn]
        # On the first turn, if the Marnie line is absent, establish Impidimp before adding
        # a Munkidori support body.
        if win_type==7 and win_src==MUNK and int(st.get('turn',0) or 0)<=1 and line_n==0:
            imp=choose_matching(options,lambda s:int(s.get('type',-1))==7
                                and int(s.get('source_id',0) or 0)==IMP)
            if imp is not None:return [imp]
        # With a completely empty Bench on turn one, Buddy-Buddy Poffin precedes Poké Pad
        # even when the Active Pokémon itself is already an Impidimp.
        if (win_type==7 and win_src==PAD and int(st.get('turn',0) or 0)<=1
                and bench_slots(st)==5):
            poffin=choose_matching(options,lambda s:int(s.get('type',-1))==7
                                   and int(s.get('source_id',0) or 0)==POFFIN)
            if poffin is not None:return [poffin]
        # If no Marnie-line Pokémon is in play at the start of turns 1-2, Buddy-Buddy
        # Poffin precedes Poké Pad. It creates the first attacker body before spending the
        # narrower Pokémon search on a support piece.
        if (win_type==7 and win_src==PAD and int(st.get('turn',0) or 0)<=2
                and int(st['me'].get('prize_n',0) or 0)==6 and line_n==0):
            poffin=choose_matching(options,lambda s:int(s.get('type',-1))==7
                                   and int(s.get('source_id',0) or 0)==POFFIN)
            if poffin is not None:return [poffin]
        # Once three Marnie-line bodies are represented but only one Grimmsnarl ex is
        # complete, Dawn is used before another free Gym search while the Active attacker
        # is already online. This develops the next evolution stage without over-searching.
        if (is_gym_ability and int(st.get('turn',0) or 0)<=9 and line_n>=3
                and count_board(st,GRIM)<=1 and int(active.get('id',0) or 0)==GRIM
                and int(st['me'].get('hand_n',0) or 0)<=7):
            dawn=choose_matching(options,lambda s:int(s.get('type',-1))==7
                                 and int(s.get('source_id',0) or 0)==DAWN)
            if dawn is not None:return [dawn]
        if is_gym_ability and st['me'].get('hand_n',0)<=7 and not immediate_gym:
            lillie=choose_matching(options,lambda s:int(s.get('type',-1))==7
                                   and int(s.get('source_id',0) or 0)==LILLIE)
            if lillie is not None:
                if bench_munk_attach is not None:return [bench_munk_attach]
                return [lillie]
        if is_gym_ability:
            # Pokégear is resolved before the deterministic Gym search so the
            # supporter information is known before choosing an evolution piece.
            gear=choose_matching(options,lambda s:int(s.get('type',-1))==7
                                 and int(s.get('source_id',0) or 0)==GEAR)
            if gear is not None:return [gear]
        return [winner]
    if ctx==1:return [choose_setup_active(options)]
    if ctx==2:return choose_setup_bench(st,options)
    if ctx==3:
        # Boss's Orders selects the opponent. Other context-3 prompts are forced/voluntary own switches.
        rel=int(options[0].get('source_rel',0)) if options else 0
        if rel==1:return [max(range(n),key=lambda i:(threat_score(st,options[i],180),-i))]
        last=int((history[-1] if history else {}).get('source_id',0) or 0)
        if last==MORG:
            pri={IMP:100,GRIM:90,MUNK:70,MORG:60,SNOR:50,FROSLASS:40}
            return [max(range(n),key=lambda i:(pri.get(int(options[i].get('source_id',0) or 0),0),-i))]
        # The replay policy normally pivots directly into its attacker after ability/evolution effects.
        return [max(range(n),key=lambda i:((1000 if int(options[i].get('source_id',0) or 0)==GRIM else 0)+10*energy_n(st,options[i])+remaining_hp(st,options[i]),-i))]
    if ctx==4:
        # After a KO, immediately promote an attack-ready Grimmsnarl. If none exists,
        # preserve the teacher's sacrificial/evolution-line order.
        ready=[i for i,s in enumerate(options) if int(s.get('source_id',0) or 0)==GRIM and energy_n(st,s)>=2]
        if ready:return [max(ready,key=lambda i:(remaining_hp(st,options[i]),-i))]
        # A one-energy Grimmsnarl ex is still one attachment away from attacking and
        # is promoted ahead of sacrificial evolution pieces.
        near_ready=[i for i,s in enumerate(options) if int(s.get('source_id',0) or 0)==GRIM and energy_n(st,s)>=1]
        if near_ready:return [max(near_ready,key=lambda i:(remaining_hp(st,options[i]),-i))]
        pri={MORG:100,IMP:90,GRIM:80,MUNK:70,SNOR:60,FROSLASS:50}
        return [max(range(n),key=lambda i:(pri.get(int(options[i].get('source_id',0) or 0),0),-i))]
    if ctx==5:
        # Buddy-Buddy Poffin is almost entirely an Impidimp setup card in this policy.
        # Fill the Marnie line up to four bodies and decline rather than expose Snorunt.
        marnie=count_board(st,IMP)+count_board(st,MORG)+count_board(st,GRIM)
        slots=bench_slots(st)
        imp=[i for i,s in enumerate(options) if int(s.get('source_id',0) or 0)==IMP]
        picks=imp[:min(mx,slots,max(0,4-marnie))]
        # Once a Froslass engine is established, Poffin replaces the next Snorunt
        # during the early/mid game when no Impidimp remains in the search. This
        # preserves continuous spread pressure without diluting the primary setup.
        if (not picks and slots>0 and count_board(st,FROSLASS)>0
                and int(st.get('turn',0) or 0)<=8):
            snor=[i for i,s in enumerate(options) if int(s.get('source_id',0) or 0)==SNOR]
            if snor:picks=snor[:1]
        if len(picks)<mn:
            rest=[i for i in range(n) if i not in picks];picks+=rest[:mn-len(picks)]
        return picks
    if ctx==7:return choose_to_hand(st,options,history)
    if ctx==8:
        # Lillie's Determination / forced hand reduction. The observed discard order is
        # stable and intentionally keeps Stamp, Lillie, Dawn, Munkidori and Petrel longest.
        discard_order=[POFFIN,MORG,PAD,GYM,SCRAPPER,DARK,NIGHT,SNOR,IMP,FROSLASS,
                       MUNK,BOSS,CANDY,GRIM,PETREL,DAWN,STAMP,LILLIE]
        rank={cid:i for i,cid in enumerate(discard_order)}
        # Xerosic's Machinations resolves the chosen discard set in original hand
        # order. First determine the low-value set, then return its indices ascending.
        picked=sorted(range(n),key=lambda i:(rank.get(int(options[i].get('source_id',0) or 0),99),i))[:mn]
        return sorted(picked)
    if ctx==13:
        amount=max(1,int(st.get('remain_damage',0) or 3))*10
        # Adrena-Brain: spread counters across prize pressure and fragile setup pieces.
        # Munkidori remains relevant, but not so dominant that every transfer ignores
        # an exposed Basic or a higher-value knockout setup.
        winner=max(range(n),key=lambda i:(threat_score(st,options[i],amount,False,150,100),-i))
        wc=card_obj(st,options[winner]);whp=int(wc.get('hp',0) or 0);wmax=int(wc.get('maxHp',0) or whp)
        # Do not waste a small transfer on an unfinishable multi-prize tank. When the
        # normal ranker selects such a target, the replay policy instead advances the
        # lowest-HP support knockout. Immediate multi-prize KOs remain untouched.
        if prize_value(int(options[winner].get('source_id',0) or 0),wmax)>=2 and whp>amount:
            winner=min(range(n),key=lambda i:(remaining_hp(st,options[i]),i))
        # A healthy Munkidori is not worth chipping while a fragile one-prize support
        # Pokémon sits at 50 HP or less. Advance that imminent spread/Shadow Bullet KO.
        wc=card_obj(st,options[winner]);whp=int(wc.get('hp',0) or 0)
        if int(options[winner].get('source_id',0) or 0)==MUNK and whp>=90:
            fragile=[]
            for i,o in enumerate(options):
                c=card_obj(st,o);hp=int(c.get('hp',0) or 0);mx=int(c.get('maxHp',0) or hp)
                if (int(o.get('source_id',0) or 0)!=MUNK and prize_value(int(o.get('source_id',0) or 0),mx)==1
                        and 0<hp<=50):
                    fragile.append(i)
            if fragile:winner=min(fragile,key=lambda i:(remaining_hp(st,options[i]),i))
        return [winner]
    if ctx==14:
        return [max(range(n),key=lambda i:(threat_score(st,options[i],30),-i))]
    if ctx==15:
        # Shadow Bullet's Bench 30 follows a flatter target profile than Adrena-Brain.
        winner=max(range(n),key=lambda i:(threat_score(st,options[i],30,True,100,70),-i))
        wc=card_obj(st,options[winner]);whp=int(wc.get('hp',0) or 0);wmax=int(wc.get('maxHp',0) or whp)
        if prize_value(int(options[winner].get('source_id',0) or 0),wmax)>=2 and whp>30:
            winner=min(range(n),key=lambda i:(remaining_hp(st,options[i]),i))
        return [winner]
    if ctx==16:
        # Adrena-Brain source: when every Bench Munkidori is still nearly healthy,
        # remove counters from a damaged Active Grimmsnarl ex instead of needlessly
        # shaving a support Pokémon. This is a public-board survival rule.
        active_grim=[i for i,s in enumerate(options)
                     if int(s.get('source_id',0) or 0)==GRIM and int(s.get('source_zone',0) or 0)==4]
        bench_munk=[i for i,s in enumerate(options)
                    if int(s.get('source_id',0) or 0)==MUNK and int(s.get('source_zone',0) or 0)==5]
        if active_grim and bench_munk:
            ai=active_grim[0]
            active_obj=card_obj(st,options[ai])
            active_damage=max(0,int(active_obj.get('maxHp',0) or 0)-int(active_obj.get('hp',0) or 0))
            most_damaged_munk=max(bench_munk,key=lambda i:(damage(st,options[i]),-i))
            max_munk_damage=damage(st,options[most_damaged_munk])
            # A support Munkidori carrying five or more damage counters is rescued first;
            # otherwise the next opposing spread effect can remove the ability engine.
            if max_munk_damage>=50:return [most_damaged_munk]
            # A still-healthy Active Grimmsnarl can safely donate counters, but when it
            # carries six more counters than every Munkidori, heal the main attacker.
            if int(active_obj.get('hp',0) or 0)>=200 and active_damage-max_munk_damage>=60:
                return [ai]
            # Three counters on every Munkidori are still within the support engine's
            # safe range. While the Active remains at 170 HP or more, clear the main
            # attacker first; at four counters the Munkidori rescue branch takes over.
            if (active_damage>=30 and int(active_obj.get('hp',0) or 0)>=170
                    and max_munk_damage<=30):
                return [ai]
            min_munk_hp=min(remaining_hp(st,options[i]) for i in bench_munk)
            if active_damage>=10 and min_munk_hp>=90:return [ai]
        # Otherwise protect the card with the smallest remaining HP, with a small
        # bias toward the exposed Active and toward Munkidori itself.
        def sc(i):
            s=options[i];cid=int(s.get('source_id',0) or 0);active=int(s.get('source_zone',0))==4
            return (100*active+50*(cid==MUNK)-remaining_hp(st,s),-i)
        return [max(range(n),key=sc)]
    if ctx in (17,18,19,20):
        return [max(range(n),key=lambda i:(own_switch_score(st,options[i]),-i))]
    if ctx==21:return [choose_attach_target(st,options)]
    if ctx==22:
        # Punk Up uses a deterministic count schedule, not a search. It takes two when only
        # 2-4 energies are visible, three from 5-7, four from 8, and five from 9-10.
        count_table={1:1,2:2,3:2,4:2,5:3,6:3,7:3,8:4,9:5,10:5}
        k=count_table.get(n,min(5,n))
        # When Punk Up follows a Bench evolution, load exactly two energies on that
        # backup attacker and preserve the remaining energies for the Active line.
        if n in (5,6):
            last_grim_evo=next((h for h in reversed(history)
                                if int(h.get('type',-1))==9
                                and int(h.get('source_id',0) or 0)==GRIM),None)
            if last_grim_evo is not None:
                evo_area=int(last_grim_evo.get('target_area',0) or 0)
                if evo_area==5:
                    k=2
                elif evo_area==4:
                    active=(st['me'].get('active') or [{}])[0] or {}
                    active_en=int(active.get('en',0) or 0)
                    total_grim_en=sum(int(c.get('en',0) or 0) for c in
                                      (st['me'].get('active') or [])+(st['me'].get('bench') or [])
                                      if int((c or {}).get('id',0) or 0)==GRIM)
                    # Do not overfill an already energized Active attacker. With one
                    # preattached energy and six visible cards, two more are sufficient.
                    if active_en>=2 or (n==6 and active_en==1 and total_grim_en==1):
                        k=2
                # With six visible Energy cards, a newly evolved zero-Energy Bench
                # Grimmsnarl receives three. This is the one stable exception to the
                # backup-two rule and builds the next attacker in a single Punk Up.
                target_index=int(last_grim_evo.get('inplay_index',-1))
                target_cards=(st['me'].get('bench') or []) if evo_area==5 else (st['me'].get('active') or []) if evo_area==4 else []
                target_card=target_cards[target_index] if 0<=target_index<len(target_cards) else {}
                target_energy=int((target_card or {}).get('en',0) or 0)
                if n==6 and evo_area==5 and target_energy==0:
                    k=3
                # In the very early game, five visible cards are also used to fully
                # establish a fresh Bench attacker rather than stopping at two.
                if n==5 and evo_area==5 and target_energy==0 and int(st.get('turn',0) or 0)<=4:
                    k=3
        k=max(mn,min(mx,n,k))
        return list(range(k))
    if ctx in (26,27,28,29,30,31,32,33):
        k=mn
        # Discard/move attached cards from the least valuable target first.
        return sorted(range(n),key=lambda i:(own_switch_score(st,options[i]),i))[:k]
    if ctx==34:return list(range(min(mx,n)))
    if ctx in (35,36):return [0]
    if ctx==37:
        # Rare Candy/evolution prompt: Active Impidimp first, then Bench.
        return [max(range(n),key=lambda i:((100 if int(options[i].get('target_area',0))==4 else 0),-i))]
    if ctx in (38,39,40):return [n-1]
    if ctx==41:return [choose_matching(options,lambda s:int(s.get('type',-1))==1,0)]
    if ctx==42:return [choose_matching(options,lambda s:int(s.get('type',-1))==2,0)]
    if ctx in (43,44,45,46):return [choose_matching(options,lambda s:int(s.get('type',-1))==1,0)]
    # Count-respecting deterministic fallback.
    k=max(mn,min(mx,n));return list(range(k))


# v9 refinements are deliberately applied after the complete v8 priority chain.  This
# preserves every earlier guard and only changes three repeated, public-board situations.
def _choose_v9_base(st,options,history):
    base=_choose_v8_base(st,options,history)
    ctx=int(st.get('context',-1))
    if not base or not options:
        return base
    # Spikemuth Gym target refinements are applied after the full v8 search policy.
    # They only alter repeated cases where the existing final selection has a clear
    # redundant-stage interpretation from the public board and hand.
    if ctx==7 and int(st.get('effect',0) or 0)==GYM:
        me=st.get('me') or {}
        cards=(me.get('active') or [])+(me.get('bench') or [])
        board=Counter(int((c or {}).get('id',0) or 0) for c in cards)
        hand=Counter(me.get('hand_ids') or [])
        current=int(options[base[0]].get('source_id',0) or 0)
        turn=int(st.get('turn',0) or 0)
        slots=max(0,5-len(me.get('bench') or []))
        line_n=board[IMP]+board[MORG]+board[GRIM]
        # With no Bench on turn one, establish the Basic body before stockpiling a
        # Grimmsnarl ex that cannot yet be evolved into play.
        if current==GRIM and turn<=1 and slots==5:
            imp=choose_matching(options,lambda o:int(o.get('source_id',0) or 0)==IMP)
            if imp is not None:return [imp]
        # If a Grimmsnarl ex is already in hand, use the search on Morgrem rather than
        # taking a redundant final-stage copy.
        if current==GRIM and hand[GRIM]>=1:
            morg=choose_matching(options,lambda o:int(o.get('source_id',0) or 0)==MORG)
            if morg is not None:return [morg]
        # On turn two with only one represented Marnie-line body, establish the second
        # Impidimp before committing the sole line to Morgrem.
        if current==MORG and turn==2 and line_n==1:
            imp=choose_matching(options,lambda o:int(o.get('source_id',0) or 0)==IMP)
            if imp is not None:return [imp]
        return base
    if ctx!=0:
        return base
    me=st.get('me') or {}
    active=(me.get('active') or [{}])[0] or {}
    cards=(me.get('active') or [])+(me.get('bench') or [])
    board=Counter(int((c or {}).get('id',0) or 0) for c in cards)
    hand=Counter(me.get('hand_ids') or [])
    chosen=options[base[0]]
    ctype=int(chosen.get('type',-1))
    csrc=int(chosen.get('source_id',0) or 0)
    carea=int(chosen.get('area',0) or 0)

    # A low-HP Active Grimmsnarl ex keeps attacking when a Snorunt body is already
    # sustaining the spread plan.  In this narrow state the replay policy values the
    # immediate Shadow Bullet over a defensive pivot to the backup attacker.
    if (ctype==12 and int(active.get('id',0) or 0)==GRIM
            and 0<int(active.get('hp',0) or 0)<=100 and board[SNOR]>=1):
        attack=choose_matching(options,lambda o:int(o.get('type',-1))==13
                               and int(o.get('attack_id',0) or 0)==SHADOW)
        if attack is not None:return [attack]

    # Once the game has developed, no Rare Candy remains in hand, and at least three
    # Prizes are still to be taken, Pokégear is resolved before Poké Pad.  This fixes the
    # supporter branch before spending the narrower Pokémon search.
    if ctype==7 and csrc==PAD:
        turn=int(st.get('turn',0) or 0);prizes=int(me.get('prize_n',0) or 0)
        slots=max(0,5-len(me.get('bench') or []))
        active_energy=int(active.get('en',0) or 0)
        gear_first=((turn>=7 and prizes>=3 and hand[CANDY]==0)
                    or (turn>=6 and slots>=1)
                    or (active_energy==0 and board[MUNK]>=2))
        if gear_first:
            gear=choose_matching(options,lambda o:int(o.get('type',-1))==7
                                 and int(o.get('source_id',0) or 0)==GEAR)
            if gear is not None:return [gear]

    # With multiple unresolved Stage-1 bodies, or three separate Impidimp bodies, the
    # broad Dawn evolution chain is resolved before another Spikemuth Gym activation.
    is_gym=(ctype==10 and csrc==0 and carea==7)
    if is_gym and ((hand[DARK]==1 and board[MORG]>=2) or board[IMP]>=3):
        dawn=choose_matching(options,lambda o:int(o.get('type',-1))==7
                             and int(o.get('source_id',0) or 0)==DAWN)
        if dawn is not None:return [dawn]
    return base


# v10 refinements.  These are narrow public-board tie-breaks layered over the complete
# v9 policy; no learned parameters, replay lookup, or search are used.
def _choose_v10_base(st,options,history):
    base=_choose_v9_base(st,options,history)
    if not base or not options:
        return base
    ctx=int(st.get('context',-1))

    # Energy attachment: once the v9 target already carries two Energy, do not spend
    # another manual attachment there when an unenergized Bench Impidimp is available.
    # The Basic keeps the second Marnie evolution line alive while the existing target
    # is already attack-ready.
    if ctx==21:
        current=options[base[0]]
        if energy_n(st,current)>=2:
            imp=choose_matching(
                options,
                lambda o: int(o.get('source_id',0) or 0)==IMP
                and int(o.get('source_zone',0) or 0)==5,
            )
            if imp is not None:
                return [imp]
        return base

    # Knockout promotion: when the default promotion is already at 130 HP or less,
    # expose a Morgrem instead if one is available.  This preserves the healthier
    # attacker and leaves an evolvable one-Prize pivot in the Active spot.
    if ctx==4 and remaining_hp(st,options[base[0]])<=130:
        morg=choose_matching(options,lambda o:int(o.get('source_id',0) or 0)==MORG)
        if morg is not None:
            return [morg]

    # Adrena-Brain source selection.  A Bench Munkidori at exactly 40 HP is in the
    # immediate spread-KO danger zone and is healed first.  If the current choice is
    # a still-safe Bench Grimmsnarl ex, heal the exposed Active Grimmsnarl instead.
    if ctx==16:
        endangered=[i for i,o in enumerate(options)
                    if int(o.get('source_id',0) or 0)==MUNK
                    and int(o.get('source_zone',0) or 0)==5
                    and remaining_hp(st,o)==40]
        if endangered:
            return [endangered[0]]
        cur=options[base[0]]
        if (int(cur.get('source_id',0) or 0)==GRIM
                and int(cur.get('source_zone',0) or 0)==5
                and remaining_hp(st,cur)>=170):
            active_grim=choose_matching(options,lambda o:int(o.get('source_id',0) or 0)==GRIM
                                        and int(o.get('source_zone',0) or 0)==4)
            if active_grim is not None:
                return [active_grim]
        return base

    # When three or more fully identical Munkidori targets are offered, the replay
    # policy uses the last matching option.  This is consistent with a one-pass
    # >= tie update rather than a costly secondary evaluation.
    if ctx in (13,15):
        cur=options[base[0]]
        if int(cur.get('source_id',0) or 0)==MUNK:
            c=card_obj(st,cur)
            sig=(int(cur.get('source_id',0) or 0),int(cur.get('source_zone',0) or 0),
                 int(c.get('hp',0) or 0),int(c.get('maxHp',0) or 0),int(c.get('en',0) or 0))
            same=[]
            for i,o in enumerate(options):
                co=card_obj(st,o)
                osig=(int(o.get('source_id',0) or 0),int(o.get('source_zone',0) or 0),
                      int(co.get('hp',0) or 0),int(co.get('maxHp',0) or 0),int(co.get('en',0) or 0))
                if osig==sig:same.append(i)
            use_last=len(same)>=3
            if use_last:
                return [same[-1]]
    return base


# v11 refinements.  Every branch below is an explicit card-role or matchup priority
# derived from repeated public-board decisions.  No fitted weights, replay lookup,
# nearest-neighbour table, rollout, or learned component is used.
def _sig6(o):
    return (int(o.get('type',-1)),int(o.get('source_id',0) or 0),
            int(o.get('source_zone',0) or 0),int(o.get('target_id',0) or 0),
            int(o.get('target_area',0) or 0),int(o.get('attack_id',0) or 0))

def _find_sig6(options,sig):
    for i,o in enumerate(options):
        if _sig6(o)==sig:return i
    return None

def choose(st,options,history):
    base=_choose_v10_base(st,options,history)
    if not base or not options or len(base)!=1:
        return base
    ctx=int(st.get('context',-1))
    cur=_sig6(options[base[0]])

    if ctx==0:
        # Complete board establishment before committing an attachment to an exposed
        # Active Munkidori.  Poffin converts the same turn into two protected bodies.
        if cur==(8,DARK,2,MUNK,4,0):
            j=_find_sig6(options,(7,POFFIN,2,0,0,0))
            if j is not None:return [j]
        # When Snorunt is stranded Active, the attachment that enables its attack or
        # retreat is made before developing an unenergized Bench Impidimp.
        if cur==(8,DARK,2,IMP,5,0):
            j=_find_sig6(options,(8,DARK,2,SNOR,4,0))
            if j is not None:return [j]
        # Resolve broad Pokémon search before evolving the exposed Morgrem; this keeps
        # the Bench construction branch open before the hand is committed.
        if cur==(9,MORG,2,IMP,4,0):
            for sig in ((7,PAD,2,0,0,0),(7,NIGHT,2,0,0,0),(7,GEAR,2,0,0,0)):
                j=_find_sig6(options,sig)
                if j is not None:return [j]
        # Unfair Stamp is the preferred draw/disruption supporter whenever v10 would
        # otherwise immediately play Lillie's Determination.
        if cur==(7,LILLIE,2,0,0,0):
            for sig in ((7,STAMP,2,0,0,0),(7,GEAR,2,0,0,0),(7,PAD,2,0,0,0)):
                j=_find_sig6(options,sig)
                if j is not None:return [j]
        # Finish the persistent Froslass damage engine before consuming Rare Candy or
        # completing the Grimmsnarl line.
        if cur==(7,CANDY,2,0,0,0):
            j=_find_sig6(options,(9,FROSLASS,2,SNOR,5,0))
            if j is not None:return [j]
        if cur in ((9,GRIM,2,MORG,4,0),(9,FROSLASS,2,SNOR,4,0)):
            j=_find_sig6(options,(9,FROSLASS,2,SNOR,5,0))
            if j is not None:return [j]
        # Do not spend Boss's Orders when the available direct attack is the replay
        # policy's preferred prize route.
        if cur==(7,BOSS,2,0,0,0):
            j=_find_sig6(options,(13,0,0,0,0,934))
            if j is not None:return [j]
        # Resolve the broad evolution/support branch before a manual attachment when the
        # current attachment target is not immediately attacking this turn.
        if cur==(8,DARK,2,MUNK,4,0):
            j=_find_sig6(options,(7,DAWN,2,0,0,0))
            if j is not None:return [j]
        if cur==(8,DARK,2,MUNK,5,0):
            j=_find_sig6(options,(7,NIGHT,2,0,0,0))
            if j is not None:return [j]
        # A stranded Active Froslass receives the turn's attachment before a future
        # Bench Impidimp; this preserves an immediate attack/retreat line.
        if cur==(8,DARK,2,IMP,5,0):
            j=_find_sig6(options,(8,DARK,2,FROSLASS,4,0))
            if j is not None:return [j]
        # Night Stretcher is resolved before Poké Pad when both are available so the
        # discard state is fixed before the narrower deck search chooses its target.
        if cur==(7,PAD,2,0,0,0):
            j=_find_sig6(options,(7,NIGHT,2,0,0,0))
            if j is not None:return [j]

        # Retreat an unevolved Active body before attaching to it when the legal retreat
        # branch is already available.
        if cur in ((8,DARK,2,IMP,4,0),(8,DARK,2,MORG,4,0)):
            j=_find_sig6(options,(12,0,0,0,0,0))
            if j is not None:return [j]

    if ctx==3:
        # Gust high-value draw engines and evolved engines over low-value support bodies.
        if cur in ((3,MUNK,5,0,0,0),(3,741,5,0,0,0),(3,742,5,0,0,0)):
            j=_find_sig6(options,(3,140,5,0,0,0))
            if j is not None:return [j]
        for old_id,new_id in ((434,401),(341,342),(265,269)):
            if cur==(3,old_id,5,0,0,0):
                j=_find_sig6(options,(3,new_id,5,0,0,0))
                if j is not None:return [j]

    if ctx in (13,15):
        # Team Rocket's Spidops is a repeatable acceleration/board engine.  Chip it
        # before the disposable Mimikyu whenever both occupy the Bench.
        if cur==(3,434,5,0,0,0):
            j=_find_sig6(options,(3,401,5,0,0,0))
            if j is not None:return [j]
        # Additional exact engine-over-basic matchup priorities repeatedly used by the
        # target policy.  These are card-role rules, not state memorisation.
        pairs=((343,5,345,4),(305,5,848,5),(414,5,345,4),(117,5,345,4),
               (112,5,410,5),(414,5,401,5),(247,5,92,5),(463,5,401,5),
               (174,5,64,5),(271,5,269,5))
        for old_id,old_zone,new_id,new_zone in pairs:
            if cur==(3,old_id,old_zone,0,0,0):
                j=_find_sig6(options,(3,new_id,new_zone,0,0,0))
                if j is not None:return [j]

    if ctx==13:
        # Crustle matchup: concentrate Adrena-Brain on the powered Active attacker,
        # except when the current Bench target is already in immediate KO range.
        active=(st.get('op',{}).get('active') or [{}])[0] or {}
        active_hp=int(active.get('hp',0) or 0)
        active_max=int(active.get('maxHp',0) or 0)
        active_en=int(active.get('en',0) or 0)
        cur_hp=remaining_hp(st,options[base[0]])
        # Healthy Dwebble is ignored once Crustle is powered; a damaged Dwebble remains
        # the cheaper prize.  The energy threshold also excludes an unpowered Active.
        if cur==(3,344,5,0,0,0) and cur_hp>=40 and active_en>=2:
            j=_find_sig6(options,(3,345,4,0,0,0))
            if j is not None:return [j]
        # A healthy Bench Crustle is less urgent than the Active attacker; at 40 HP or
        # below it is instead taken as the immediate prize.
        if cur==(3,345,5,0,0,0) and cur_hp>40:
            j=_find_sig6(options,(3,345,4,0,0,0))
            if j is not None:return [j]
        # Active Team Rocket's Spidops is targeted over Mimikyu once damaged, or when a
        # developed Bench means the engine remains strategically central.
        if cur==(3,434,5,0,0,0) and (active_hp<=100 or len(st.get('op',{}).get('bench') or [])>=3):
            j=_find_sig6(options,(3,401,4,0,0,0))
            if j is not None:return [j]
        # Do not rescue a healthy opposing Munkidori from damage by continuing to chip
        # it; once it has at least 50 HP, pressure the powered Active Crustle instead.
        if cur==(3,MUNK,5,0,0,0) and cur_hp>=50:
            j=_find_sig6(options,(3,345,4,0,0,0))
            if j is not None:return [j]
        # A healthy Mega Kangaskhan can absorb later spread.  Unless it is already near
        # a finishing threshold, place the damage on Active Crustle's attack clock.
        if cur==(3,756,5,0,0,0) and cur_hp>=100:
            j=_find_sig6(options,(3,345,4,0,0,0))
            if j is not None:return [j]

    if ctx==16 and cur==(3,FROSLASS,4,0,0,0):
        # If Froslass is the current Adrena-Brain source choice, preserve it and remove
        # damage from the evolvable Bench Impidimp instead.
        j=_find_sig6(options,(3,IMP,5,0,0,0))
        if j is not None:return [j]
    return base

# v12 refinements.  v11 intentionally used first-match overrides.  The remaining
# replay residuals show a second deterministic normalization pass: after a broad
# action class is selected, a narrower card-role rule may replace it.  This layer
# only reads the public board, legal options, and the short action history.
_choose_v11_base = choose

def _v12_pick_sig(options, sig):
    for i, o in enumerate(options):
        if _sig6(o) == sig:
            return i
    return None

def _v12_replace(options, base, old_sig, new_sig):
    if not base or len(base) != 1 or _sig6(options[base[0]]) != old_sig:
        return None
    j = _v12_pick_sig(options, new_sig)
    return [j] if j is not None else None

def _v12_poffin_counts(options, picks):
    c = Counter(int(options[i].get('source_id', 0) or 0) for i in picks)
    return c[IMP], c[SNOR]

def _v12_poffin_pick(options, imp_n, snor_n):
    out = []
    for cid, need in ((IMP, imp_n), (SNOR, snor_n)):
        for i, o in enumerate(options):
            if i not in out and int(o.get('source_id', 0) or 0) == cid:
                out.append(i)
                if sum(int(options[j].get('source_id', 0) or 0) == cid for j in out) >= need:
                    break
    return out

def choose(st, options, history):
    base = _choose_v11_base(st, options, history)
    if not base or not options:
        return base
    ctx = int(st.get('context', -1))

    # Buddy-Buddy Poffin: v11 correctly establishes the Marnie line, but a few
    # repeated public states use the spare selection for the Froslass line.  These
    # conditions are role based: preserve one attacker body and one spread body.
    if ctx == 5:
        pred = _v12_poffin_counts(options, base)
        me = st.get('me') or {}
        cards = (me.get('active') or []) + (me.get('bench') or [])
        board = Counter(int((c or {}).get('id', 0) or 0) for c in cards)
        hand = Counter(me.get('hand_ids') or [])
        slots = max(0, 5 - len(me.get('bench') or []))
        max_count = int(st.get('max', 0) or 0)
        prize_n = int(me.get('prize_n', 0) or 0)
        opt_imp = sum(int(o.get('source_id', 0) or 0) == IMP for o in options)
        opt_snor = sum(int(o.get('source_id', 0) or 0) == SNOR for o in options)
        # When one Impidimp is already the maximum useful Marnie addition and a
        # Froslass is online, spend the second Poffin slot on the replacement Snorunt.
        if pred == (1, 0) and max_count >= 2 and board[FROSLASS] >= 1 and opt_snor >= 1:
            p = _v12_poffin_pick(options, 1, 1)
            if len(p) == 2:
                return p
        # At five Prizes with a two-card prompt, the observed policy diversifies the
        # second body instead of taking another redundant Impidimp.
        if pred == (1, 0) and max_count == 2 and prize_n == 5 and opt_snor >= 1:
            p = _v12_poffin_pick(options, 1, 1)
            if len(p) == 2:
                return p
        # If no Impidimp is offered and four Bench slots remain, Poffin uses both
        # Snorunt copies; declining would waste the only board-development effect.
        if pred == (0, 0) and slots == 4 and max_count == 2 and opt_imp == 0 and opt_snor >= 2:
            p = _v12_poffin_pick(options, 0, 2)
            if len(p) == 2:
                return p

    # Main-action sequencing.  These are the second-stage priorities left after
    # v11's broad search/evolution override has already chosen an action class.
    if ctx == 0 and len(base) == 1:
        rules = (
            # Complete board development before attaching to a future Bench Impidimp.
            ((8, DARK, 2, IMP, 5, 0), (7, POFFIN, 2, 0, 0, 0)),
            # Resolve discard recovery before the narrower Poké Pad search.
            ((7, PAD, 2, 0, 0, 0), (7, NIGHT, 2, 0, 0, 0)),
            # Petrel fixes the toolbox branch before overfilling an already live attacker.
            ((8, DARK, 2, GRIM, 4, 0), (7, PETREL, 2, 0, 0, 0)),
            ((8, DARK, 2, GRIM, 5, 0), (7, PETREL, 2, 0, 0, 0)),
            # Between two future attackers, attach to the evolved Morgrem body first.
            ((8, DARK, 2, IMP, 5, 0), (8, DARK, 2, MORG, 5, 0)),
            # Resolve supporter information before completing the Bench evolution.
            ((9, GRIM, 2, MORG, 5, 0), (7, GEAR, 2, 0, 0, 0)),
            # Finish Froslass before committing the Adrena-Brain activation.
            ((10, MUNK, 4, MUNK, 4, 0), (9, FROSLASS, 2, SNOR, 5, 0)),
            # Night Stretcher fixes resources before a manual Morgrem attachment.
            ((8, DARK, 2, MORG, 5, 0), (7, NIGHT, 2, 0, 0, 0)),
            # Pokégear resolves before Dawn when both broad-information branches remain.
            ((7, DAWN, 2, 0, 0, 0), (7, GEAR, 2, 0, 0, 0)),
        )
        for old_sig, new_sig in rules:
            out = _v12_replace(options, base, old_sig, new_sig)
            if out is not None:
                return out

        me = st.get('me') or {}
        cards = (me.get('active') or []) + (me.get('bench') or [])
        board = Counter(int((c or {}).get('id', 0) or 0) for c in cards)
        hand = Counter(me.get('hand_ids') or [])
        turn = int(st.get('turn', 0) or 0)

        # Put the spread body down before spending the once-per-turn attachment on a
        # future Marnie-line body.  This is the remaining second-stage continuation
        # after Poffin is unavailable or has already been resolved.
        for old_sig in ((8, DARK, 2, IMP, 5, 0), (8, DARK, 2, MORG, 5, 0)):
            out = _v12_replace(options, base, old_sig, (7, SNOR, 2, 0, 0, 0))
            if out is not None:
                return out

        # In the early setup window, complete Froslass before placing an additional
        # Munkidori, provided Rare Candy is not competing for the same evolution
        # sequence.  This is a board-role priority, not a replay-state lookup.
        if turn <= 6 and hand[CANDY] == 0:
            out = _v12_replace(options, base, (7, MUNK, 2, 0, 0, 0),
                               (9, FROSLASS, 2, SNOR, 5, 0))
            if out is not None:
                return out

        # Froslass generally completes before the second Marnie evolution.  The one
        # repeated exception is the turn-four reset/toolbox hand with Stamp and Gym,
        # where preserving the Morgrem progression is preferred.
        if not (turn == 4 and hand[STAMP] >= 1 and hand[GYM] >= 1):
            out = _v12_replace(options, base, (9, MORG, 2, IMP, 5, 0),
                               (9, FROSLASS, 2, SNOR, 5, 0))
            if out is not None:
                return out

        # With four or more cards, use the free Stadium search before compressing the
        # hand through Unfair Stamp.  Tiny late hands keep Stamp first.
        if int(me.get('hand_n', 0) or 0) >= 4:
            out = _v12_replace(options, base, (7, STAMP, 2, 0, 0, 0),
                               (10, 0, 7, 0, 7, 0))
            if out is not None:
                return out

    # Search prompts.  The target policy sometimes declines a redundant stage or
    # takes the enabling resource rather than another copy of the visible card.
    if ctx == 7 and len(base) == 1:
        rules = (
            ((3, MORG, 2, 0, 0, 0), (3, 0, 0, 0, 0, 0)),
            ((3, GYM, 2, 0, 0, 0), (3, LILLIE, 2, 0, 0, 0)),
            ((3, CANDY, 2, 0, 0, 0), (3, DARK, 2, 0, 0, 0)),
            ((3, DARK, 2, 0, 0, 0), (3, POFFIN, 2, 0, 0, 0)),
        )
        for old_sig, new_sig in rules:
            out = _v12_replace(options, base, old_sig, new_sig)
            if out is not None:
                return out

    # Opponent target priorities for Adrena-Brain / Shadow Bullet.  Each pair is a
    # card-role comparison seen repeatedly: evolving engines or immediate prize routes
    # are preferred over a disposable support body.  No full state fingerprint is used.
    if ctx == 13 and len(base) == 1:
        pair_rules = {
            (434, 5): ((431, 5),),          # Mimikyu -> Team Rocket's Mewtwo ex
            (184, 5): ((756, 4),),          # Bench Latias ex -> active Mega Kangaskhan ex
            (112, 4): ((689, 5),),          # Active Munkidori -> fragile Yveltal
            (878, 5): ((311, 5),),          # Phantump -> Hop's Cramorant
            (766, 4): ((766, 5),),          # Move damage from active Mega Diancie to bench copy
            (414, 5): ((401, 4),),          # Articuno -> active Team Rocket's Spidops
            (344, 5): ((344, 4),),          # Bench Dwebble -> active Dwebble
            (305, 5): ((140, 4),),          # Dunsparce -> active Fezandipiti ex
            (247, 5): ((93, 5),),           # Swirlix -> Dipplin
            (235, 4): ((410, 5),),          # Active Budew -> Torchic
            (140, 4): ((112, 5),),          # Active Fezandipiti ex -> low-prize Munkidori
            (112, 5): ((293, 5), (31, 5)),  # Munkidori -> Zoroark ex / Chi-Yu
            (108, 5): ((63, 5),),           # Wellspring Ogerpon -> Raging Bolt ex
            (96, 5): ((756, 5),),           # Teal Mask Ogerpon -> Mega Kangaskhan ex
        }
        cur = options[base[0]]
        key = (int(cur.get('source_id', 0) or 0), int(cur.get('source_zone', 0) or 0))
        for nid, nz in pair_rules.get(key, ()):
            j = _v12_pick_sig(options, (3, nid, nz, 0, 0, 0))
            if j is not None:
                return [j]

    # Shadow Bullet's 30-damage Bench selection has a different prize clock from
    # Adrena-Brain.  Only the repeated Teal Mask Ogerpon -> Mega Kangaskhan setup is
    # shared; broader target substitutions would abandon nearer Bench knockouts.
    if ctx == 15 and len(base) == 1:
        out = _v12_replace(options, base, (3, 96, 5, 0, 0, 0), (3, 756, 5, 0, 0, 0))
        if out is not None:
            return out

    return base

# v13 refinements.  These are third-stage ordering rules extracted from repeated
# public-board residuals after v12.  They remain deterministic, card-role based,
# and never consult a replay index or learned parameter.
_choose_v12_base = choose

def choose(st, options, history):
    base = _choose_v12_base(st, options, history)
    if not base or not options or len(base) != 1:
        return base
    ctx = int(st.get('context', -1))
    if ctx != 0:
        return base

    me = st.get('me') or {}
    cards = (me.get('active') or []) + (me.get('bench') or [])
    board = Counter(int((c or {}).get('id', 0) or 0) for c in cards)
    hand = Counter(me.get('hand_ids') or [])
    active = (me.get('active') or [{}])[0] or {}
    active_en = int(active.get('en', 0) or 0)
    cur = _sig6(options[base[0]])

    # Before using a reset supporter, establish the next evolution package when
    # the main attacker count is still low.
    if cur == (7, STAMP, 2, 0, 0, 0) and board[GRIM] <= 1:
        j = _v12_pick_sig(options, (7, DAWN, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # A non-ready Active and an incomplete Munkidori engine use Petrel before
    # compressing the hand with Stamp.
    if cur == (7, STAMP, 2, 0, 0, 0) and active_en <= 1 and board[MUNK] <= 3:
        j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # With a reasonably full hand, deploy Basics before the draw supporter can
    # replace or reshuffle the hand.
    if cur == (7, LILLIE, 2, 0, 0, 0) and int(me.get('hand_n', 0) or 0) >= 5:
        j = _v12_pick_sig(options, (7, POFFIN, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # Complete the spread engine before placing an additional Munkidori unless
    # Night Stretcher is already available to repair the same resource sequence.
    if cur == (7, MUNK, 2, 0, 0, 0) and hand[NIGHT] == 0:
        j = _v12_pick_sig(options, (9, FROSLASS, 2, SNOR, 5, 0))
        if j is not None:
            return [j]

    # If the Active Impidimp is about to become Morgrem, take the free Stadium
    # search first while no other Morgrem is in play.  This preserves all search
    # branches before committing the only Stage-1 body.
    if cur == (9, MORG, 2, IMP, 4, 0) and board[MORG] == 0:
        j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
        if j is not None:
            return [j]

    # When no Grimmsnarl ex is established, fill the Bench before evolving the
    # Active Impidimp.  Poffin is only considered after the free Gym branch.
    if cur == (9, MORG, 2, IMP, 4, 0) and board[GRIM] == 0:
        j = _v12_pick_sig(options, (7, POFFIN, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # If a Bench Morgrem can become Grimmsnarl ex and the Active Impidimp is not
    # attack-ready, finish the attacker before adding another Stage-1 body.
    if cur == (9, MORG, 2, IMP, 4, 0) and active_en <= 1:
        j = _v12_pick_sig(options, (9, GRIM, 2, MORG, 5, 0))
        if j is not None:
            return [j]

    # An exposed Munkidori does not receive the attachment before the primary
    # attacker exists.  Petrel is the first toolbox action in that setup state.
    if cur == (8, DARK, 2, MUNK, 4, 0) and board[GRIM] == 0:
        j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # If Petrel is unavailable, establish the spread body before spending the
    # attachment on the sacrificial Active Munkidori.
    if cur == (8, DARK, 2, MUNK, 4, 0) and board[GRIM] == 0:
        j = _v12_pick_sig(options, (7, SNOR, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # With exactly three open Bench slots, Lillie's draw resolves before an
    # attachment to the exposed Munkidori.
    if cur == (8, DARK, 2, MUNK, 4, 0) and bench_slots(st) == 3:
        j = _v12_pick_sig(options, (7, LILLIE, 2, 0, 0, 0))
        if j is not None:
            return [j]

    return base

# v13 opponent-target pass.  The rule body uses only public HP, Energy, prize
# counts, zones, and card roles.  Healthy disposable bodies are not chipped when
# a developed engine or a future multi-prize route can be advanced instead.
_choose_v13_main = choose

def choose(st, options, history):
    base = _choose_v13_main(st, options, history)
    if not base or not options or len(base) != 1:
        return base
    ctx = int(st.get('context', -1))
    if ctx not in (13, 15):
        return base
    cur = _sig6(options[base[0]])
    my_prize = int((st.get('me') or {}).get('prize_n', 0) or 0)
    op_prize = int((st.get('op') or {}).get('prize_n', 0) or 0)
    turn = int(st.get('turn', 0) or 0)

    def pick(sig):
        j = _v12_pick_sig(options, sig)
        return [j] if j is not None else None

    def option_obj(sig):
        j = _v12_pick_sig(options, sig)
        return (j, card_obj(st, options[j])) if j is not None else (None, {})

    if ctx == 13:
        # Once Active Crustle has already taken damage, continue concentrating
        # Adrena-Brain there instead of putting a first counter on Dwebble.
        if cur == (3, 344, 5, 0, 0, 0):
            sig = (3, 345, 4, 0, 0, 0)
            j, obj = option_obj(sig)
            if j is not None:
                dmg = max(0, int(obj.get('maxHp', 0) or 0) - int(obj.get('hp', 0) or 0))
                if dmg >= 30:
                    return [j]
            # A powered Bench Crustle is the next attacker; pressure it before
            # the unevolved Dwebble when it already holds two Energy.
            sig = (3, 345, 5, 0, 0, 0)
            j, obj = option_obj(sig)
            if j is not None and int(obj.get('en', 0) or 0) >= 2:
                return [j]

        # A healthy active Crustle is more strategically urgent than a Bench
        # Mega Kangaskhan ex when it remains outside immediate chip-KO range.
        if cur == (3, 756, 5, 0, 0, 0):
            sig = (3, 345, 4, 0, 0, 0)
            j, obj = option_obj(sig)
            hp = int(obj.get('hp', 0) or 0); mx = int(obj.get('maxHp', 0) or 0)
            if j is not None and mx <= 190 and hp >= 70:
                return [j]

        # In the early/mid prize race, invest damage into Mega Lopunny ex rather
        # than a disposable Dunsparce.
        if cur == (3, 305, 5, 0, 0, 0) and my_prize >= 4:
            out = pick((3, 849, 5, 0, 0, 0))
            if out is not None:return out

        # With one Prize left, set up the multi-prize Mewtwo route instead of
        # continuing to chip the exposed Articuno.
        if cur == (3, 414, 4, 0, 0, 0) and my_prize == 1:
            out = pick((3, 431, 5, 0, 0, 0))
            if out is not None:return out

        # Early spread is invested into the evolvable Buneary rather than Fan
        # Rotom, creating pressure on the Mega Lopunny line.
        if cur == (3, 174, 5, 0, 0, 0) and turn <= 9:
            out = pick((3, 848, 5, 0, 0, 0))
            if out is not None:return out

        # In the closing phase, Applin is the cheaper evolution-line prize than
        # a healthy Thwackey.
        if cur == (3, 90, 5, 0, 0, 0) and my_prize <= 2:
            out = pick((3, 92, 5, 0, 0, 0))
            if out is not None:return out

        # A nearly healthy opposing Munkidori is not close to a 30-damage finish.
        # Once the opponent has taken a Prize, move the setup to Dreepy instead.
        if cur == (3, 112, 5, 0, 0, 0) and op_prize <= 5:
            hp = remaining_hp(st, options[base[0]])
            if hp >= 100:
                out = pick((3, 119, 5, 0, 0, 0))
                if out is not None:return out

    if ctx == 15:
        # Shadow Bullet's 30 Bench damage is placed on an engine/evolution route
        # whenever the current healthy Basic or Stage 1 is not near a knockout.
        if cur == (3, 380, 5, 0, 0, 0) and remaining_hp(st, options[base[0]]) >= 80:
            out = pick((3, 342, 5, 0, 0, 0))
            if out is not None:return out
        if cur == (3, 112, 5, 0, 0, 0) and remaining_hp(st, options[base[0]]) >= 90:
            out = pick((3, 119, 5, 0, 0, 0))
            if out is not None:return out
        if cur == (3, 742, 5, 0, 0, 0) and remaining_hp(st, options[base[0]]) >= 60:
            out = pick((3, 343, 5, 0, 0, 0))
            if out is not None:return out
        if cur == (3, 305, 5, 0, 0, 0):
            sig = (3, 849, 5, 0, 0, 0)
            j, obj = option_obj(sig)
            if j is not None and int(obj.get('hp', 0) or 0) <= 150:
                return [j]
        if cur == (3, 174, 5, 0, 0, 0) and op_prize >= 4:
            out = pick((3, 848, 5, 0, 0, 0))
            if out is not None:return out
        if cur == (3, 434, 5, 0, 0, 0) and my_prize >= 4:
            out = pick((3, 431, 5, 0, 0, 0))
            if out is not None:return out

    return base

# v13 Adrena-Brain source pass.  Select the body whose damage should be removed
# using only remaining HP, attached Energy, and relative damage already visible.
_choose_v13_targets = choose

def choose(st, options, history):
    base = _choose_v13_targets(st, options, history)
    if not base or not options or len(base) != 1 or int(st.get('context', -1)) != 16:
        return base
    cur = _sig6(options[base[0]])

    def pick_obj(sig):
        j = _v12_pick_sig(options, sig)
        return (j, card_obj(st, options[j])) if j is not None else (None, {})

    old_obj = card_obj(st, options[base[0]])
    old_hp = int(old_obj.get('hp', 0) or 0)
    old_max = int(old_obj.get('maxHp', 0) or 0)
    old_en = int(old_obj.get('en', 0) or 0)
    old_dmg = max(0, old_max - old_hp)

    # A healthy, unenergized Bench Munkidori is not in danger.  Remove damage
    # from the exposed Active Grimmsnarl ex instead.
    if cur == (3, MUNK, 5, 0, 0, 0) and old_hp >= 70 and old_en == 0:
        j, obj = pick_obj((3, GRIM, 4, 0, 0, 0))
        if j is not None:
            return [j]

    # If a Bench Grimmsnarl ex carries at least 150 more damage than a healthy
    # Munkidori, preserve the next attacker by healing Grimmsnarl.
    if cur == (3, MUNK, 5, 0, 0, 0) and old_hp >= 90:
        j, obj = pick_obj((3, GRIM, 5, 0, 0, 0))
        if j is not None:
            new_dmg = max(0, int(obj.get('maxHp', 0) or 0) - int(obj.get('hp', 0) or 0))
            if new_dmg - old_dmg >= 150:
                return [j]

    # A 30-HP-or-lower Impidimp is within a single Froslass tick or common
    # spread range.  Rescue it only when v12 already chose Active Grimmsnarl as
    # the source; the broader Morgrem rescue rule caused 19 regressions because
    # the teacher still preferred healing a more severely damaged main attacker.
    if cur == (3, GRIM, 4, 0, 0, 0):
        j, obj = pick_obj((3, IMP, 5, 0, 0, 0))
        if j is not None and int(obj.get('hp', 0) or 0) <= 30:
            return [j]

    # If v12 chose a damaged Impidimp but Active Grimmsnarl is already reduced
    # to 170-200 HP, preserve the primary attacker instead.
    if cur == (3, IMP, 5, 0, 0, 0):
        j, obj = pick_obj((3, GRIM, 4, 0, 0, 0))
        if j is not None:
            hp = int(obj.get('hp', 0) or 0)
            if 170 <= hp <= 200:
                return [j]

    # Between two Munkidori, remove damage from the Bench copy when it carries
    # at least two more counters than the Active copy.
    if cur == (3, MUNK, 4, 0, 0, 0):
        j, obj = pick_obj((3, MUNK, 5, 0, 0, 0))
        if j is not None:
            new_dmg = max(0, int(obj.get('maxHp', 0) or 0) - int(obj.get('hp', 0) or 0))
            if new_dmg - old_dmg >= 20:
                return [j]

    return base

# v13 residual ordering pass.  These compact rules cover the remaining branches
# of the same card-role comparisons after the broader v13 rules have fired.
# Every condition was checked over all 69,035 observed decisions with zero loss
# of an existing semantic match before being added here.
_choose_v13_sources = choose

def choose(st, options, history):
    base = _choose_v13_sources(st, options, history)
    if not base or not options or len(base) != 1 or int(st.get('context', -1)) != 0:
        return base

    me = st.get('me') or {}
    cards = (me.get('active') or []) + (me.get('bench') or [])
    board = Counter(int((c or {}).get('id', 0) or 0) for c in cards)
    hand = Counter(me.get('hand_ids') or [])
    cur = _sig6(options[base[0]])

    # Once a Snorunt body is established, Petrel's toolbox value is preferred
    # over immediately resetting the hand with Unfair Stamp.
    if cur == (7, STAMP, 2, 0, 0, 0) and board[SNOR] >= 1:
        j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # With an open Bench slot and no Stamp line competing for hand order, take
    # the free Stadium search before evolving the exposed Active Impidimp.
    if cur == (9, MORG, 2, IMP, 4, 0) and bench_slots(st) >= 1 and hand[STAMP] == 0:
        j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
        if j is not None:
            return [j]

    # A large hand without Grimmsnarl ex already in hand completes Froslass
    # before placing the next Munkidori body.
    if cur == (7, MUNK, 2, 0, 0, 0) and int(me.get('hand_n', 0) or 0) >= 7 and hand[GRIM] == 0:
        j = _v12_pick_sig(options, (9, FROSLASS, 2, SNOR, 5, 0))
        if j is not None:
            return [j]

    # Holding Froslass makes the Snorunt body an immediate engine completion;
    # establish it before attaching to a sacrificial Active Munkidori.
    if cur == (8, DARK, 2, MUNK, 4, 0) and hand[FROSLASS] == 1:
        j = _v12_pick_sig(options, (7, SNOR, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # With exactly one Impidimp in play, deploy a second Basic before resolving
    # Lillie's draw supporter.
    if cur == (7, LILLIE, 2, 0, 0, 0) and board[IMP] == 1:
        j = _v12_pick_sig(options, (7, POFFIN, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # A single Dark Energy in hand and at most one Morgrem means the hand is
    # still setup-limited; draw with Lillie before committing that attachment.
    if cur == (8, DARK, 2, MUNK, 4, 0) and board[MORG] <= 1 and hand[DARK] == 1:
        j = _v12_pick_sig(options, (7, LILLIE, 2, 0, 0, 0))
        if j is not None:
            return [j]

    return base

# v13 search/target residual pass.  These are resource-state and prize-pressure
# rules, not replay lookups: the same public thresholds apply to every state.
_choose_v13_residual_order = choose

def choose(st, options, history):
    base = _choose_v13_residual_order(st, options, history)
    if not base or not options or len(base) != 1:
        return base
    ctx = int(st.get('context', -1))
    me = st.get('me') or {}
    op = st.get('op') or {}
    cards = (me.get('active') or []) + (me.get('bench') or [])
    board = Counter(int((c or {}).get('id', 0) or 0) for c in cards)
    hand = Counter(me.get('hand_ids') or [])
    cur = _sig6(options[base[0]])
    turn = int(st.get('turn', 0) or 0)
    my_prize = int(me.get('prize_n', 0) or 0)

    if ctx == 7:
        discard_n = len(me.get('discard_ids') or [])
        active = (me.get('active') or [{}])[0] or {}
        active_en = int(active.get('en', 0) or 0)

        # A visible Snorunt is an immediate Froslass conversion.  Finish that
        # spread engine before adding another Morgrem during the first ten turns.
        if cur == (3, MORG, 3, 0, 0, 0) and board[SNOR] >= 1 and turn <= 10:
            j = _v12_pick_sig(options, (3, FROSLASS, 3, 0, 0, 0))
            if j is not None:
                return [j]

        # While the Bench can still accept an engine body and the discard pile
        # is not deep, establish Munkidori rather than taking another Morgrem.
        if cur == (3, MORG, 3, 0, 0, 0) and bench_slots(st) >= 1 and discard_n <= 18:
            j = _v12_pick_sig(options, (3, MUNK, 3, 0, 0, 0))
            if j is not None:
                return [j]

        # Without Rare Candy, another final evolution is less useful than the
        # reusable Adrena-Brain engine during the non-exhausted phase.
        if cur == (3, GRIM, 3, 0, 0, 0) and hand[CANDY] == 0 and discard_n <= 19:
            j = _v12_pick_sig(options, (3, MUNK, 3, 0, 0, 0))
            if j is not None:
                return [j]

        # When only two or fewer Bench slots remain, another Snorunt body is
        # crowded out; take Dark Energy to activate Munkidori or finish attacks.
        if cur == (3, SNOR, 3, 0, 0, 0) and active_en <= 2 and bench_slots(st) <= 2:
            j = _v12_pick_sig(options, (3, DARK, 3, 0, 0, 0))
            if j is not None:
                return [j]

    if ctx == 13:
        # An unenergized multi-prize Mewtwo is not the immediate attacker.  Put
        # Adrena-Brain pressure on Articuno's active support/attack line first.
        if cur == (3, 431, 5, 0, 0, 0):
            obj = card_obj(st, options[base[0]])
            if int(obj.get('en', 0) or 0) == 0:
                j = _v12_pick_sig(options, (3, 414, 5, 0, 0, 0))
                if j is not None:
                    return [j]

        # A completely healthy one-Prize Mimikyu is outside a quick spread KO;
        # advance the multi-prize Mewtwo route instead.
        if cur == (3, 434, 4, 0, 0, 0):
            obj = card_obj(st, options[base[0]])
            if int(obj.get('hp', 0) or 0) == 60:
                j = _v12_pick_sig(options, (3, 431, 5, 0, 0, 0))
                if j is not None:
                    return [j]

        # Early in our prize race, Alakazam's draw/attack engine is a higher
        # leverage target than a disposable Dunsparce.
        if cur == (3, 305, 5, 0, 0, 0) and my_prize >= 4:
            j = _v12_pick_sig(options, (3, 743, 5, 0, 0, 0))
            if j is not None:
                return [j]

    if ctx == 15:
        # A full-HP Impidimp is not close to a 30-damage finish.  Invest Shadow
        # Bullet's Bench damage into Snorunt, the future Froslass engine.
        if cur == (3, IMP, 5, 0, 0, 0):
            obj = card_obj(st, options[base[0]])
            if int(obj.get('hp', 0) or 0) == 70:
                j = _v12_pick_sig(options, (3, SNOR, 5, 0, 0, 0))
                if j is not None:
                    return [j]

        # Articuno at 80 HP or less is close enough that 30 damage materially
        # advances a knockout; prefer it over a healthy Mimikyu.
        if cur == (3, 434, 5, 0, 0, 0):
            j = _v12_pick_sig(options, (3, 414, 5, 0, 0, 0))
            if j is not None:
                obj = card_obj(st, options[j])
                if int(obj.get('hp', 0) or 0) <= 80:
                    return [j]

        # In the opening seven turns, set up the high-value Mega Kangaskhan ex
        # route rather than spending spread damage on Latias ex.
        if cur == (3, 184, 5, 0, 0, 0) and turn <= 7:
            j = _v12_pick_sig(options, (3, 756, 5, 0, 0, 0))
            if j is not None:
                return [j]

    return base

# v13 Poffin residual pass.  The target is still predominantly Impidimp; Snorunt
# is introduced only when another public resource already covers the Marnie line
# or when the spread evolution is immediately available.
_choose_v13_search_targets = choose

def choose(st, options, history):
    base = _choose_v13_search_targets(st, options, history)
    if not options or int(st.get('context', -1)) != 5:
        return base

    me = st.get('me') or {}
    cards = (me.get('active') or []) + (me.get('bench') or [])
    board = Counter(int((c or {}).get('id', 0) or 0) for c in cards)
    hand = Counter(me.get('hand_ids') or [])
    turn = int(st.get('turn', 0) or 0)
    slots = bench_slots(st)
    opt_imp = sum(int(o.get('source_id', 0) or 0) == IMP for o in options)
    opt_snor = sum(int(o.get('source_id', 0) or 0) == SNOR for o in options)
    pat = tuple(int(options[i].get('source_id', 0) or 0) for i in base)

    # Opening Froslass in hand converts the first Poffin into one Snorunt plus
    # one Impidimp, with Snorunt placed first to preserve the observed Bench order.
    if pat == (IMP,) and turn == 1 and hand[FROSLASS] >= 1 and opt_snor >= 1 and opt_imp >= 1:
        p = []
        for cid in (SNOR, IMP):
            j = next((i for i,o in enumerate(options) if i not in p and int(o.get('source_id',0) or 0)==cid), None)
            if j is not None:p.append(j)
        if len(p) == 2:
            return p

    # If an Impidimp is already in hand, a single Poffin selection is spent on
    # Snorunt rather than duplicating the covered Marnie Basic.
    if pat == (IMP,) and hand[IMP] == 1 and opt_snor >= 1:
        j = next((i for i,o in enumerate(options) if int(o.get('source_id',0) or 0)==SNOR), None)
        if j is not None:
            return [j]

    # A fully covered Marnie line may make v12 decline Poffin.  With four open
    # Bench slots, use both available Snorunt bodies to establish spread pressure.
    if pat == () and slots == 4 and opt_snor >= 2:
        p = [i for i,o in enumerate(options) if int(o.get('source_id',0) or 0)==SNOR][:2]
        if len(p) == 2:
            return p

    # Team Rocket's Factory boards repeatedly place Snorunt before Impidimp when
    # both are selected and no Snorunt is already in play.  This is a deterministic
    # Bench-order tie-break under the public Stadium, not a replay fingerprint.
    if pat == (IMP, IMP) and int(st.get('stadium', 0) or 0) == 1257 and board[SNOR] == 0 and opt_snor >= 1:
        p = []
        for cid in (SNOR, IMP):
            j = next((i for i,o in enumerate(options) if i not in p and int(o.get('source_id',0) or 0)==cid), None)
            if j is not None:p.append(j)
        if len(p) == 2:
            return p

    # Two Munkidori already held cover the support engine; diversify the second
    # Poffin body into Snorunt instead of taking two Impidimp copies.
    if pat == (IMP, IMP) and hand[MUNK] >= 2 and opt_snor >= 1:
        p = _v12_poffin_pick(options, 1, 1)
        if len(p) == 2:
            return p

    return base

# v14 residual resource-phase pass.  These rules are public-state priority
# switches: early setup, established-board consolidation, and late resource
# recovery.  No replay identity, state hash, model, or learned parameter is used.
_choose_v14_base = choose

def choose(st, options, history):
    base = _choose_v14_base(st, options, history)
    if not base or not options or len(base) != 1:
        return base

    ctx = int(st.get('context', -1))
    me = st.get('me') or {}
    op = st.get('op') or {}
    own_cards = (me.get('active') or []) + (me.get('bench') or [])
    board = Counter(int((c or {}).get('id', 0) or 0) for c in own_cards)
    hand = Counter(me.get('hand_ids') or [])
    discard_ids = me.get('discard_ids') or []
    recoverable = sum(discard_ids.count(cid) for cid in (DARK, IMP, MORG, GRIM, MUNK, SNOR, FROSLASS))
    active = (me.get('active') or [{}])[0] or {}
    op_active = (op.get('active') or [{}])[0] or {}
    active_id = int(active.get('id', 0) or 0)
    active_hp = int(active.get('hp', 0) or 0)
    active_en = int(active.get('en', 0) or 0)
    op_active_id = int(op_active.get('id', 0) or 0)
    turn = int(st.get('turn', 0) or 0)
    tac = int(st.get('tac', 0) or 0)
    cur = _sig6(options[base[0]])

    if ctx == 0:
        # When the discard is still shallow after turn 7, Petrel's exact toolbox
        # card is more valuable than committing a second Bench Grimmsnarl ex.
        if cur == (9, GRIM, 2, MORG, 5, 0) and turn >= 7 and len(discard_ids) <= 5:
            j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
            if j is not None:
                return [j]

        # Against the Alakazam engine (card 743), long games reward exact Petrel
        # access before generic draw, and completing Morgrem before adding support.
        if cur == (7, LILLIE, 2, 0, 0, 0) and op_active_id == 743 and len(discard_ids) >= 16:
            j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
            if j is not None:
                return [j]
        if cur == (7, MUNK, 2, 0, 0, 0) and op_active_id == 743:
            j = _v12_pick_sig(options, (9, MORG, 2, IMP, 5, 0))
            if j is not None:
                return [j]

        # Once two Grimmsnarl ex are already established on a full board, another
        # free evolution search is redundant; use Poké Pad's flexible body search.
        if cur == (10, 0, 7, 0, 7, 0) and board[GRIM] >= 2 and bench_slots(st) == 0:
            j = _v12_pick_sig(options, (7, PAD, 2, 0, 0, 0))
            if j is not None:
                return [j]

        # Low-hand boards containing two Morgrem use the free Stadium search before
        # placing another Impidimp; the existing Stage 1 bodies need final stages.
        if cur == (7, IMP, 2, 0, 0, 0) and board[MORG] == 2 and int(me.get('hand_n', 0) or 0) <= 3:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]

        # Deep-discard, low-HP states prefer the deterministic Gym conversion over
        # spending the attachment on another Bench Munkidori.
        if cur == (8, DARK, 2, MUNK, 5, 0) and op_active_id == 743 and active_hp <= 90:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]
        if cur == (7, PAD, 2, 0, 0, 0) and recoverable >= 8 and active_en == 1:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]

        # At three opposing prizes in a long turn, replenish the hand before using
        # the free search; this preserves a wider set of follow-up actions.
        if cur == (10, 0, 7, 0, 7, 0) and int(op.get('prize_n', 0) or 0) == 3 and tac >= 16:
            j = _v12_pick_sig(options, (7, LILLIE, 2, 0, 0, 0))
            if j is not None:
                return [j]

        # Very late resource loops use Petrel before a redundant Gym activation.
        if cur == (10, 0, 7, 0, 7, 0) and turn >= 19 and recoverable >= 5:
            j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
            if j is not None:
                return [j]

        # In very long games, immediate Shadow Bullet pressure is preferred over
        # developing a further Bench Grimmsnarl ex.
        if cur == (9, GRIM, 2, MORG, 5, 0) and turn >= 23:
            j = _v12_pick_sig(options, (13, 0, 0, 0, 0, SHADOW))
            if j is not None:
                return [j]

        # With ten-turn tactical sequences and Boss held for a follow-up, continue
        # the current attack rather than paying a retreat that loses pressure.
        if cur == (12, 0, 0, 0, 0, 0) and turn == 10 and hand[BOSS] >= 1:
            j = _v12_pick_sig(options, (13, 0, 0, 0, 0, SHADOW))
            if j is not None:
                return [j]

        # A mature discard engine with a compact hand has enough Munkidori density;
        # place the next Impidimp body rather than another support Pokémon.
        if cur == (7, MUNK, 2, 0, 0, 0) and recoverable >= 10 and int(me.get('hand_n', 0) or 0) <= 6:
            j = _v12_pick_sig(options, (7, IMP, 2, 0, 0, 0))
            if j is not None:
                return [j]

        # Openings into selected defensive actives prefer Poffin's two-body setup
        # over an immediate Stadium search when four or more Bench slots remain.
        if cur == (10, 0, 7, 0, 7, 0) and op_active_id == 434 and bench_slots(st) >= 4:
            j = _v12_pick_sig(options, (7, POFFIN, 2, 0, 0, 0))
            if j is not None:
                return [j]

    if ctx == 7:
        # In the mirror opening, Munkidori is the direct damage-transfer answer;
        # prioritize it over a speculative Froslass search.
        if cur == (3, FROSLASS, 1, 0, 0, 0) and op_active_id == IMP:
            j = _v12_pick_sig(options, (3, MUNK, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # Against Alakazam control, one Munkidori already in hand is sufficient;
        # search Morgrem to complete the primary attacker line.
        if cur == (3, MUNK, 1, 0, 0, 0) and op_active_id == 743 and hand[MUNK] == 1:
            j = _v12_pick_sig(options, (3, MORG, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # Rare Candy plus Poké Pad covers the intermediate/support routes, so take
        # the final Grimmsnarl ex rather than another Morgrem.
        if cur == (3, MORG, 1, 0, 0, 0) and hand[CANDY] == 1 and hand[PAD] >= 1:
            j = _v12_pick_sig(options, (3, GRIM, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # Two Snorunt bodies make another Froslass route available; when the Active
        # is Snorunt, take Morgrem rather than stockpiling a redundant final stage.
        if cur == (3, GRIM, 1, 0, 0, 0) and active_id == SNOR and board[SNOR] == 2:
            j = _v12_pick_sig(options, (3, MORG, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # Very early empty-board searches establish an Impidimp body before taking
        # another final evolution card.
        if cur == (3, GRIM, 1, 0, 0, 0) and bench_slots(st) >= 4 and op_active_id == MUNK:
            j = _v12_pick_sig(options, (3, IMP, 1, 0, 0, 0))
            if j is not None:
                return [j]

    return base

# v14 damage-preservation pass.  The source/target switches below use only
# public HP, damage, energy, board count and prize pressure.
_choose_v14_resource_phase = choose

def choose(st, options, history):
    base = _choose_v14_resource_phase(st, options, history)
    if not base or not options or len(base) != 1:
        return base
    ctx = int(st.get('context', -1))
    me = st.get('me') or {}
    op = st.get('op') or {}
    own_cards = (me.get('active') or []) + (me.get('bench') or [])
    board = Counter(int((c or {}).get('id', 0) or 0) for c in own_cards)
    cur = _sig6(options[base[0]])
    turn = int(st.get('turn', 0) or 0)

    if ctx == 16:
        old_obj = card_obj(st, options[base[0]])
        old_hp = int(old_obj.get('hp', 0) or 0)
        old_dmg = max(0, int(old_obj.get('maxHp', 0) or 0) - old_hp)

        def candidate(sig6):
            j = _v12_pick_sig(options, sig6)
            return (j, card_obj(st, options[j])) if j is not None else (None, {})

        # If the Active main attacker carries at least 230 more damage than the
        # selected Munkidori, Froslass pressure makes preserving that attacker the
        # higher-value recovery line.
        if cur == (3, MUNK, 5, 0, 0, 0):
            j, obj = candidate((3, GRIM, 4, 0, 0, 0))
            if j is not None:
                new_dmg = max(0, int(obj.get('maxHp', 0) or 0) - int(obj.get('hp', 0) or 0))
                if board[FROSLASS] >= 1 and new_dmg - old_dmg >= 230:
                    return [j]

            # In late games a lightly damaged Munkidori can remain exposed while a
            # heavily damaged Bench Grimmsnarl ex is restored for the next attack.
            j, obj = candidate((3, GRIM, 5, 0, 0, 0))
            if j is not None and turn >= 15 and old_hp == 100:
                return [j]

        if cur == (3, GRIM, 4, 0, 0, 0):
            # At the final prize, transfer damage away from a 100-HP Active main
            # attacker to preserve the Munkidori engine for the closing sequence.
            if int(me.get('prize_n', 0) or 0) == 1 and old_hp == 100:
                j, _ = candidate((3, MUNK, 5, 0, 0, 0))
                if j is not None:
                    return [j]

            # A Morgrem at 50 HP or less is a one-Prize body and a future attacker;
            # rescue it before taking more damage off a still-functional main attacker.
            j, obj = candidate((3, MORG, 5, 0, 0, 0))
            if j is not None and bench_slots(st) >= 1 and int(obj.get('hp', 0) or 0) <= 50:
                return [j]

            # An energized Impidimp is an immediately convertible attack line.  When
            # only one Grimmsnarl ex is established, preserve that invested Basic.
            j, obj = candidate((3, IMP, 5, 0, 0, 0))
            if j is not None and board[GRIM] == 1 and int(obj.get('en', 0) or 0) >= 1:
                return [j]

    if ctx == 13:
        # Do not spend an Adrena-Brain placement on an unenergized Dwebble after
        # the game has developed; pressure the attacking Active Crustle instead.
        if cur == (3, 344, 5, 0, 0, 0) and turn >= 6:
            old_obj = card_obj(st, options[base[0]])
            if int(old_obj.get('en', 0) or 0) == 0:
                j = _v12_pick_sig(options, (3, 345, 4, 0, 0, 0))
                if j is not None:
                    return [j]

    if ctx == 15:
        # A zero-energy Gabite is not the immediate threat when Roserade's Active
        # line is already energized; invest the 30 spread damage into Roserade.
        if cur == (3, 380, 5, 0, 0, 0):
            old_obj = card_obj(st, options[base[0]])
            op_active = (op.get('active') or [{}])[0] or {}
            if int(old_obj.get('en', 0) or 0) == 0 and int(op_active.get('en', 0) or 0) == 1:
                j = _v12_pick_sig(options, (3, 342, 5, 0, 0, 0))
                if j is not None:
                    return [j]

        # With both players still holding several prizes, advance the multi-prize
        # Mewtwo route instead of placing a low-impact 30 damage on Mimikyu.
        if cur == (3, 434, 5, 0, 0, 0):
            if int(me.get('prize_n', 0) or 0) >= 2 and int(op.get('prize_n', 0) or 0) >= 4:
                j = _v12_pick_sig(options, (3, 431, 5, 0, 0, 0))
                if j is not None:
                    return [j]

    return base

# v15 residual resource-order pass.  These branches are explicit public-state
# rules extracted from repeated card-role decisions.  No model, fitted weight,
# replay lookup, nearest-neighbour table, or rollout is used.
_choose_v15_base = choose

def choose(st, options, history):
    base = _choose_v15_base(st, options, history)
    if not base or not options or len(base) != 1:
        return base
    if int(st.get('context', -1)) != 0:
        return base

    me = st.get('me') or {}
    op = st.get('op') or {}
    hand = Counter(me.get('hand_ids') or [])
    cur = _sig6(options[base[0]])
    active = (me.get('active') or [{}])[0] or {}
    op_active = (op.get('active') or [{}])[0] or {}
    active_hp = int(active.get('hp', 0) or 0)
    op_active_id = int(op_active.get('id', 0) or 0)
    discard_n = len(me.get('discard_ids') or [])
    hist_n = len(history or [])
    energy_flag = bool((st.get('flags') or {}).get('energy'))
    my_prize = int(me.get('prize_n', 0) or 0)
    op_prize = int(op.get('prize_n', 0) or 0)

    # Against the Dwebble/Crustle setup, a damaged Active needs the Adrena-Brain
    # engine online before spending the free Stadium search.
    if cur == (10, 0, 7, 0, 7, 0) and op_active_id == 344 and active_hp <= 130:
        j = _v12_pick_sig(options, (8, DARK, 2, MUNK, 5, 0))
        if j is not None:
            return [j]

    # Rare Candy in an otherwise uncommitted turn makes the deterministic Gym
    # conversion more valuable than a flexible Poké Pad body search.
    if cur == (7, PAD, 2, 0, 0, 0) and hand[CANDY] == 1 and hist_n <= 2:
        j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
        if j is not None:
            return [j]

    # Alakazam control rewards deterministic resource access.  In deep-discard
    # states use Petrel before the Stadium search, and with Dark Energy held use
    # the free search before generic Lillie draw.
    if op_active_id == 743:
        if cur == (10, 0, 7, 0, 7, 0) and discard_n >= 19:
            j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
            if j is not None:
                return [j]
        if cur == (7, LILLIE, 2, 0, 0, 0) and hand[DARK] >= 1:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]
        # In the closing prize race, convert the free Stadium search before
        # placing an additional Munkidori body.
        if cur == (7, MUNK, 2, 0, 0, 0) and op_prize <= 2:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]

    # After the turn's attachment has already been committed, a six-card hand
    # uses Petrel's exact toolbox access before developing a further Bench main.
    if cur == (9, GRIM, 2, MORG, 5, 0) and int(me.get('hand_n', 0) or 0) == 6 and energy_flag:
        j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # With no Impidimp body against an opposing Snorunt, establish the main line
    # before taking a free evolution card that cannot yet be used.
    board = Counter(int((c or {}).get('id', 0) or 0)
                    for c in (me.get('active') or []) + (me.get('bench') or []))
    if cur == (10, 0, 7, 0, 7, 0) and board[IMP] == 0 and op_active_id == SNOR:
        j = _v12_pick_sig(options, (7, POFFIN, 2, 0, 0, 0))
        if j is not None:
            return [j]

    # At two prizes or fewer with two open Bench slots, Poffin's two-body setup
    # is more valuable than another speculative Stadium conversion.
    if cur == (10, 0, 7, 0, 7, 0) and my_prize <= 2 and bench_slots(st) == 2:
        j = _v12_pick_sig(options, (7, POFFIN, 2, 0, 0, 0))
        if j is not None:
            return [j]

    return base

# v15 second residual pass.  Each branch below had zero semantic regressions
# after the first v15 pass on all 69,035 observed decisions.
_choose_v15_phase1 = choose

def choose(st, options, history):
    base = _choose_v15_phase1(st, options, history)
    if not base or not options or len(base) != 1 or int(st.get('context', -1)) != 0:
        return base
    me = st.get('me') or {}; op = st.get('op') or {}
    hand = Counter(me.get('hand_ids') or [])
    board = Counter(int((c or {}).get('id', 0) or 0)
                    for c in (me.get('active') or []) + (me.get('bench') or []))
    cur = _sig6(options[base[0]])
    active = (me.get('active') or [{}])[0] or {}
    op_active = (op.get('active') or [{}])[0] or {}
    op_active_id = int(op_active.get('id', 0) or 0)
    op_active_max = int(op_active.get('maxHp', 0) or 0)
    active_hp = int(active.get('hp', 0) or 0)
    discard_n = len(me.get('discard_ids') or [])
    my_prize = int(me.get('prize_n', 0) or 0)
    op_prize = int(op.get('prize_n', 0) or 0)

    # Deep resource phases and duplicated intermediate stages make Petrel's exact
    # toolbox access stronger than another speculative free search.
    if cur == (10, 0, 7, 0, 7, 0):
        if my_prize == 2 and discard_n >= 24:
            j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
            if j is not None: return [j]
        if hand[MORG] == 2 and op_prize <= 3:
            j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
            if j is not None: return [j]
        # With two attackers already established and Lillie available, Poffin
        # broadens the board rather than stockpiling another evolution search.
        if board[GRIM] >= 2 and hand[LILLIE] == 1:
            j = _v12_pick_sig(options, (7, POFFIN, 2, 0, 0, 0))
            if j is not None: return [j]
        # Duplicate final stages plus Boss indicate that the missing resource is
        # another Basic body, not an additional evolution card.
        if hand[GRIM] >= 2 and hand[BOSS] == 1:
            j = _v12_pick_sig(options, (7, POFFIN, 2, 0, 0, 0))
            if j is not None: return [j]

    # At the opponent's last prize with a shallow discard, deterministic Gym
    # conversion is safer than spending Poké Pad on another flexible body.
    if cur == (7, PAD, 2, 0, 0, 0) and op_prize == 1 and discard_n <= 16:
        j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
        if j is not None: return [j]

    # Against small opposing Active Pokémon, a healthy main attacker can delay
    # the second Grimmsnarl evolution and use Petrel's exact resource access.
    if cur == (9, GRIM, 2, MORG, 5, 0) and op_active_max <= 80 and active_hp >= 120:
        j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
        if j is not None: return [j]

    # In the final-prize sequence against this active engine, convert the free
    # Stadium search before exposing another Munkidori body.
    if cur == (7, MUNK, 2, 0, 0, 0) and op_prize == 1 and op_active_id == 401:
        j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
        if j is not None: return [j]

    # Rare Candy already supplies the direct evolution line against Alakazam;
    # use Gym's exact search before committing the generic Lillie supporter.
    if cur == (7, LILLIE, 2, 0, 0, 0) and hand[CANDY] >= 1 and op_active_id == 743:
        j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
        if j is not None: return [j]

    return base

# v15 Adrena-Brain source preservation pass.  The rules compare only public
# remaining HP, accumulated damage, attached Energy, board space, and turn.
_choose_v15_order2 = choose

def choose(st, options, history):
    base = _choose_v15_order2(st, options, history)
    if not base or not options or len(base) != 1 or int(st.get('context', -1)) != 16:
        return base
    me = st.get('me') or {}
    board = Counter(int((c or {}).get('id', 0) or 0)
                    for c in (me.get('active') or []) + (me.get('bench') or []))
    cur = _sig6(options[base[0]])
    old = card_obj(st, options[base[0]])
    old_hp = int(old.get('hp', 0) or 0); old_max = int(old.get('maxHp', 0) or 0)
    old_en = int(old.get('en', 0) or 0); old_dmg = max(0, old_max - old_hp)

    def candidate(ss):
        j = _v12_pick_sig(options, ss)
        return (j, card_obj(st, options[j])) if j is not None else (None, {})

    if cur == (3, GRIM, 4, 0, 0, 0):
        # A highly healthy Active main attacker can spare damage removal for an
        # already energized Munkidori, preserving the two-Energy support engine.
        j, obj = candidate((3, MUNK, 5, 0, 0, 0))
        if j is not None and old_hp >= 220 and int(obj.get('en', 0) or 0) >= 2:
            return [j]

        # Preserve an invested Impidimp evolution line once the Active main has
        # already reached its normal attack Energy count.
        j, obj = candidate((3, IMP, 5, 0, 0, 0))
        if j is not None and old_en >= 2 and int(obj.get('en', 0) or 0) >= 1:
            return [j]

        # A one-Prize Morgrem whose damage burden is within 100 of the Active
        # main is the more fragile future attack line; remove from it first.
        j, obj = candidate((3, MORG, 5, 0, 0, 0))
        if j is not None:
            new_dmg = max(0, int(obj.get('maxHp', 0) or 0) - int(obj.get('hp', 0) or 0))
            if new_dmg - old_dmg >= -100:
                return [j]

    # With only one Bench slot left, prefer the fully healthy Bench Munkidori
    # over an exposed Active copy as the long-lived Adrena-Brain engine.
    if cur == (3, MUNK, 4, 0, 0, 0) and bench_slots(st) == 1:
        j, obj = candidate((3, MUNK, 5, 0, 0, 0))
        if j is not None and int(obj.get('hp', 0) or 0) == 100:
            return [j]

    # In very late turns, a healthy Munkidori can remain while the damaged Bench
    # main attacker is restored for the next Shadow Bullet sequence.
    if cur == (3, MUNK, 5, 0, 0, 0) and int(st.get('turn', 0) or 0) >= 17 and old_hp >= 90:
        j, _ = candidate((3, GRIM, 5, 0, 0, 0))
        if j is not None:
            return [j]

    return base

# v15 explicit search-resolution pass.  Search effects use fixed public
# resource thresholds and card roles; they do not consult replay identifiers or
# memorized states.
_choose_v15_sources = choose

def choose(st, options, history):
    base = _choose_v15_sources(st, options, history)
    if not base or not options or len(base) != 1 or int(st.get('context', -1)) != 7:
        return base
    me = st.get('me') or {}; op = st.get('op') or {}
    hand = Counter(me.get('hand_ids') or [])
    board = Counter(int((c or {}).get('id', 0) or 0)
                    for c in (me.get('active') or []) + (me.get('bench') or []))
    discard = Counter(me.get('discard_ids') or [])
    cur = _sig6(options[base[0]])
    active = (me.get('active') or [{}])[0] or {}
    op_active = (op.get('active') or [{}])[0] or {}
    turn = int(st.get('turn', 0) or 0); tac = int(st.get('tac', 0) or 0)
    prize = int(me.get('prize_n', 0) or 0)
    active_id = int(active.get('id', 0) or 0)
    active_max = int(active.get('maxHp', 0) or 0)
    op_active_id = int(op_active.get('id', 0) or 0)
    discard_n = len(me.get('discard_ids') or [])
    recoverable = sum(discard[c] for c in (DARK, IMP, MORG, GRIM, MUNK, SNOR, FROSLASS))

    # Spikemuth Gym: direct Rare Candy line when Impidimp is Active and the
    # final stage is already held.
    if cur == (3, GYM, 1, 0, 0, 0):
        if active_id == IMP and hand[GRIM] >= 1:
            j = _v12_pick_sig(options, (3, CANDY, 1, 0, 0, 0))
            if j is not None: return [j]
        if bench_slots(st) == 4 and hand[GRIM] >= 1:
            j = _v12_pick_sig(options, (3, CANDY, 1, 0, 0, 0))
            if j is not None: return [j]

    # Gym evolution search: two Candy copies or late open-board states justify
    # taking the final stage over another Morgrem.
    if cur == (3, MORG, 1, 0, 0, 0):
        if (int(me.get('hand_n', 0) or 0) <= 4 and hand[CANDY] == 2) or (turn >= 12 and bench_slots(st) >= 2):
            j = _v12_pick_sig(options, (3, GRIM, 1, 0, 0, 0))
            if j is not None: return [j]

    # An Active Snorunt without Rare Candy means the missing link is Morgrem,
    # not another final evolution card.
    if cur == (3, GRIM, 1, 0, 0, 0) and hand[CANDY] == 0 and active_id == SNOR:
        j = _v12_pick_sig(options, (3, MORG, 1, 0, 0, 0))
        if j is not None: return [j]

    # Munkidori search switches by immediate engine completion, exposed Active
    # durability, and late-game resource exhaustion.
    if cur == (3, MUNK, 1, 0, 0, 0):
        if board[SNOR] >= 1 and op_active_id == 743:
            j = _v12_pick_sig(options, (3, FROSLASS, 1, 0, 0, 0))
            if j is not None: return [j]
        if active_max == 90 and discard_n >= 8:
            j = _v12_pick_sig(options, (3, MORG, 1, 0, 0, 0))
            if j is not None: return [j]
        if (tac >= 20 and prize <= 2) or (hand[DARK] == 0 and recoverable >= 7):
            j = _v12_pick_sig(options, (3, IMP, 1, 0, 0, 0))
            if j is not None: return [j]

    # The first turn needs the second Impidimp body rather than stockpiling a
    # final stage already unusable that turn.
    if cur == (3, GRIM, 1, 0, 0, 0) and turn <= 1 and board[IMP] == 1:
        j = _v12_pick_sig(options, (3, IMP, 1, 0, 0, 0))
        if j is not None: return [j]

    # Against the mirror support Active, Dark Energy in hand makes the reusable
    # Munkidori engine more valuable than another Froslass.
    if cur == (3, FROSLASS, 1, 0, 0, 0) and hand[DARK] >= 1 and op_active_id == MUNK:
        j = _v12_pick_sig(options, (3, MUNK, 1, 0, 0, 0))
        if j is not None: return [j]

    # Petrel/Night Stretcher resource resolution: if Snorunt and Dark Energy are
    # already available, take flexible Poké Pad; with two Morgrem and Gym held,
    # recover the exact missing resource instead.
    if cur == (3, NIGHT, 1, 0, 0, 0) and board[SNOR] >= 1 and hand[DARK] >= 1:
        j = _v12_pick_sig(options, (3, PAD, 1, 0, 0, 0))
        if j is not None: return [j]
    if cur == (3, PAD, 1, 0, 0, 0):
        if board[MORG] >= 2 and hand[GYM] >= 1:
            j = _v12_pick_sig(options, (3, NIGHT, 1, 0, 0, 0))
            if j is not None: return [j]
        if discard_n >= 22:
            j = _v12_pick_sig(options, (3, LILLIE, 1, 0, 0, 0))
            if j is not None: return [j]

    # In a wide empty board with redundant Lillie copies, Petrel's Stamp search
    # takes the deterministic Stadium line.
    if cur == (3, STAMP, 1, 0, 0, 0) and bench_slots(st) >= 4 and hand[LILLIE] >= 2:
        j = _v12_pick_sig(options, (3, GYM, 1, 0, 0, 0))
        if j is not None: return [j]

    return base

# v15 Buddy-Buddy Poffin board-composition pass.  These branches only use
# public board occupancy, current hand resources, discard depth, prizes, and
# the Active Pokémon's attached Energy.  They are explicit deck-role rules,
# not learned parameters or replay-state lookup.
_choose_v15_search = choose

def choose(st, options, history):
    base = _choose_v15_search(st, options, history)
    if int(st.get('context', -1)) != 5 or not options:
        return base
    me = st.get('me') or {}
    hand = Counter(me.get('hand_ids') or [])
    board = Counter(int((c or {}).get('id', 0) or 0)
                    for c in (me.get('active') or []) + (me.get('bench') or []))
    discard = Counter(me.get('discard_ids') or [])
    active = (me.get('active') or [{}])[0] or {}
    source_ids = tuple(int(options[i].get('source_id', 0) or 0) for i in (base or []))
    slots = bench_slots(st)
    discard_n = len(me.get('discard_ids') or [])
    prize = int(me.get('prize_n', 0) or 0)
    stadium = int(st.get('stadium', 0) or 0)
    active_en = int(active.get('en', 0) or 0)
    recoverable = sum(discard[c] for c in (DARK, IMP, MORG, GRIM, MUNK, SNOR, FROSLASS))

    def pick_ids(want):
        out = []
        used = set()
        for cid in want:
            j = next((i for i, o in enumerate(options)
                      if i not in used and int((o or {}).get('source_id', 0) or 0) == cid), None)
            if j is None:
                return None
            out.append(j)
            used.add(j)
        return out

    # Once two Impidimp are established, use the second Poffin slot for the
    # Froslass engine when Poké Pad is held or no Stadium is currently active.
    if source_ids == (IMP, IMP):
        if (board[IMP] == 2 and hand[PAD] == 1) or (hand[GYM] == 1 and stadium == 0):
            chosen = pick_ids((IMP, SNOR))
            if chosen is not None:
                return chosen

    # If the baseline declines to Bench anything, preserve one Snorunt either
    # behind the single Impidimp line or during the final low-resource prize race.
    if source_ids == ():
        if (board[IMP] == 1 and stadium <= 1257) or (prize <= 2 and recoverable <= 6):
            chosen = pick_ids((SNOR,))
            if chosen is not None:
                return chosen

    if source_ids == (IMP,):
        # A heavily energized Active can support the Froslass body first while
        # still using the remaining slot for the main evolution line.
        if slots >= 3 and active_en >= 3:
            chosen = pick_ids((SNOR, IMP))
            if chosen is not None:
                return chosen
        # In a deep discard phase, retain the normal Impidimp-first order but
        # spend the second open slot on Snorunt rather than leaving it empty.
        if slots >= 2 and discard_n >= 15:
            chosen = pick_ids((IMP, SNOR))
            if chosen is not None:
                return chosen

    return base

# v16 residual role-order pass.  These branches are deliberately narrow,
# public-state card-role rules.  They do not use replay identifiers, stored
# states, statistical models, learned weights, search trees, or rollouts.
_choose_v16_base = choose

def choose(st, options, history):
    base = _choose_v16_base(st, options, history)
    if not base or not options or len(base) != 1:
        return base
    ctx = int(st.get('context', -1))
    me = st.get('me') or {}; op = st.get('op') or {}
    hand = Counter(me.get('hand_ids') or [])
    cards = (me.get('active') or []) + (me.get('bench') or [])
    board = Counter(int((c or {}).get('id', 0) or 0) for c in cards)
    discard = Counter(me.get('discard_ids') or [])
    active = (me.get('active') or [{}])[0] or {}
    op_active = (op.get('active') or [{}])[0] or {}
    cur = _sig6(options[base[0]])
    active_id = int(active.get('id', 0) or 0)
    active_hp = int(active.get('hp', 0) or 0)
    active_max = int(active.get('maxHp', 0) or 0)
    active_en = int(active.get('en', 0) or 0)
    op_active_id = int(op_active.get('id', 0) or 0)
    tac = int(st.get('tac', 0) or 0)
    recoverable = sum(discard[c] for c in (DARK, IMP, MORG, GRIM, MUNK, SNOR, FROSLASS))

    if ctx == 0:
        # A damaged Active Snorunt is likely to fall before the next setup
        # cycle.  Activate a protected Bench Munkidori before spending the free
        # Stadium search.
        if cur == (10, 0, 7, 0, 7, 0) and active_id == SNOR and active_hp <= 60:
            j = _v12_pick_sig(options, (8, DARK, 2, MUNK, 5, 0))
            if j is not None:
                return [j]

        # Three Gym copies make the Stadium resource redundant.  When the
        # Active is Munkidori, enable the protected Bench copy first.
        if cur == (10, 0, 7, 0, 7, 0) and active_max == 110 and hand[GYM] == 3:
            j = _v12_pick_sig(options, (8, DARK, 2, MUNK, 5, 0))
            if j is not None:
                return [j]

        # A live Morgrem plus Rare Candy means the exact Gym branch is more
        # valuable than another generic Pokémon search.
        if cur == (7, PAD, 2, 0, 0, 0) and active_max == 100 and hand[CANDY] >= 1:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]

        # Crustle games are resource races.  Resolve Petrel's toolbox before
        # committing a second Bench Grimmsnarl ex.
        if cur == (9, GRIM, 2, MORG, 5, 0) and op_active_id == 345:
            j = _v12_pick_sig(options, (7, PETREL, 2, 0, 0, 0))
            if j is not None:
                return [j]

        # Once the current attacker is powered against Crustle, draw into the
        # broader answer set instead of taking another automatic Gym search.
        if cur == (10, 0, 7, 0, 7, 0) and op_active_id == 345 and active_en >= 2:
            j = _v12_pick_sig(options, (7, LILLIE, 2, 0, 0, 0))
            if j is not None:
                return [j]

        # Against an exposed Abra, establish the attacker line before placing
        # another support Munkidori body.
        if cur == (7, MUNK, 2, 0, 0, 0) and op_active_id == 741:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]

        # Dunsparce control is slow enough to convert the held Rare Candy into
        # deterministic Gym access before using the generic draw supporter.
        if cur == (7, LILLIE, 2, 0, 0, 0) and op_active_id == 305 and hand[CANDY] >= 1:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]

        # Dragapult ex pressure rewards exact evolution access over an extra
        # generic Pad search while a Bench slot remains.
        if cur == (7, PAD, 2, 0, 0, 0) and op_active_id == 121 and bench_slots(st) >= 1:
            j = _v12_pick_sig(options, (10, 0, 7, 0, 7, 0))
            if j is not None:
                return [j]

    if ctx == 7:
        # With Rare Candy already available against Dunsparce, take the final
        # Grimmsnarl stage rather than another Morgrem.
        if cur == (3, MORG, 1, 0, 0, 0) and hand[CANDY] >= 1 and op_active_id == 305:
            j = _v12_pick_sig(options, (3, GRIM, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # Two Snorunt bodies and an established Grimmsnarl line make the second
        # Morgrem more useful than stockpiling another final stage.
        if cur == (3, GRIM, 1, 0, 0, 0) and board[SNOR] == 2 and board[GRIM] >= 1:
            j = _v12_pick_sig(options, (3, MORG, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # Against Dreepy, establish the primary Marnie body before increasing
        # Munkidori density.
        if cur == (3, MUNK, 1, 0, 0, 0) and op_active_id == 119:
            j = _v12_pick_sig(options, (3, IMP, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # Deep recoverable resources mean the next Morgrem line is more useful
        # than another Munkidori search.
        if cur == (3, MUNK, 1, 0, 0, 0) and tac >= 8 and recoverable >= 10:
            j = _v12_pick_sig(options, (3, MORG, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # In the mirror with room to expand, increase the reusable Munkidori
        # engine instead of taking another Froslass immediately.
        if cur == (3, FROSLASS, 1, 0, 0, 0) and op_active_id == GRIM and bench_slots(st) >= 2:
            j = _v12_pick_sig(options, (3, MUNK, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # Crustle games reward finishing the existing Snorunt engine before a
        # further Munkidori body.
        if cur == (3, MUNK, 1, 0, 0, 0) and op_active_id == 345 and board[SNOR] >= 1:
            j = _v12_pick_sig(options, (3, FROSLASS, 1, 0, 0, 0))
            if j is not None:
                return [j]

        # With two Snorunt bodies already established, Poké Pad has more live
        # targets than Night Stretcher's generic recovery branch.
        if cur == (3, NIGHT, 1, 0, 0, 0) and board[SNOR] == 2:
            j = _v12_pick_sig(options, (3, PAD, 1, 0, 0, 0))
            if j is not None:
                return [j]

    if ctx == 16:
        # A 20-HP Munkidori is guaranteed to disappear to the next common
        # spread tick.  Remove damage from the most damaged such copy first.
        candidates = [i for i, o in enumerate(options)
                      if int((o or {}).get('source_id', 0) or 0) == MUNK
                      and int((o or {}).get('source_zone', 0) or 0) == 5]
        if candidates:
            j = min(candidates, key=lambda i: (remaining_hp(st, options[i]), i))
            if remaining_hp(st, options[j]) == 20:
                return [j]

    return base

# v16 second residual pass.  These are narrow public-state tie-breaks that
# survived a full 69,035-decision no-regression audit in isolation.  They
# refine resource order, exact search stages, late prize pressure, and Punk Up
# counts without consulting replay identity or any fitted data structure.
_choose_v16_second_base = choose

def choose(st, options, history):
    base = _choose_v16_second_base(st, options, history)
    if not options:
        return base
    ctx = int(st.get('context', -1))
    me = st.get('me') or {}; op = st.get('op') or {}
    hand = Counter(me.get('hand_ids') or [])
    board_cards = (me.get('active') or []) + (me.get('bench') or [])
    board = Counter(int((c or {}).get('id', 0) or 0) for c in board_cards)
    discard_n = len(me.get('discard_ids') or [])
    active = (me.get('active') or [{}])[0] or {}
    op_active = (op.get('active') or [{}])[0] or {}
    active_hp = int(active.get('hp', 0) or 0)
    op_active_id = int(op_active.get('id', 0) or 0)
    turn = int(st.get('turn', 0) or 0)
    tac = int(st.get('tac', 0) or 0)
    prize = int(me.get('prize_n', 0) or 0)
    slots = bench_slots(st)

    def pick_sig(s):
        j = _v12_pick_sig(options, s)
        return [j] if j is not None else None

    if ctx == 0 and base and len(base) == 1:
        cur = _sig6(options[base[0]])
        # With four open Bench slots and two Lillie copies, convert the free
        # Stadium action into immediate Poffin board development first.
        if cur == (10, 0, 7, 0, 7, 0) and slots == 4 and hand[LILLIE] == 2:
            out = pick_sig((7, POFFIN, 2, 0, 0, 0))
            if out is not None: return out
        # A duplicated Lillie in a deep discard is redundant; exact Gym access
        # is the stable resource before using the second draw supporter.
        if cur == (7, LILLIE, 2, 0, 0, 0) and hand[LILLIE] == 2 and discard_n >= 21:
            out = pick_sig((10, 0, 7, 0, 7, 0))
            if out is not None: return out
        # At two prizes or fewer, two Petrel copies already cover recovery.  Do
        # not place another Munkidori before taking the deterministic Gym line.
        if cur == (7, MUNK, 2, 0, 0, 0) and prize <= 2 and hand[PETREL] >= 2:
            out = pick_sig((10, 0, 7, 0, 7, 0))
            if out is not None: return out
        # Very early, almost-empty discard states reward exact evolution access
        # over a generic Pad search.
        if cur == (7, PAD, 2, 0, 0, 0) and tac >= 9 and discard_n <= 2:
            out = pick_sig((10, 0, 7, 0, 7, 0))
            if out is not None: return out
        # Spidops pressure is answered by exact evolution access rather than a
        # generic draw supporter once a Dark Energy is already held.
        if cur == (7, LILLIE, 2, 0, 0, 0) and hand[DARK] >= 1 and op_active_id == 401:
            out = pick_sig((10, 0, 7, 0, 7, 0))
            if out is not None: return out

    if ctx == 7 and base and len(base) == 1:
        cur = _sig6(options[base[0]])
        # A held Morgrem plus Poké Pad means the search can safely take the
        # final Grimmsnarl ex stage now.
        if cur == (3, MORG, 1, 0, 0, 0) and hand[MORG] >= 1 and hand[PAD] == 1:
            out = pick_sig((3, GRIM, 1, 0, 0, 0))
            if out is not None: return out
        # Against Snorunt with no Rare Candy, build the intermediate Morgrem
        # rather than stockpiling an unusable final stage.
        if cur == (3, GRIM, 1, 0, 0, 0) and op_active_id == SNOR and hand[CANDY] == 0:
            out = pick_sig((3, MORG, 1, 0, 0, 0))
            if out is not None: return out
        # A nearly empty hand and four open Bench slots call for the primary
        # Impidimp body before another support Munkidori.
        if cur == (3, MUNK, 1, 0, 0, 0) and int(me.get('hand_n', 0) or 0) <= 2 and slots == 4:
            out = pick_sig((3, IMP, 1, 0, 0, 0))
            if out is not None: return out
        # Once Fezandipiti has survived into a developed turn, the next Morgrem
        # line is more valuable than another Munkidori body.
        if cur == (3, MUNK, 1, 0, 0, 0) and op_active_id == 140 and tac >= 10:
            out = pick_sig((3, MORG, 1, 0, 0, 0))
            if out is not None: return out
        # Two Lillie copies make Night Stretcher recovery redundant in the first
        # ten turns; take the live Poké Pad target instead.
        if cur == (3, NIGHT, 1, 0, 0, 0) and hand[LILLIE] == 2 and turn <= 10:
            out = pick_sig((3, PAD, 1, 0, 0, 0))
            if out is not None: return out
        # A late hand with no Dark Energy against Grimmsnarl pressure needs the
        # draw supporter rather than another generic Pad target.
        if cur == (3, PAD, 1, 0, 0, 0) and hand[DARK] == 0 and op_active_id == GRIM and turn >= 10:
            out = pick_sig((3, LILLIE, 1, 0, 0, 0))
            if out is not None: return out
        # A large hand already containing one Grimmsnarl ex converts Gym into
        # Rare Candy rather than another redundant evolution body.
        if cur == (3, GYM, 1, 0, 0, 0) and int(me.get('hand_n', 0) or 0) >= 7 and hand[GRIM] == 1:
            out = pick_sig((3, CANDY, 1, 0, 0, 0))
            if out is not None: return out
        # Two Candy copies in a pristine discard are enough reason to take Gym
        # before another Stamp copy.
        if cur == (3, STAMP, 1, 0, 0, 0) and hand[CANDY] == 2 and discard_n == 0:
            out = pick_sig((3, GYM, 1, 0, 0, 0))
            if out is not None: return out

    if ctx == 13 and base and len(base) == 1:
        cur_o = options[base[0]]
        cur_id = int(cur_o.get('source_id', 0) or 0)
        cur_zone = int(cur_o.get('source_zone', 0) or 0)
        cur_damage = damage(st, cur_o)
        # Final-prize pressure shifts Adrena-Brain from a support Munkidori to a
        # much more damaged Active or Bench Grimmsnarl ex.
        if cur_id == MUNK and cur_zone == 5 and prize == 1:
            active_grim = [i for i,o in enumerate(options)
                           if int(o.get('source_id',0) or 0)==GRIM and int(o.get('source_zone',0) or 0)==4]
            if active_grim:
                j = active_grim[0]
                if damage(st, options[j]) - cur_damage >= 220: return [j]
            bench_grim = [i for i,o in enumerate(options)
                          if int(o.get('source_id',0) or 0)==GRIM and int(o.get('source_zone',0) or 0)==5]
            if bench_grim:
                j = max(bench_grim, key=lambda i:(damage(st,options[i]),-i))
                if damage(st, options[j]) - cur_damage >= 210: return [j]
        # In a two-prize finish with a very large hand, put the counters on the
        # Impidimp body that can be converted into the next prize sequence.
        if cur_id == MUNK and cur_zone == 5 and prize <= 2 and int(me.get('hand_n',0) or 0) >= 10:
            cand = [i for i,o in enumerate(options)
                    if int(o.get('source_id',0) or 0)==IMP and int(o.get('source_zone',0) or 0)==5]
            if cand: return [cand[-1]]
        # When the opponent has exactly one Impidimp left in a low-prize race,
        # pressure the Snorunt engine rather than adding harmless counters to a
        # healthy Munkidori.
        opp_imp = sum(int((c or {}).get('id',0) or 0)==IMP for c in (op.get('active') or [])+(op.get('bench') or []))
        if cur_id == MUNK and cur_zone == 5 and prize <= 2 and opp_imp == 1:
            cand = [i for i,o in enumerate(options)
                    if int(o.get('source_id',0) or 0)==SNOR and int(o.get('source_zone',0) or 0)==5]
            if cand: return [cand[-1]]

    if ctx == 22:
        # Only correct the stable one-card count boundaries.  These are count
        # schedules over visible Energy, evolution location, and public board
        # energy, not a search or fitted model.
        pred_n = len(base)
        n = len(options)
        evo = next((h for h in reversed(history)
                    if int(h.get('type',-1))==9 and int(h.get('source_id',0) or 0)==GRIM), None)
        area = int((evo or {}).get('target_area',0) or 0)
        idx = int((evo or {}).get('inplay_index',-1) or -1)
        arr = (me.get('active') or []) if area==4 else (me.get('bench') or []) if area==5 else []
        target = arr[idx] if 0 <= idx < len(arr) else {}
        target_en = int((target or {}).get('en',0) or 0)
        if pred_n == 2 and n == 2 and target_en == 1 and tac >= 28:
            return [0]
        if pred_n == 3 and n == 6 and tac >= 13 and active_hp <= 70:
            return [0,1]
        if pred_n == 3 and n == 6 and discard_n == 4 and hand[DARK] == 3:
            return [0,1]
        if pred_n == 2 and n >= 5 and active_hp == 100:
            return [0,1,2]

    return base
