from __future__ import annotations
"""Narrow public-state residuals.

Each rule is asymmetric: it may replace only a low-commitment baseline action,
never a visible attack, evolution, ability, or high-value disruption line.
"""
import policy_features as pf
import human_memory as hm
import human_controller as hc
DARK=7; MUNKIDORI=112; IMPIDIMP=646; MORGREM=647; GRIM=648; FROSLASS=104
STAMP=1080; POKEGEAR=1122; POKEPAD=1152; STADIUM=1259; STRETCHER=1097; POFFIN=1086
SUPPORTERS={1182,1219,1227,1231}
LOW_MAIN={POKEGEAR,POKEPAD,STADIUM,POFFIN,STRETCHER,IMPIDIMP,MUNKIDORI,FROSLASS}
OGERPON=96; BOSS=1182
_PENDING_BOSS_SERIAL=None
_PENDING_FINAL_SERIAL=None
_PENDING_FINAL_COUNTERS=None
_PENDING_SHADOW_SERIAL=None

def reset():
 global _PENDING_BOSS_SERIAL,_PENDING_FINAL_SERIAL,_PENDING_FINAL_COUNTERS,_PENDING_SHADOW_SERIAL
 _PENDING_BOSS_SERIAL=None
 _PENDING_FINAL_SERIAL=None
 _PENDING_FINAL_COUNTERS=None
 _PENDING_SHADOW_SERIAL=None

def _legal(obs,a):return hc._legal(obs,a)
def _cid(x):return hc._cid(x)
def _players(obs):return hc._players(obs)
def _inplay(p):return hc._inplay(p)
def _energy(x):return hc._energy(x)
def _hp(x):return hc._hp(x)
def _damage(x):return hc._damage(x)
def _card(obs,o):return hc._card_at(obs,o)
def _target(obs,o):return hc._target_at(obs,o)
def _effect(obs):
 s=obs.get('select') or {};return _cid(s.get('effect')) or _cid(s.get('contextCard'))
def _sem(obs,i):
 o=(obs.get('select') or {}).get('option') or []
 return pf.semantic(obs,o[i]) if 0<=i<len(o) else {}

def _blank(obs):
 s=obs.get('select') or {}
 if int(s.get('minCount',0) or 0)==0 and _legal(obs,[]):return []
 for i,o in enumerate(s.get('option') or []):
  if _cid(_card(obs,o))==0:
   a=[i]
   if _legal(obs,a):return a
 return None


def _pokegear_take_only_supporter(obs,baseline):
 s=obs.get('select') or {};ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 if ctx!=7 or _effect(obs)!=POKEGEAR:return None
 opts=s.get('option') or []
 if not (baseline==[] or (isinstance(baseline,list) and len(baseline)==1 and 0<=baseline[0]<len(opts) and _cid(_card(obs,opts[baseline[0]]))==0)):return None
 cand=[i for i,o in enumerate(opts) if _cid(_card(obs,o)) in SUPPORTERS]
 if len(cand)!=1:return None
 a=[cand[0]]
 return a if _legal(obs,a) and a!=baseline else None

def _pokegear_whiff(obs,mem):
 s=obs.get('select') or {};ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 if ctx!=7 or _effect(obs)!=POKEGEAR:return None
 if mem.get('profile') not in ('wall','grass_fast','grass_control') or float(mem.get('confidence',0) or 0)<.45:return None
 seen=mem.get('seen') or {}
 if not any(int(seen.get(c,0) or 0)>0 for c in (96,25,344,345,117,756)):return None
 ids=[_cid(_card(obs,o)) for o in s.get('option') or []]
 if any(c in SUPPORTERS for c in ids):return None
 return _blank(obs)

def _baseline_low(obs,baseline):
 if not isinstance(baseline,list) or len(baseline)!=1:return False
 q=_sem(obs,baseline[0]);typ=int(q.get('type',-1));sid=int(q.get('source_id',0))
 return typ==7 and sid in LOW_MAIN

def _wall_attach(obs,baseline,mem):
 s=obs.get('select') or {};ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 if ctx!=0 or mem.get('profile')!='wall' or float(mem.get('confidence',0) or 0)<.75:return None
 if not _baseline_low(obs,baseline):return None
 _,me,opp=_players(obs)
 # Do not force another Energy after the damage-transfer engine is already online.
 if sum(1 for x in _inplay(me) if _cid(x)==MUNKIDORI and _energy(x)>0)>=2:return None
 cand=[]
 for i,o in enumerate(s.get('option') or []):
  q=pf.semantic(obs,o);t=_target(obs,o)
  if int(q.get('type',-1))==8 and _cid(t)==MUNKIDORI and _energy(t)==0:
   # Mature copies first; Active is slightly safer against an immediate gust.
   mature=0 if t.get('appearThisTurn',False) else 1
   active=1 if int(q.get('inplay_area',0) or 0)==4 else 0
   cand.append((mature,active,-int(t.get('serial',0) or 0),i))
 if not cand:return None
 a=[max(cand)[-1]]
 return a if _legal(obs,a) else None

def _wall_damage_source(obs,baseline,mem):
 s=obs.get('select') or {};ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 if ctx!=16:return None
 opts=s.get('option') or []; urgent=[]
 for i,o in enumerate(opts):
  c=_card(obs,o)
  # Public-state survival invariant: remove damage from a Munkidori that is
  # already one small hit from being lost.  This is matchup-independent and
  # leaves all non-urgent transfer choices to the stable baseline.
  if _cid(c)==MUNKIDORI and _damage(c)>=60 and _hp(c)<=50:urgent.append((_damage(c),-_hp(c),-int(c.get('serial',0) or 0),i))
 if not urgent:return None
 a=[max(urgent)[-1]]
 return a if _legal(obs,a) else None

def _active(p):
 a=(p or {}).get('active') or []
 return a[0] if a and isinstance(a[0],dict) else None

def _pinsir_reserve_energy(obs,mem):
 s=obs.get('select') or {};opts=s.get('option') or []
 ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 if ctx!=21 or _effect(obs)!=GRIM or mem.get('profile')!='grass_control' or float(mem.get('confidence',0) or 0)<.75:return None
 _,me,opp=_players(obs);oa=_active(opp)
 if _cid(oa)!=25 or _energy(oa)<2:return None
 # Against the delayed-KO attacker, bank surplus acceleration on a reserve
 # Grimmsnarl instead of the exposed one-prize Active.
 ma=_active(me)
 if _cid(ma)==GRIM and _energy(ma)<2:return None
 cand=[]
 for i,o in enumerate(opts):
  c=_card(obs,o)
  if _cid(c)==GRIM and int(c.get('serial',-1) or -1)!=int((ma or {}).get('serial',-2) or -2):
   cand.append((-_energy(c),-int(c.get('serial',0) or 0),i))
 if not cand:return None
 a=[max(cand)[-1]]
 return a if _legal(obs,a) else None

def _boss_finish(obs,mem):
 global _PENDING_BOSS_SERIAL
 s=obs.get('select') or {};opts=s.get('option') or []
 ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 effect=_effect(obs)
 # Complete a previously selected gust with the frozen target.
 if _PENDING_BOSS_SERIAL is not None and (ctx==3 or effect==BOSS):
  for i,o in enumerate(opts):
   c=_card(obs,o)
   if int((c or {}).get('serial',-1) or -1)==_PENDING_BOSS_SERIAL:
    a=[i];_PENDING_BOSS_SERIAL=None
    return a if _legal(obs,a) else None
  _PENDING_BOSS_SERIAL=None
 if ctx!=0 or mem.get('profile') not in ('grass_fast','grass_control') or float(mem.get('confidence',0) or 0)<.75:return None
 _,me,opp=_players(obs);ma=_active(me);oa=_active(opp)
 if _cid(ma)!=GRIM or _energy(ma)<2:return None
 # Only the final two-prize line is forced. Other gust decisions stay with the baseline.
 if len((me or {}).get('prize') or [])>2:return None
 if not any(int(_sem(obs,i).get('type',-1))==13 for i in range(len(opts))):return None
 boss_i=None
 for i in range(len(opts)):
  q=_sem(obs,i)
  if int(q.get('type',-1))==7 and int(q.get('source_id',0))==BOSS:boss_i=i;break
 if boss_i is None:return None
 targets=[]
 for c in (opp or {}).get('bench') or []:
  if not isinstance(c,dict) or _cid(c)!=OGERPON:continue
  hp=int(c.get('hp',0) or 0);ser=int(c.get('serial',-1) or -1)
  if 0<hp<=180:targets.append((hp,-ser,ser))
 if not targets:return None
 # Do not gust away an equivalent two-prize Active knockout.
 if _cid(oa)==OGERPON and 0<int((oa or {}).get('hp',0) or 0)<=180:return None
 _PENDING_BOSS_SERIAL=min(targets)[-1]
 a=[boss_i]
 return a if _legal(obs,a) else None



ABILITY_IDS={28,36,37,44,49,56,57,66,72,74,75,79,80,81,83,86,90,93,96,98,100,102,104,106,107,109,112,116,117,118,120,122,123,125,126,132,133,135,139,140,141,142,144,147,150,155,156,158,159,167,170,173,174,175,180,182,184,190,198,202,203,205,207,209,210,211,214,221,225,230,232,238,240,247,249,250,255,256,259,262,269,271,272,279,283,287,290,293,297,304,310,315,317,322,326,330,340,342,343,345,351,353,356,357,359,362,380,383,392,401,414,416,424,428,431,436,439,442,449,457,458,461,475,481,487,497,504,505,512,525,530,533,537,542,547,558,564,569,576,596,598,604,618,623,631,637,641,648,652,666,674,675,685,688,698,705,710,711,713,716,725,742,743,748,750,755,756,766,772,784,788,795,799,806,813,818,824,829,834,835,847,851,854,856,858,859,866,871,882,886,896,898,901,903,904,911,915,924,962,968,970,976,993,994,1009,1019,1022,1024,1027,1029,1033,1036,1040,1045,1052,1054,1059,1071,1099,1136,1138,1150,1151}

def _final_prize_damage_map(obs,baseline):
 """Convert an already-started damage-transfer ability into a public exact win."""
 global _PENDING_FINAL_SERIAL,_PENDING_FINAL_COUNTERS
 s=obs.get('select') or {}
 ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 if _effect(obs)!=MUNKIDORI:
  if ctx==0:
   _PENDING_FINAL_SERIAL=None;_PENDING_FINAL_COUNTERS=None
  return None
 _,me,opp=_players(obs);oa=_active(opp)
 if not isinstance(oa,dict) or len((me or {}).get('prize') or [])>1:
  _PENDING_FINAL_SERIAL=None;_PENDING_FINAL_COUNTERS=None;return None
 hp=_hp(oa)
 if not (0<hp<=30):return None
 froslass_live=any(_cid(c)==FROSLASS and _hp(c)>0 for c in _inplay(me))
 checkup=10 if froslass_live and _cid(oa) in ABILITY_IDS else 0
 need_damage=max(0,hp-checkup)
 need_counters=max(1,(need_damage+9)//10)
 if need_counters>3:return None
 if ctx==16:
  cand=[]
  for i,o in enumerate(s.get('option') or []):
   c=_card(obs,o);d=_damage(c)
   if isinstance(c,dict) and d>=10*need_counters:
    cand.append((d,-_hp(c),-int(c.get('serial',0) or 0),i))
  if not cand:return None
  a=[max(cand)[-1]]
  _PENDING_FINAL_SERIAL=int(oa.get('serial',-1) or -1)
  _PENDING_FINAL_COUNTERS=need_counters
  if isinstance(baseline,list) and len(baseline)==1:
   bc=_card(obs,(s.get('option') or [])[baseline[0]]) if 0<=baseline[0]<len(s.get('option') or []) else None
   if isinstance(bc,dict) and _damage(bc)>=10*need_counters:return None
  return a if _legal(obs,a) and a!=baseline else None
 if ctx==40 and _PENDING_FINAL_SERIAL is not None and _PENDING_FINAL_COUNTERS is not None:
  if isinstance(baseline,list) and len(baseline)==1 and 0<=baseline[0]<len(s.get('option') or []):
   if int((s.get('option') or [])[baseline[0]].get('number',0) or 0)>=_PENDING_FINAL_COUNTERS:return None
  for i,o in enumerate(s.get('option') or []):
   if int(o.get('number',0) or 0)==_PENDING_FINAL_COUNTERS:
    a=[i]
    return a if _legal(obs,a) and a!=baseline else None
  _PENDING_FINAL_SERIAL=None;_PENDING_FINAL_COUNTERS=None;return None
 if ctx==13 and _PENDING_FINAL_SERIAL is not None:
  if isinstance(baseline,list) and len(baseline)==1 and 0<=baseline[0]<len(s.get('option') or []):
   bc=_card(obs,(s.get('option') or [])[baseline[0]])
   if int((bc or {}).get('serial',-2) or -2)==_PENDING_FINAL_SERIAL:
    _PENDING_FINAL_SERIAL=None;_PENDING_FINAL_COUNTERS=None;return None
  for i,o in enumerate(s.get('option') or []):
   c=_card(obs,o)
   if int((c or {}).get('serial',-2) or -2)==_PENDING_FINAL_SERIAL:
    a=[i];_PENDING_FINAL_SERIAL=None;_PENDING_FINAL_COUNTERS=None
    return a if _legal(obs,a) and a!=baseline else None
  _PENDING_FINAL_SERIAL=None;_PENDING_FINAL_COUNTERS=None
 return None

def _shadow_bullet_double_ko(obs,baseline):
 s=obs.get('select') or {};ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 if ctx!=0:return None
 _,me,opp=_players(obs);ma=_active(me);oa=_active(opp)
 # Evolve first only when the public board shows a complete two-target knockout.
 if _cid(ma)!=MORGREM or _energy(ma)<2 or not (0<_hp(oa)<=180):return None
 if not any(isinstance(c,dict) and _cid(c) not in (96,117) and 0<_hp(c)<=30 for c in (opp or {}).get('bench') or []):return None
 for i,o in enumerate(s.get('option') or []):
  q=pf.semantic(obs,o)
  if int(q.get('type',-1))==9 and int(q.get('source_id',0) or 0)==GRIM and int(q.get('target_id',0) or 0)==MORGREM and int(q.get('target_area',0) or 0)==4:
   a=[i]
   return a if _legal(obs,a) and a!=baseline else None
 return None


def _exact_shadow_bullet_finish(obs,baseline):
 """Complete a visibly exact final-prize Shadow Bullet line.

 This guard is matchup-independent. It acts only when the current Active
 Grimmsnarl can attack and the public board proves that the 180/30 split
 supplies every remaining prize. All other states retain the stable action.
 """
 global _PENDING_SHADOW_SERIAL
 s=obs.get('select') or {};opts=s.get('option') or []
 ctx=int(s.get('context',-1) if s.get('context') is not None else -1)
 effect=_effect(obs)
 # Freeze the exact 30-damage target selected at the main-action decision.
 if _PENDING_SHADOW_SERIAL is not None and ctx==15 and effect==GRIM:
  for i,o in enumerate(opts):
   c=_card(obs,o)
   if int((c or {}).get('serial',-1) or -1)==_PENDING_SHADOW_SERIAL:
    a=[i];_PENDING_SHADOW_SERIAL=None
    if isinstance(baseline,list) and baseline==a:return None
    return a if _legal(obs,a) else None
  _PENDING_SHADOW_SERIAL=None
 if ctx!=0:
  return None
 _PENDING_SHADOW_SERIAL=None
 _,me,opp=_players(obs);ma=_active(me);oa=_active(opp)
 if _cid(ma)!=GRIM or _energy(ma)<2:return None
 remaining=len((me or {}).get('prize') or [])
 if remaining!=1:return None
 # The known attack-blocking Active cards are deliberately excluded.
 if not isinstance(oa,dict) or _cid(oa) in (117,345):return None
 active_prize=0
 if 0<_hp(oa)<=180:
  active_prize=3 if _cid(oa) in {652,662,678,687,695,723,737,747,754,756,766,772,781,790,828,849,861,868,886,896,904,919,928,932,939,1006,1031,1040,1056,1064} else (2 if int((oa or {}).get('maxHp',0) or 0)>=180 else 1)
 targets=[]
 for c in (opp or {}).get('bench') or []:
  if not isinstance(c,dict) or _cid(c) in hc.TERA_BENCH or not (0<_hp(c)<=30):continue
  prize=3 if _cid(c) in {652,662,678,687,695,723,737,747,754,756,766,772,781,790,828,849,861,868,886,896,904,919,928,932,939,1006,1031,1040,1056,1064} else (2 if int(c.get('maxHp',0) or 0)>=180 else 1)
  serial=int(c.get('serial',-1) or -1)
  if serial>=0:targets.append((prize,-_hp(c),-serial,serial))
 best=max(targets) if targets else None
 bench_prize=best[0] if best else 0
 if active_prize+bench_prize<remaining:return None
 attack_i=None
 for i,o in enumerate(opts):
  q=pf.semantic(obs,o)
  if int(q.get('type',-1))==13:
   attack_i=i;break
 if attack_i is None:return None
 if best and active_prize<remaining:
  _PENDING_SHADOW_SERIAL=best[-1]
 elif best and active_prize+bench_prize>=remaining:
  # Also freeze the bench target for exact two-prize split finishes.
  _PENDING_SHADOW_SERIAL=best[-1]
 a=[attack_i]
 if isinstance(baseline,list) and baseline==a:return None
 return a if _legal(obs,a) else None

def choose(obs,baseline):
 mem=hm.update(obs)
 for fn,args in ((_final_prize_damage_map,(obs,baseline)),(_exact_shadow_bullet_finish,(obs,baseline)),(_pokegear_take_only_supporter,(obs,baseline)),(_shadow_bullet_double_ko,(obs,baseline)),(_pokegear_whiff,(obs,mem)),(_boss_finish,(obs,mem)),(_pinsir_reserve_energy,(obs,mem)),(_wall_damage_source,(obs,baseline,mem)),(_wall_attach,(obs,baseline,mem))):
  try:a=fn(*args)
  except Exception:a=None
  if a is not None and _legal(obs,a):return a
 return baseline
