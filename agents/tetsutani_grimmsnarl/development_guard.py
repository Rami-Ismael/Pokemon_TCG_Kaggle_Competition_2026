from __future__ import annotations
"""Cross-family guarded development refinements."""
from collections import Counter
import human_controller as hc
DARK=7;FROSLASS=104;MUNKIDORI=112;IMPIDIMP=646;MORGREM=647;GRIM=648;SNORUNT=860
CANDY=1079;POFFIN=1086;STRETCHER=1097;POKEPAD=1152
USE_STRETCHER=False;USE_POKEPAD=False
def reset(): return None
def _ctx(obs):
 s=obs.get('select') or {};v=s.get('context');return int(v if v is not None else -1)
def _effect(obs):
 s=obs.get('select') or {};return hc._cid(s.get('effect')) or hc._cid(s.get('contextCard'))
def _walk(player,zones=('active','bench','hand')):
 for z in zones:
  for c in (player or {}).get(z) or []:
   if isinstance(c,dict):yield c

def _counts(player):return Counter(hc._cid(c) for c in _walk(player))
def _opts(obs):return (obs.get('select') or {}).get('option') or []
def _card(obs,i):
 o=_opts(obs);return hc._card_at(obs,o[i]) if 0<=i<len(o) else None
def _indices(obs,cid):return [i for i,o in enumerate(_opts(obs)) if hc._cid(hc._card_at(obs,o))==cid]
def _legal(obs,a):return hc._legal(obs,a)
def _poffin(obs,baseline):
 if _ctx(obs)!=5 or _effect(obs)!=POFFIN or not isinstance(baseline,list) or len(baseline)!=2:return None
 chosen=[hc._cid(_card(obs,i)) for i in baseline]
 if chosen.count(IMPIDIMP)!=2 and chosen.count(SNORUNT)!=2:return None
 imp=_indices(obs,IMPIDIMP);sno=_indices(obs,SNORUNT)
 if not imp or not sno:return None
 _,me,_=hc._players(obs);cnt=_counts(me);hand=Counter(hc._hand_ids(me));inplay=hc._inplay(me)
 grim_presence=cnt[IMPIDIMP]+cnt[MORGREM]+cnt[GRIM]
 snow_presence=cnt[SNORUNT]+cnt[FROSLASS]
 # Seed both engines only from an empty board, or after the main attacker line
 # is visibly secured. A partial Impidimp->Morgrem line without Grimmsnarl is
 # not diversified: redundancy is more valuable there.
 attacker_secured=(
  any(hc._cid(c)==GRIM for c in inplay) or
  (any(hc._cid(c)==MORGREM for c in inplay) and hand[GRIM]>0) or
  (any(hc._cid(c)==IMPIDIMP and not c.get('appearThisTurn',False) for c in inplay) and hand[GRIM]>0 and (hand[CANDY]>0 or hand[MORGREM]>0))
 )
 if grim_presence==0 and snow_presence==0:
  out=[imp[0],sno[0]]
 elif snow_presence==0 and attacker_secured and chosen.count(SNORUNT)==0:
  out=[sno[0],baseline[0]]
 elif grim_presence==0 and snow_presence>0 and chosen.count(IMPIDIMP)==0:
  out=[imp[0],baseline[0]]
 else:return None
 out=sorted(set(out));return out if len(out)==2 and out!=baseline and _legal(obs,out) else None
def _stretcher(obs,baseline):
 if not USE_STRETCHER or _ctx(obs)!=7 or _effect(obs)!=STRETCHER or not isinstance(baseline,list) or len(baseline)!=1:return None
 _,me,_=hc._players(obs);a=(me.get('active') or [None])[0];hand=Counter(hc._hand_ids(me))
 # Only redirect when the returned Energy immediately restores Shadow Bullet.
 if hc._cid(a)!=GRIM or hc._energy(a)>=2 or hand[DARK]>0:return None
 ids=_indices(obs,DARK)
 if not ids:return None
 out=[ids[0]];return out if out!=baseline and _legal(obs,out) else None
def _pokepad(obs,baseline):
 if not USE_POKEPAD or _ctx(obs)!=7 or _effect(obs)!=POKEPAD or not isinstance(baseline,list) or len(baseline)!=1:return None
 _,me,_=hc._players(obs);cnt=_counts(me);hand=Counter(hc._hand_ids(me));old=hc._cid(_card(obs,baseline[0]))
 # Exact missing evolution only: a live Snorunt and no Froslass anywhere. The
 # override is allowed only when the baseline would add a redundant utility copy.
 if cnt[SNORUNT]>0 and cnt[FROSLASS]==0 and hand[FROSLASS]==0 and old==MUNKIDORI and cnt[MUNKIDORI]>=2:
  ids=_indices(obs,FROSLASS)
  if ids:
   out=[ids[0]];return out if out!=baseline and _legal(obs,out) else None
 return None
def choose(obs,baseline):
 if not _legal(obs,baseline):return baseline
 for fn in (_poffin,_stretcher,_pokepad):
  try:a=fn(obs,baseline)
  except Exception:a=None
  if a is not None and _legal(obs,a):return a
 return baseline

# Punk Up is an "up to 5" search. Select only the Energy that can be placed on
# visible Marnie's Pokémon without exceeding the two-Energy attack plan. This
# preserves the remaining Darkness Energy for manual Munkidori activation and
# future attackers instead of forcing surplus copies onto a single Active.
def _punk_up_useful_count(obs, baseline):
 if _ctx(obs)!=22 or _effect(obs)!=GRIM or not isinstance(baseline,list):return None
 sel=obs.get('select') or {};opts=sel.get('option') or []
 if int(sel.get('minCount',0) or 0)!=0 or not opts:return None
 _,me,opp=hc._players(obs)
 # Preserve the established mirror policy: optional reserve Energy has a
 # different value when both boards expose the same evolution line.
 if any(hc._cid(c) in (IMPIDIMP,MORGREM,GRIM) for c in hc._inplay(opp)):return None
 targets=[c for c in hc._inplay(me) if hc._cid(c) in (IMPIDIMP,MORGREM,GRIM)]
 active=(me.get('active') or [None])[0]
 # Only trim the optional count when the current attacker is established and
 # a second visible line can use acceleration. Lone-attacker states retain the
 # established reserve-Energy behavior.
 if hc._cid(active)!=GRIM or len(targets)<2:return None
 need=sum(max(0,2-hc._energy(c)) for c in targets)
 need=min(int(sel.get('maxCount',0) or 0),len(opts),need)
 # Declining all Energy is legal only when every visible target is already ready.
 if need>=len(baseline):return None
 out=sorted(range(need))
 return out if out!=baseline and _legal(obs,out) else None

# Preserve the v10 guards; only reduce a visibly surplus optional acceleration
# count after the original policy has selected the Energy class.
def choose(obs,baseline):
 if not _legal(obs,baseline):return baseline
 for fn in (_poffin,_stretcher,_pokepad,_punk_up_useful_count):
  try:a=fn(obs,baseline)
  except Exception:a=None
  if a is not None and _legal(obs,a):return a
 return baseline
