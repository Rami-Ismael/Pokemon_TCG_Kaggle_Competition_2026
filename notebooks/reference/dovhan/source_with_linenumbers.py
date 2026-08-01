from __future__ import annotations
import re, warnings, unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='muted')
pd.set_option('display.max_colwidth', 60)

CARD_PATH = '/kaggle/input/competitions/pokemon-tcg-ai-battle-challenge-strategy/EN_Card_Data.csv'
STAGE_COL = 'Stage (Pokémon)/Type (Energy and Trainer)'   # authoritative classification column

df_raw = pd.read_csv(CARD_PATH)
print(f'Loaded {len(df_raw)} rows × {len(df_raw.columns)} columns')
print(df_raw.columns.tolist())
df_raw.head(3)
df = df_raw.copy()
STR_COLS = ['Card Name','Category','Type','Weakness','Resistance (Type)',
            STAGE_COL,'Move Name','Cost','Damage','Retreat','Rule']
for c in STR_COLS:
    if c in df.columns:
        df[c] = df[c].fillna('').astype(str).str.strip()
df['HP'] = pd.to_numeric(df['HP'], errors='coerce').fillna(0).astype(int)

def card_type(row) -> str:
    stage = str(row.get(STAGE_COL, '')).lower()
    cat   = str(row.get('Category', ''))
    if 'energy' in stage: return 'energy'
    if any(k in stage for k in ['supporter','item','stadium','technical machine','pokémon tool']): return 'trainer'
    if any(k in stage for k in ['basic','stage 1','stage 2','fossil','tera','ancient','future']): return 'pokemon'
    if 'okémon' in cat: return 'pokemon'
    return 'unknown'

df['card_type'] = df.apply(card_type, axis=1)

print('=== STAGE COLUMN VALUES ===')
print(df[STAGE_COL].value_counts().to_string())
print('\n=== CARD TYPE COUNTS (all 2022 rows) ===')
print(df['card_type'].value_counts())

pokemon_df = df[df['card_type'] == 'pokemon'].reset_index(drop=True)
trainer_df = df[df['card_type'] == 'trainer'].reset_index(drop=True)
energy_df  = df[df['card_type'] == 'energy'].reset_index(drop=True)
print(f'\nPokémon rows : {len(pokemon_df)}')
print(f'Trainer rows : {len(trainer_df)}')
print(f'Energy rows  : {len(energy_df)}')
display(pokemon_df[['Card Name','HP','Type','Weakness',STAGE_COL,'Move Name','Cost','Damage']].head(4))
display(energy_df [['Card Name','Type',STAGE_COL]].head(4))
df['Damage_num'] = df['Damage'].apply(
    lambda x: int(re.sub(r'[^\\d]','', str(x).split('+')[0].split('×')[0].split('x')[0]) or 0)
)
df_cards = (
    df.sort_values('Damage_num', ascending=False)
      .drop_duplicates(subset=['Card ID'], keep='first')
      .reset_index(drop=True)
)
pokemon_u = df_cards[df_cards['card_type'] == 'pokemon'].reset_index(drop=True)
trainer_u = df_cards[df_cards['card_type'] == 'trainer'].reset_index(drop=True)
energy_u  = df_cards[df_cards['card_type'] == 'energy'].reset_index(drop=True)

print(f'Unique cards total : {len(df_cards)}')
print(f'  Pokémon          : {len(pokemon_u)}')
print(f'  Trainer          : {len(trainer_u)}')
print(f'  Energy           : {len(energy_u)}')
display(pokemon_u[['Card Name','HP','Type','Weakness','Move Name','Cost','Damage']].head(8))
ENERGY_TYPES: List[str] = ['G','R','W','L','P','F','D','M','C']
TYPE_TO_IDX:  Dict[str,int] = {t:i for i,t in enumerate(ENERGY_TYPES)}
N_TYPES = len(ENERGY_TYPES)
TYPE_COLORS = {
    'G':'#78C850','R':'#F08030','W':'#6890F0','L':'#F8D030',
    'P':'#A040A0','F':'#C03028','D':'#705848','M':'#B8B8D0','C':'#A8A878'
}
_SYM = re.compile(r'\{([A-Z]+)\}|(●)')   # ● = colorless energy in this dataset

def parse_cost(s: str) -> torch.Tensor:
    """Parse energy cost string into a per-type count vector of shape [N_TYPES]."""
    vec = torch.zeros(N_TYPES, dtype=torch.long)
    if not isinstance(s,str) or s.lower() in ('','n/a','0','nan'): return vec
    for sym, dot in _SYM.findall(s):
        if dot == '●': vec[TYPE_TO_IDX['C']] += 1
        elif sym in TYPE_TO_IDX: vec[TYPE_TO_IDX[sym]] += 1
    return vec

def parse_damage(s: str) -> Tuple[int, float]:
    """Return (base_damage, scaling_factor). '30+' → (30,1.0); '20×' → (0,20.0)."""
    if not isinstance(s,str): return 0, 0.0
    s = s.strip()
    if s.lower() in ('','n/a','nan','0'): return 0, 0.0
    if s.endswith('+'):
        try: return int(s[:-1]), 1.0
        except: return 0, 1.0
    if s.endswith('×') or s.endswith('x'):
        try: return 0, float(s[:-1])
        except: return 0, 0.0
    try: return int(s), 0.0
    except: return 0, 0.0

def can_pay(available: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    """Vectorised energy payment: available [B, N_TYPES], cost [N_TYPES] → BoolTensor [B]."""
    typed = cost.clone(); typed[TYPE_TO_IDX['C']] = 0
    surplus = available - typed.unsqueeze(0)
    can_t = (surplus >= 0).all(-1)
    can_c = surplus.clamp(0).sum(-1) >= cost[TYPE_TO_IDX['C']]
    return can_t & can_c

def compute_damage(base, scaling, counter, bonus, weakness, resistance, res=30):
    """Damage = (base + scaling*counter + bonus) ×2 if weak, −30 if resist. Floor 0."""
    raw = base.float() + scaling * counter.float() + bonus.float()
    raw = torch.where(weakness, raw * 2, raw)
    raw = torch.where(resistance, raw - res, raw)
    return raw.clamp(0).long()

# Smoke tests
assert parse_cost('{R}●●')[TYPE_TO_IDX['R']].item() == 1
assert parse_cost('{R}●●')[TYPE_TO_IDX['C']].item() == 2
assert parse_cost('●●●')[TYPE_TO_IDX['C']].item()   == 3
print('All parsers OK  ● = colorless {C}')
print(f'  parse_cost({{R}}●●) = {parse_cost("{R}●●").tolist()}')
print(f'  parse_damage("30+") = {parse_damage("30+")}')
print(f'  parse_damage("20×") = {parse_damage("20×")}')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Pokémon Card Statistics', fontsize=16, fontweight='bold')

hp = pokemon_u['HP'][pokemon_u['HP'] > 0]
axes[0].hist(hp, bins=30, color='#6890F0', edgecolor='white')
axes[0].axvline(hp.mean(), color='red', linestyle='--', label=f'Mean: {hp.mean():.0f}')
axes[0].set_title('HP Distribution'); axes[0].set_xlabel('HP'); axes[0].legend()

dmg_vals = pokemon_u['Damage'].apply(lambda x: parse_damage(str(x))[0])
dmg_vals = dmg_vals[dmg_vals > 0]
axes[1].hist(dmg_vals, bins=25, color='#F08030', edgecolor='white')
axes[1].axvline(dmg_vals.mean(), color='navy', linestyle='--', label=f'Mean: {dmg_vals.mean():.0f}')
axes[1].set_title('Base Damage Distribution'); axes[1].set_xlabel('Damage'); axes[1].legend()

type_counts = pokemon_u['Type'].str.extract(r'\{([A-Z])\}')[0].value_counts()
colors = [TYPE_COLORS.get(t,'#888') for t in type_counts.index]
axes[2].bar(type_counts.index, type_counts.values, color=colors, edgecolor='white')
axes[2].set_title('Pokémon Type Distribution'); axes[2].set_xlabel('Type')

plt.tight_layout(); plt.show()

print('\n=== TOP 10 POKEMON BY HP ===')
display(pokemon_u[['Card Name','HP','Type','Expansion']].sort_values('HP',ascending=False).head(10).reset_index(drop=True))

print('\n=== TOP 10 HARDEST ATTACKS ===')
pokemon_u['dmg_num'] = pokemon_u['Damage'].apply(lambda x: parse_damage(str(x))[0])
display(pokemon_u[['Card Name','dmg_num','Move Name','Cost','Type']].sort_values('dmg_num',ascending=False).head(10).reset_index(drop=True))
types_8 = ENERGY_TYPES[:-1]   # exclude colorless
weakness_map = {t:{t2:0 for t2 in types_8} for t in types_8}
for _, row in pokemon_u.iterrows():
    pt = re.search(r'\{([A-Z])\}', str(row['Type']))
    wk = re.search(r'\{([A-Z])\}', str(row['Weakness']))
    if pt and wk:
        p, w = pt.group(1), wk.group(1)
        if p in weakness_map and w in weakness_map:
            weakness_map[w][p] += 1

matrix = pd.DataFrame(weakness_map).loc[types_8, types_8].fillna(0)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white', ax=ax)
ax.set_title('Type Matchup Heatmap\n(cell = Pokémon of column-type weak to row-type)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Defending Type'); ax.set_ylabel('Attacking Type (×2)')
plt.tight_layout(); plt.show()

row_sums = matrix.sum(axis=1).sort_values(ascending=False)
print('\n=== MOST EFFECTIVE ATTACKING TYPES ===')
for t, v in row_sums.items():
    bar = '█' * int(v // 15)
    print(f'  {t} hits {int(v):3d} Pokémon for ×2  {bar}')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Cost Analysis', fontsize=15, fontweight='bold')

pokemon_u['retreat_num'] = pd.to_numeric(pokemon_u['Retreat'], errors='coerce').fillna(0)
rc = pokemon_u['retreat_num'].value_counts().sort_index()
axes[0].bar(rc.index.astype(str), rc.values, color='#A040A0', edgecolor='white')
axes[0].set_title('Retreat Cost Distribution')
axes[0].set_xlabel('Energy to Retreat'); axes[0].set_ylabel('Count')

cost_by_type: Dict[str,List[int]] = {t:[] for t in types_8}
for _, row in pokemon_u.iterrows():
    pt = re.search(r'\{([A-Z])\}', str(row['Type']))
    if pt and pt.group(1) in cost_by_type:
        c = parse_cost(str(row['Cost'])).sum().item()
        if c > 0: cost_by_type[pt.group(1)].append(c)

avg_cost = {t: round(np.mean(v),2) if v else 0 for t,v in cost_by_type.items()}
colors = [TYPE_COLORS.get(t,'#888') for t in avg_cost]
axes[1].bar(list(avg_cost.keys()), list(avg_cost.values()), color=colors, edgecolor='white')
axes[1].axhline(np.mean(list(avg_cost.values())), color='red', linestyle='--', label='Overall mean')
axes[1].set_title('Avg Attack Cost by Type'); axes[1].set_xlabel('Type')
axes[1].set_ylabel('Avg Energy Cost'); axes[1].legend()

plt.tight_layout(); plt.show()
print('\n=== AVERAGE ATTACK COST PER TYPE ===')
for t,v in sorted(avg_cost.items(), key=lambda x:-x[1]):
    print(f'  {t}: {v:.2f} energy per attack')
# ── Power Creep ─────────────────────────────────────────────────────────────
poke_tl = pokemon_u.copy()
poke_tl['dmg_n'] = poke_tl['Damage'].apply(lambda x: parse_damage(str(x))[0])
poke_tl['hp_n']  = pd.to_numeric(poke_tl['HP'], errors='coerce')

exp_stats = (
    poke_tl[poke_tl['dmg_n'] > 0]
    .groupby('Expansion')
    .agg(avg_dmg=('dmg_n','mean'), avg_hp=('hp_n','mean'), cards=('dmg_n','count'))
    .sort_values('avg_dmg', ascending=False)
)
print('=== POWER CREEP: TOP 10 EXPANSIONS BY AVG DAMAGE ===')
display(exp_stats.head(10).round(1))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Power Creep Timeline', fontsize=14, fontweight='bold')
top15 = exp_stats.head(15)
axes[0].barh(top15.index[::-1], top15['avg_dmg'][::-1], color='#F08030', edgecolor='white')
axes[0].axvline(exp_stats['avg_dmg'].mean(), color='navy', linestyle='--',
                label=f"Mean: {exp_stats['avg_dmg'].mean():.0f}")
axes[0].set_title('Avg Damage by Expansion (top 15)'); axes[0].set_xlabel('Avg Damage'); axes[0].legend()
axes[1].barh(top15.index[::-1], top15['avg_hp'][::-1], color='#6890F0', edgecolor='white')
axes[1].axvline(exp_stats['avg_hp'].mean(), color='red', linestyle='--',
                label=f"Mean: {exp_stats['avg_hp'].mean():.0f}")
axes[1].set_title('Avg HP by Expansion (top 15)'); axes[1].set_xlabel('Avg HP'); axes[1].legend()
plt.tight_layout(); plt.show()

# ── Effect Text Mining ───────────────────────────────────────────────────────
MECHANICS: Dict[str, List[str]] = {
    'draw':    ['draw a card','draw 2','draw 3','draw until'],
    'search':  ['search your deck'],
    'discard': ['discard'],
    'heal':    ['heal','remove.*damage counter'],
    'ko':      ['knock out','k\\.o\\.'],
    'coin':    ['flip a coin','heads','tails'],
    'energy':  ['attach.*energy','energy from'],
    'bench':   ['benched','bench'],
}
effects = df_cards[df_cards['card_type']=='pokemon']['Effect Explanation'].fillna('').astype(str).str.lower()
mech_counts: Dict[str,int] = {}
for mech, patterns in MECHANICS.items():
    mech_counts[mech] = int(effects.str.contains('|'.join(patterns), regex=True).sum())

mech_s = pd.Series(mech_counts).sort_values(ascending=False)
print('\n=== ATTACK MECHANIC FREQUENCY ===')
for m, v in mech_s.items():
    bar = '█' * (v // 10)
    print(f'  {m:<10} {v:4d} cards ({v/len(pokemon_u)*100:4.1f}%)  {bar}')

fig, ax = plt.subplots(figsize=(11, 4))
colors_m = ['#F08030','#6890F0','#A040A0','#78C850','#F8D030','#C03028','#705848','#B8B8D0']
bars = ax.bar(mech_s.index, mech_s.values, color=colors_m[:len(mech_s)], edgecolor='white')
ax.set_title('TCG Mechanic Frequency in Attack Effects', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Pokémon cards')
for b, v in zip(bars, mech_s.values):
    ax.text(b.get_x()+b.get_width()/2, v+2, str(v), ha='center', fontsize=9)
plt.tight_layout(); plt.show()
import networkx as nx

edge_data: Dict[tuple, int] = {}
for _, row in pokemon_u.iterrows():
    pt = re.search(r'\{([A-Z])\}', str(row['Type']))
    wk = re.search(r'\{([A-Z])\}', str(row['Weakness']))
    if pt and wk and pt.group(1) in TYPE_TO_IDX and wk.group(1) in TYPE_TO_IDX:
        key = (wk.group(1), pt.group(1))   # attacker exploits defender's weakness
        edge_data[key] = edge_data.get(key, 0) + 1

G = nx.DiGraph()
for (src, dst), w in edge_data.items():
    G.add_edge(src, dst, weight=w)

node_sizes = {n: pokemon_u['Type'].str.contains(f'{{{n}}}', regex=False).sum() for n in G.nodes()}
pos = nx.spring_layout(G, seed=42, k=2.5)

fig, ax = plt.subplots(figsize=(13, 10))
ax.set_title('Type Synergy Network\n(arrow = attacks for ×2 · thickness = # Pokémon)',
             fontsize=13, fontweight='bold')
ax.axis('off')
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_w = max(edge_weights)
nx.draw_networkx_nodes(G, pos, ax=ax,
    node_color=[TYPE_COLORS.get(n,'#888') for n in G.nodes()],
    node_size=[node_sizes.get(n,50)*4+300 for n in G.nodes()], alpha=0.92)
nx.draw_networkx_labels(G, pos, ax=ax,
    labels={n: f"{n}\n({node_sizes.get(n,0)})" for n in G.nodes()},
    font_size=9, font_weight='bold')
nx.draw_networkx_edges(G, pos, ax=ax,
    width=[w/max_w*8 for w in edge_weights],
    edge_color=[TYPE_COLORS.get(u,'#888') for u,v in G.edges()],
    alpha=0.7, arrows=True, arrowstyle='-|>', arrowsize=20,
    connectionstyle='arc3,rad=0.15')
edge_labels = {(u,v): G[u][v]['weight'] for u,v in G.edges() if G[u][v]['weight'] > 30}
nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)
plt.tight_layout(); plt.show()

top_edges = sorted(edge_data.items(), key=lambda x: -x[1])[:10]
print('\n=== TOP 10 TYPE MATCHUPS (most Pokémon to exploit) ===')
print(f'{"Attacker":<10} {"Defender":<10} {"# Pokémon"}')
print('-'*34)
for (atk, dfn), cnt in top_edges:
    print(f'  {atk} → {dfn:<10} {cnt:>5}')
@dataclass
class CardEncoder:
    df: pd.DataFrame
    max_hp: int = 380

    def __post_init__(self) -> None:
        self._build()

    def _encode_row(self, row: pd.Series) -> List[float]:
        f: List[float] = []
        f.append(int(row.get('HP',0) or 0) / self.max_hp)
        pt = re.search(r'\{([A-Z])\}', str(row.get('Type','')))
        tv = [0.0]*N_TYPES
        if pt and pt.group(1) in TYPE_TO_IDX: tv[TYPE_TO_IDX[pt.group(1)]] = 1.0
        f.extend(tv)
        wk = re.search(r'\{([A-Z])\}', str(row.get('Weakness','')))
        wv = [0.0]*N_TYPES
        if wk and wk.group(1) in TYPE_TO_IDX: wv[TYPE_TO_IDX[wk.group(1)]] = 1.0
        f.extend(wv)
        ret = str(row.get('Retreat','0')).strip()
        f.append((int(ret) if ret.isdigit() else 0) / 5.0)
        f.extend((parse_cost(str(row.get('Cost',''))).float() / 5.0).tolist())
        base, scale = parse_damage(str(row.get('Damage','')))
        f.append(min(base,300)/300.0)
        f.append(min(scale,10.0)/10.0)
        ct = str(row.get('card_type',''))
        f += [float(ct=='pokemon'), float(ct=='trainer'), float(ct=='energy')]
        stage = str(row.get(STAGE_COL,'')).lower()
        f += [float('basic' in stage and 'energy' not in stage),
              float('stage 1' in stage), float('stage 2' in stage)]
        return f

    def _build(self) -> None:
        rows = [self._encode_row(r) for _,r in self.df.iterrows()]
        self.card_matrix = torch.tensor(rows, dtype=torch.float32).clamp(0.0, 1.0)
        self.card_ids: List[int]      = self.df['Card ID'].tolist()
        self.id_to_idx: Dict[int,int] = {cid:i for i,cid in enumerate(self.card_ids)}

    @property
    def feature_dim(self) -> int: return self.card_matrix.shape[1]

    def lookup(self, ids: List[int]) -> torch.Tensor:
        return self.card_matrix[[self.id_to_idx[i] for i in ids]]

encoder = CardEncoder(df_cards)
print(f'Card matrix : {encoder.card_matrix.shape}')
print(f'Feature dim : {encoder.feature_dim}')
print(f'Value range : [{encoder.card_matrix.min():.3f}, {encoder.card_matrix.max():.3f}]')
rows_show = []
for i in range(min(6, len(encoder.card_ids))):
    cid   = encoder.card_ids[i]
    name  = df_cards[df_cards['Card ID']==cid]['Card Name'].values[0]
    ctype = df_cards[df_cards['Card ID']==cid]['card_type'].values[0]
    vec   = encoder.card_matrix[i]
    rows_show.append({'Card':name,'type':ctype,'HP(n)':round(vec[0].item(),3),
                      'Dmg(n)':round(vec[29].item(),3),'Cost(n)':round(vec[20:29].sum().item(),3)})
display(pd.DataFrame(rows_show))
STATE_DIM = encoder.feature_dim * 6 + 20   # 37*6 + 20 = 242
N_ACTIONS = 64

class PokemonTCGNet(nn.Module):
    def __init__(self, state_dim:int, n_actions:int,
                 hidden:Tuple[int,...]=(512,256,128), dropout:float=0.1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = state_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk  = nn.Sequential(*layers)
        self.policy = nn.Sequential(nn.Linear(d,d//2), nn.ReLU(), nn.Linear(d//2,n_actions))
        self.value  = nn.Sequential(nn.Linear(d,d//2), nn.ReLU(), nn.Linear(d//2,1))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)

    def forward(self, s:torch.Tensor,
                mask:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor,torch.Tensor]:
        x = self.trunk(s)
        logits = self.policy(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))  # block illegal actions
        return logits, self.value(x)

    def act(self, s, mask=None, deterministic=False):
        logits, v = self.forward(s, mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a, dist.log_prob(a), v.squeeze(-1)

net   = PokemonTCGNet(STATE_DIM, N_ACTIONS)
total = sum(p.numel() for p in net.parameters())
print(net)
print(f'\nTotal parameters : {total:,}')
print(f'State dim        : {STATE_DIM}')
print(f'Action space     : {N_ACTIONS}')
dummy = torch.randn(4, STATE_DIM)
logits, val = net(dummy)
print(f'Output shapes    : logits={logits.shape}  value={val.shape}')
@dataclass
class PPOConfig:
    lr:           float = 3e-4
    gamma:        float = 0.99
    gae_lambda:   float = 0.95
    clip_eps:     float = 0.2
    value_coef:   float = 0.5
    entropy_coef: float = 0.01
    n_epochs:     int   = 4
    batch_size:   int   = 64
    device:       str   = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_gae(rewards, values, dones, cfg):
    """Compute Generalised Advantage Estimation for a trajectory."""
    T = len(rewards); adv = torch.zeros(T); g = 0.0
    for t in reversed(range(T)):
        nv = values[t+1] if t+1 < T else 0.0
        d  = rewards[t] + cfg.gamma * nv * (1 - dones[t]) - values[t]
        g  = d + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * g
        adv[t] = g
    return adv, adv + values[:T]

def ppo_update(net, optimizer, states, actions, old_lp,
               returns, advantages, masks, cfg):
    """One PPO update over a stored trajectory buffer."""
    T = states.shape[0]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    stats = {'policy':0.0,'value':0.0,'entropy':0.0,'steps':0}
    for _ in range(cfg.n_epochs):
        for idx in torch.randperm(T).split(cfg.batch_size):
            s=states[idx]; a=actions[idx]; olp=old_lp[idx]
            ret=returns[idx]; adv=advantages[idx]
            m = masks[idx] if masks is not None else None
            logits,v = net(s,m)
            dist = torch.distributions.Categorical(logits=logits)
            lp=dist.log_prob(a); ent=dist.entropy().mean()
            ratio = (lp - olp).exp()
            pl = -torch.min(ratio*adv,
                            ratio.clamp(1-cfg.clip_eps, 1+cfg.clip_eps)*adv).mean()
            vl = F.mse_loss(v.squeeze(-1), ret)
            loss = pl + cfg.value_coef * vl - cfg.entropy_coef * ent
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            stats['policy']+=pl.item(); stats['value']+=vl.item()
            stats['entropy']+=ent.item(); stats['steps']+=1
    return stats

cfg       = PPOConfig()
optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
print(f'PPO ready | device={cfg.device} | lr={cfg.lr} | clip_eps={cfg.clip_eps} | γ={cfg.gamma} | λ={cfg.gae_lambda}')
def cost_compatible(cost_str: str, main_type: str, max_off: int = 1) -> bool:
    """True if attack uses at most max_off non-main, non-colorless energy symbols."""
    vec = parse_cost(cost_str)
    off = sum(vec[i].item() for i,t in enumerate(ENERGY_TYPES[:-1]) if t != main_type and vec[i] > 0)
    return off <= max_off

# ── Real TCG rules constants ──────────────────────────────────────────────────
DECK_SIZE           = 60
HAND_SIZE           = 7
PRIZE_CARDS         = 6
MIN_BASICS          = 12
MIN_BASICS_FRACTION = MIN_BASICS / DECK_SIZE   # 0.20

def simulate_mulligan_rate(n_basics: int, deck_size: int = DECK_SIZE,
                            hand_size: int = HAND_SIZE, trials: int = 50_000) -> float:
    """Monte Carlo P(opening hand has zero Basic Pokémon) — without-replacement sampling."""
    rng  = np.random.default_rng(42)
    deck = np.zeros(deck_size, dtype=np.int8)
    deck[:n_basics] = 1
    idx   = rng.random((trials, deck_size)).argsort(axis=1)
    hands = deck[idx[:, :hand_size]]
    return float((hands.sum(axis=1) == 0).mean())

def build_deck_smart(pokemon_type: str = 'R', size: int = 60) -> pd.DataFrame:
    """
    Build a legal 60-card deck for the given energy type.

    Composition:
      12 Basic Pokémon  (top 3 × 4 copies)  — 20% of deck → <10% mulligan rate
       6 Stage 1        (top 2 × 3 copies)
       6 Pre-evolutions (basic for each stage 1 × 3 copies)
      15 Basic Energy
      12 Item Trainers
       8 Supporter Trainers
       1 Stadium
    Total = 60 cards
    """
    tp = f'{{{pokemon_type}}}'

    basics_pool = pokemon_u[
        pokemon_u['Type'].str.contains(tp, na=False) &
        pokemon_u[STAGE_COL].str.lower().str.contains('basic pokémon', na=False)
    ].copy()
    basics_pool['compat'] = basics_pool['Cost'].apply(lambda c: cost_compatible(str(c), pokemon_type))
    basics_pool['dmg_n']  = basics_pool['Damage'].apply(lambda x: parse_damage(str(x))[0])
    top3 = basics_pool[basics_pool['compat']].sort_values('dmg_n', ascending=False).head(3)
    basics = pd.concat([top3]*4, ignore_index=True)

    stage1_pool = pokemon_u[
        pokemon_u['Type'].str.contains(tp, na=False) &
        pokemon_u[STAGE_COL].str.lower().str.contains('stage 1 pokémon', na=False)
    ].copy()
    stage1_pool['compat'] = stage1_pool['Cost'].apply(lambda c: cost_compatible(str(c), pokemon_type))
    stage1_pool['dmg_n']  = stage1_pool['Damage'].apply(lambda x: parse_damage(str(x))[0])
    top2s1 = stage1_pool[stage1_pool['compat']].sort_values('dmg_n', ascending=False).head(2)
    stage1 = pd.concat([top2s1]*3, ignore_index=True)

    prevos = []
    for _, s1 in top2s1.iterrows():
        prev = df_raw[df_raw['Card Name'] == s1['Card Name']]['Previous stage'].dropna()
        if len(prev) > 0:
            pn = str(prev.values[0]).strip()
            m  = pokemon_u[pokemon_u['Card Name'] == pn]
            if len(m) > 0: prevos.append(pd.concat([m.iloc[[0]]]*3, ignore_index=True))
    prevos_df = pd.concat(prevos, ignore_index=True) if prevos else pd.DataFrame()

    energy_match = energy_u[energy_u['Type'].str.contains(tp, na=False)]
    if len(energy_match) == 0: energy_match = energy_u.head(1)
    energy_rows = pd.concat([energy_match]*20, ignore_index=True).head(15)

    items = trainer_u[trainer_u[STAGE_COL].str.lower().str.contains('item', na=False)]
    supps = trainer_u[trainer_u[STAGE_COL].str.lower().str.contains('supporter', na=False)]
    stads = trainer_u[trainer_u[STAGE_COL].str.lower().str.contains('stadium', na=False)]
    if len(items) < 6: items = trainer_u.head(12)
    if len(supps) < 4: supps = trainer_u.tail(8)
    item_cards = pd.concat([items.head(6)]*2, ignore_index=True).head(12)
    supp_cards = pd.concat([supps.head(4)]*2, ignore_index=True).head(8)
    stad_cards = stads.head(1) if len(stads) > 0 else pd.DataFrame()

    parts = [basics, stage1]
    if len(prevos_df) > 0: parts.append(prevos_df)
    parts += [energy_rows, item_cards, supp_cards]
    if len(stad_cards) > 0: parts.append(stad_cards)

    deck = pd.concat(parts, ignore_index=True).head(size)
    cols = ['Card Name','card_type',STAGE_COL,'Type','HP','Damage','Cost']
    return deck[[c for c in cols if c in deck.columns]]

def print_deck_list(deck: pd.DataFrame, title: str = 'DECK LIST') -> None:
    """Print a formatted deck list grouped by section with copy counts and stats."""
    sections = {
        'Pokémon': deck[deck['card_type'] == 'pokemon'],
        'Energy':  deck[deck['card_type'] == 'energy'],
        'Trainer': deck[deck['card_type'] == 'trainer'],
    }
    total = len(deck)
    print(f'\n{"═"*56}')
    print(f'  {title}  ({total} cards)')
    print(f'{"═"*56}')
    for section, rows in sections.items():
        if len(rows) == 0: continue
        unique = rows['Card Name'].value_counts()
        print(f'\n  ── {section} ({len(rows)}) ──────────────────────────────────')
        for name, cnt in unique.items():
            row = rows[rows['Card Name'] == name].iloc[0]
            if section == 'Pokémon':
                info = f"HP:{row.get('HP','')}  DMG:{row.get('Damage','')}"
            elif section == 'Energy':
                info = str(row.get('Type', ''))
            else:
                info = str(row.get(STAGE_COL, ''))[:28]
            print(f'    {cnt}x  {name:<30} {info}')
    basics_n = int(((deck['card_type']=='pokemon') &
                     deck[STAGE_COL].str.lower().str.contains('basic', na=False) &
                    ~deck[STAGE_COL].str.lower().str.contains('energy', na=False)).sum())
    rate = simulate_mulligan_rate(basics_n, deck_size=total, hand_size=HAND_SIZE) * 100
    pn = len(deck[deck['card_type']=='pokemon'])
    en = len(deck[deck['card_type']=='energy'])
    tn = len(deck[deck['card_type']=='trainer'])
    print(f'\n{"═"*56}')
    print(f'  Pokémon: {pn}  |  Energy: {en}  |  Trainer: {tn}  |  Total: {total}')
    print(f'  Basic Pokémon: {basics_n} ({basics_n/total*100:.0f}% of deck)')
    print(f'  Mulligan rate: {rate:.1f}%  {"✓ SAFE (<10%)" if rate < 10 else "⚠ RISKY"}')
    print(f'{"═"*56}\n')

# ── Summary table ─────────────────────────────────────────────────────────────
print('=== 60-CARD DECK COMPOSITION BY TYPE ===')
print(f'{"Type":<6} {"Cards":<8} {"Basic Pkmn":<14} {"Stage1":<10} {"Energy":<10} {"Trainer":<10} {"Mulligan%"}')
print('─'*72)
for t in ['R','W','G','L','P','F','D','M']:
    d  = build_deck_smart(t, size=60)
    nb = int(((d['card_type']=='pokemon') &
               d[STAGE_COL].str.lower().str.contains('basic', na=False) &
              ~d[STAGE_COL].str.lower().str.contains('energy', na=False)).sum())
    ns = int(((d['card_type']=='pokemon') & d[STAGE_COL].str.lower().str.contains('stage', na=False)).sum())
    ne = int((d['card_type']=='energy').sum())
    nt = int((d['card_type']=='trainer').sum())
    rate = simulate_mulligan_rate(nb, deck_size=len(d), hand_size=7) * 100 if nb > 0 else 100.0
    flag = '✓ SAFE' if rate < 10 else '⚠ RISKY'
    print(f'  {t:<4} {len(d):<8} {nb:<14} {ns:<10} {ne:<10} {nt:<10} {rate:.1f}% {flag}')

# ── Show deck lists for all 8 types ──────────────────────────────────────────
for t in ['R','W','G','L','P','F','D','M']:
    print_deck_list(build_deck_smart(t, size=60), title=f'DECK — TYPE {t}')
# Constants DECK_SIZE, HAND_SIZE, MIN_BASICS, simulate_mulligan_rate — defined in Cell 13

print('=== MULLIGAN BRICK RISK — 60-CARD DECK, 7-CARD HAND (REAL RULES) ===')
print(f'{"Basics in deck":<18} {"% of deck":<12} {"Brick rate":<14} Verdict')
print('─'*58)
for n in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
    rate    = simulate_mulligan_rate(n, deck_size=DECK_SIZE, hand_size=HAND_SIZE)
    pct     = n / DECK_SIZE * 100
    flag    = '← COMFORT FLOOR' if n == MIN_BASICS else ''
    verdict = ('UNPLAYABLE' if rate > 0.4 else
               'HIGH RISK'  if rate > 0.2 else
               'MARGINAL'   if rate > 0.1 else
               'SAFE ✓')
    print(f'  {n:<16} {pct:5.1f}%       {rate*100:5.1f}%         {verdict:<12} {flag}')

print(f'\n  Community rule: ≥{MIN_BASICS} Basics ({MIN_BASICS_FRACTION*100:.0f}% of deck) → <10% brick rate')

# ── Plot ──────────────────────────────────────────────────────────────────────
basics_range = list(range(1, 25))
rates_60 = [simulate_mulligan_rate(n, deck_size=60, hand_size=7)*100 for n in basics_range]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Mulligan Brick Risk — Official 60-Card Format (7-card opening hand)',
             fontsize=13, fontweight='bold')

axes[0].plot(basics_range, rates_60, color='#F08030', linewidth=3, marker='o', markersize=6)
axes[0].axhline(10, color='green', linestyle='--', linewidth=1.5, label='10% comfort threshold')
axes[0].axhline(5,  color='navy',  linestyle=':',  linewidth=1.5, label='5%  safe threshold')
axes[0].fill_between(basics_range, rates_60, 10,
    where=[r > 10 for r in rates_60], alpha=0.12, color='red', label='Danger zone')
axes[0].annotate(f'Community floor\n{MIN_BASICS} Basics\n{rates_60[MIN_BASICS-1]:.1f}% brick',
    xy=(MIN_BASICS, rates_60[MIN_BASICS-1]),
    xytext=(MIN_BASICS+2, rates_60[MIN_BASICS-1]+12),
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', fc='#f8d030', alpha=0.9))
axes[0].set_title('Brick Rate vs Basic Count (60-card deck)')
axes[0].set_xlabel('Basic Pokémon in Deck')
axes[0].set_ylabel('Brick Rate (%)')
axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlim(1, 24)

# Right: expected opponent bonus cards (geometric: E[mulligans] = p/(1-p))
expected_bonus = [r/100 / max(1 - r/100, 1e-6) for r in rates_60]
axes[1].bar(basics_range, expected_bonus, color='#C03028', edgecolor='white', alpha=0.8)
axes[1].axvline(MIN_BASICS, color='green', linestyle='--', linewidth=2,
                label=f'Min safe ({MIN_BASICS} basics)')
axes[1].set_title('Expected Bonus Cards Gifted to Opponent per Game')
axes[1].set_xlabel('Basic Pokémon in Deck')
axes[1].set_ylabel('Expected opponent bonus draws')
axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# ── Format comparison ─────────────────────────────────────────────────────────
print('\n=== FORMAT COMPARISON: 60-CARD vs 20-CARD PROPORTIONAL EQUIVALENT ===')
print(f'{"60-card deck":<20} {"Brick rate":<14} {"20-card equiv":<18} {"20-card brick rate"}')
print('─'*68)
for n60 in [4, 8, 12, 16]:
    r60 = simulate_mulligan_rate(n60, deck_size=60, hand_size=7) * 100
    n20 = round(n60 / 3)
    r20 = simulate_mulligan_rate(n20, deck_size=20, hand_size=5) * 100
    print(f'  {n60} Basics ({n60/60*100:.0f}%)      {r60:5.1f}%         '
          f'{n20} Basics ({n20/20*100:.0f}%)      {r20:5.1f}%')

print(f'\n→ Rule: {MIN_BASICS} basics/60 cards = {MIN_BASICS//3} basics/20 cards = 20% of deck in both formats')
print(f'→ PPO agent must enforce MIN_BASICS_FRACTION={MIN_BASICS_FRACTION} when sampling decks')
def analyse_deck_smart(deck: pd.DataFrame) -> Dict[str, float]:
    """Compute composite deck score across efficiency, HP, consistency, trainer support."""
    poke  = deck[deck['card_type'] == 'pokemon']
    enrg  = deck[deck['card_type'] == 'energy']
    train = deck[deck['card_type'] == 'trainer']
    hp    = pd.to_numeric(poke['HP'], errors='coerce').dropna()
    dmg   = poke['Damage'].apply(lambda x: parse_damage(str(x))[0])
    cost  = poke['Cost'].apply(lambda x: parse_cost(str(x)).sum().item())
    avg_hp   = hp.mean()   if len(hp)   > 0 else 0
    avg_dmg  = dmg.mean()  if len(dmg)  > 0 else 0
    avg_cost = cost.mean() if len(cost) > 0 else 1
    eff           = avg_dmg / max(avg_cost, 1)
    consistency   = min(len(enrg) / 8.0, 1.0)
    trainer_score = min(len(train) / 4.0, 1.0)
    final = round(eff*10 + avg_hp/10 + consistency*50 + trainer_score*20, 2)
    return {'avg_hp':round(avg_hp,1),'avg_dmg':round(avg_dmg,1),'avg_cost':round(avg_cost,2),
            'efficiency':round(eff,2),'energy_count':int(len(enrg)),
            'trainer_count':int(len(train)),'consistency':round(consistency,2),'SCORE':final}

results: Dict[str, Dict] = {}
for t in ['R','W','G','L','P','F','D','M']:
    try:    results[t] = analyse_deck_smart(build_deck_smart(t))
    except Exception as e: results[t] = {'SCORE':0,'error':str(e)}

display(pd.DataFrame(results).T.sort_values('SCORE', ascending=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Smart Deck Comparison', fontsize=14, fontweight='bold')
types  = list(results.keys())
colors = [TYPE_COLORS.get(t,'#888') for t in types]
scores = [results[t].get('SCORE',0) for t in types]
best   = types[scores.index(max(scores))]
axes[0].bar(types, scores, color=colors, edgecolor='white')
axes[0].set_title('Final Score'); axes[0].set_ylabel('Score')
axes[0].annotate(f'★ {best}', xy=(best, max(scores)),
    xytext=(0,5), textcoords='offset points', ha='center', font_weight='bold')
effs = [results[t].get('efficiency',0) for t in types]
axes[1].bar(types, effs, color=colors, edgecolor='white')
axes[1].set_title('Damage per Energy'); axes[1].set_ylabel('DMG / Energy')
axes[1].axhline(np.mean(effs), color='red', linestyle='--', label=f'Mean {np.mean(effs):.1f}')
axes[1].legend()
hps = [results[t].get('avg_hp',0) for t in types]
axes[2].bar(types, hps, color=colors, edgecolor='white')
axes[2].set_title('Average Pokémon HP'); axes[2].set_ylabel('HP')
axes[2].axhline(np.mean(hps), color='red', linestyle='--', label=f'Mean {np.mean(hps):.0f}')
axes[2].legend()
plt.tight_layout(); plt.show()
print(f'\n★ RECOMMENDED DECK TYPE: {best}  (score={max(scores):.2f})')
from sklearn.linear_model    import Ridge
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics         import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error as _rms
    rmse_fn = lambda y, p: _rms(y, p)
except ImportError:
    from sklearn.metrics import mean_squared_error
    rmse_fn = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))

model_df = df_cards[df_cards['card_type'] == 'pokemon'].copy()
model_df['_hp']   = pd.to_numeric(model_df['HP'], errors='coerce').fillna(0)
model_df['_dmg']  = model_df['Damage'].apply(lambda x: parse_damage(str(x))[0]).astype(float)
model_df['_cost'] = model_df['Cost'].apply(lambda x: parse_cost(str(x)).sum().item())
model_df['_type'] = model_df['Type'].str.extract(r'\{([A-Z])\}')[0].fillna('C')
model_df['_stage']= model_df[STAGE_COL].str.lower().str.extract(r'(basic|stage 1|stage 2)')[0].fillna('basic')
model_df['_exp']  = model_df['Expansion'].fillna('UNK')
model_df = model_df[model_df['_dmg'] > 0].reset_index(drop=True)
print(f'Modeling rows: {len(model_df)} Pokémon with numeric damage')

X = model_df[['_hp','_cost','_type','_stage','_exp']].copy()
y = model_df['_dmg'].values
groups = model_df['Card ID'].values

try:    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except: ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
pre = ColumnTransformer([('num','passthrough',['_hp','_cost']),('cat',ohe,['_type','_stage','_exp'])])

models_ml = {'Mean baseline': None, 'Ridge': Pipeline([('pre',pre),('reg',Ridge(alpha=1.0))])}
try:
    from lightgbm import LGBMRegressor
    models_ml['LightGBM'] = Pipeline([('pre',pre),
        ('reg',LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=63,random_state=42,verbosity=-1))])
    print('LightGBM available ✓')
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    models_ml['HistGBM'] = Pipeline([('pre',pre),
        ('reg',HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,random_state=42))])
    print('LightGBM not installed — using HistGradientBoosting')

gkf = GroupKFold(n_splits=5)
results_ml: Dict[str,Dict] = {n:{'mae':[],'rmse':[],'r2':[],'pred':np.zeros_like(y,dtype=float)} for n in models_ml}
for fold,(tr,va) in enumerate(gkf.split(X, y, groups)):
    y_tr,y_va = y[tr],y[va]
    for name,mdl in models_ml.items():
        if mdl is None: pred = np.full(len(va), y_tr.mean())
        else: mdl.fit(X.iloc[tr],y_tr); pred = mdl.predict(X.iloc[va])
        results_ml[name]['pred'][va] = pred
        results_ml[name]['mae'].append(mean_absolute_error(y_va,pred))
        results_ml[name]['rmse'].append(rmse_fn(y_va,pred))
        results_ml[name]['r2'].append(r2_score(y_va,pred))

score_df = pd.DataFrame({n:{'MAE':np.mean(v['mae']),'RMSE':np.mean(v['rmse']),'R²':np.mean(v['r2'])}
                         for n,v in results_ml.items()}).T.round(3)
print('\n=== DAMAGE PREDICTION MODEL COMPARISON ===')
display(score_df)
champ = score_df['R²'].idxmax()
print(f'\nBest model: {champ}  R²={score_df.loc[champ,"R²"]:.3f}  MAE={score_df.loc[champ,"MAE"]:.1f} dmg')

pred  = results_ml[champ]['pred']; resid = y - pred; lims = [0, float(y.max())*1.05]
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle(f'Damage Predictor: {champ}', fontsize=13, fontweight='bold')
axes[0].scatter(pred, y, s=10, alpha=0.4, color='#6890F0')
axes[0].plot(lims, lims, '--', color='red', lw=1.5)
axes[0].set(xlim=lims, ylim=lims, xlabel='Predicted Damage', ylabel='Actual Damage', title='Predicted vs Actual'); axes[0].grid(alpha=0.3)
axes[1].scatter(model_df['_cost'], resid, s=10, alpha=0.4, color='#F08030')
axes[1].axhline(0, color='red', lw=1.5, ls='--')
axes[1].set(xlabel='Total Energy Cost', ylabel='Residual', title='Residuals vs Cost'); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
print(f'\nPPO Neural Net → strategic action selection in self-play')
print(f'{champ}         → predicts expected damage from static features')
print('Both are complementary in a full AI system.')
print('=' * 62)
print('FULL PIPELINE TEST ON REAL DATA')
print('=' * 62)

print(f'\n✓ Dataset: {len(df_raw)} rows, {len(df_cards)} unique cards')
print(f'  Pokémon: {len(pokemon_u)}  Trainer: {len(trainer_u)}  Energy: {len(energy_u)}')

best_type  = max(['R','W','G','L','P','F','D','M'],
    key=lambda t: analyse_deck_smart(build_deck_smart(t)).get('SCORE',0))
best_deck  = build_deck_smart(best_type)
best_stats = analyse_deck_smart(best_deck)
print(f'\n✓ Best deck type : {best_type}  (score={best_stats["SCORE"]:.2f})')
print(f'  Avg HP={best_stats["avg_hp"]}  Avg DMG={best_stats["avg_dmg"]}  Efficiency={best_stats["efficiency"]}')
print(f'  Energy: {best_stats["energy_count"]}  Trainer: {best_stats["trainer_count"]}')
print('\n✓ Best deck cards:')
display(best_deck)

sample_ids = []
for name in best_deck[best_deck['card_type']=='pokemon']['Card Name'].values[:6]:
    match = df_cards[df_cards['Card Name'] == name]
    if len(match) > 0: sample_ids.append(match['Card ID'].values[0])
if sample_ids:
    card_feats = encoder.lookup(sample_ids[:min(6,len(sample_ids))])
    state = torch.zeros(1, STATE_DIM)
    flat  = card_feats.flatten()
    state[0, :min(len(flat), STATE_DIM)] = flat[:STATE_DIM]
    action, log_prob, value = net.act(state, deterministic=False)
    names_used = [df_cards[df_cards['Card ID']==cid]['Card Name'].values[0] for cid in sample_ids[:3]]
    print(f'\n✓ Neural network inference on real deck cards:')
    print(f'  Input cards : {names_used}')
    print(f'  Action      : {action.item()}  Log prob: {log_prob.item():.4f}  Value: {value.item():.4f}')

print('\n✓ Damage mechanics test:')
for name in best_deck[best_deck['card_type']=='pokemon']['Card Name'].values[:3]:
    row = pokemon_u[pokemon_u['Card Name']==name]
    if len(row) > 0:
        row = row.iloc[0]
        base, scale = parse_damage(str(row['Damage']))
        dmg = compute_damage(torch.tensor([base]),torch.tensor([scale]),
                             torch.tensor([0]),torch.tensor([0]),
                             torch.tensor([False]),torch.tensor([False]))
        print(f'  {name}: {base} base → final {dmg.item()}')

print('\n✓ Energy payment test:')
for _, row in pokemon_u[pokemon_u['Cost'].str.len()>0].head(3).iterrows():
    cost_vec = parse_cost(str(row['Cost']))
    avail    = torch.zeros(1, N_TYPES, dtype=torch.long)
    avail[0, TYPE_TO_IDX['R']] = 4
    avail[0, TYPE_TO_IDX['C']] = 2
    result = can_pay(avail, cost_vec)
    print(f'  {row["Card Name"]}: cost={row["Cost"]} → {"✓ CAN" if result.item() else "✗ CANNOT"} pay with 4R+2C')

print(f'\n{"="*62}')
print('SUMMARY')
print(f'{"="*62}')
print(f'  Cards in dataset     : {len(df_cards)}')
print(f'  Unique Pokémon types : {pokemon_u["Type"].str.extract(r"{([A-Z])}")[0].nunique()}')
print(f'  Recommended type     : {best_type}')
print(f'  Network parameters   : {sum(p.numel() for p in net.parameters()):,}')
print(f'  Training device      : {cfg.device}')
print(f'  All systems          : READY ✓')
class TestAll(unittest.TestCase):

    # ── Parsers ────────────────────────────────────────────
    def test_parse_cost_empty(self):
        for s in ('','n/a','0','nan'):
            self.assertTrue(parse_cost(s).eq(0).all(), f"failed for '{s}'")

    def test_parse_cost_dot_symbol(self):
        v = parse_cost('{R}●●')
        self.assertEqual(v[TYPE_TO_IDX['R']].item(), 1)
        self.assertEqual(v[TYPE_TO_IDX['C']].item(), 2)

    def test_parse_cost_mixed(self):
        v = parse_cost('{R}{R}{C}')
        self.assertEqual(v[TYPE_TO_IDX['R']].item(), 2)
        self.assertEqual(v[TYPE_TO_IDX['C']].item(), 1)

    def test_parse_damage_variants(self):
        self.assertEqual(parse_damage('80'),  (80, 0.0))
        self.assertEqual(parse_damage('30+'), (30, 1.0))
        self.assertEqual(parse_damage('20×'), (0, 20.0))
        self.assertEqual(parse_damage('nan'), (0, 0.0))

    # ── Energy logic ───────────────────────────────────────
    def test_can_pay_colorless_wildcard(self):
        avail = torch.zeros(2, N_TYPES, dtype=torch.long)
        avail[0, TYPE_TO_IDX['R']] = 2
        avail[1, TYPE_TO_IDX['R']] = 1
        avail[1, TYPE_TO_IDX['G']] = 1
        result = can_pay(avail, parse_cost('{R}{C}'))
        self.assertTrue(result[0].item())
        self.assertTrue(result[1].item())

    def test_can_pay_wrong_type_fails(self):
        avail = torch.zeros(1, N_TYPES, dtype=torch.long)
        avail[0, TYPE_TO_IDX['G']] = 3
        self.assertFalse(can_pay(avail, parse_cost('{R}{R}')).item())

    # ── Damage calculation ─────────────────────────────────
    def test_damage_weakness_resistance(self):
        out = compute_damage(torch.tensor([80]), torch.tensor([0.0]),
                             torch.tensor([0]),  torch.tensor([0]),
                             torch.tensor([True]), torch.tensor([True]))
        self.assertEqual(out.item(), 130)   # 80×2 − 30 = 130

    def test_damage_floor_zero(self):
        out = compute_damage(torch.tensor([10]), torch.tensor([0.0]),
                             torch.tensor([0]),  torch.tensor([0]),
                             torch.tensor([False]), torch.tensor([True]))
        self.assertEqual(out.item(), 0)     # 10 − 30 = −20 → clamp to 0

    def test_damage_scaling(self):
        out = compute_damage(torch.tensor([0]),  torch.tensor([20.0]),
                             torch.tensor([3]),  torch.tensor([0]),
                             torch.tensor([False]), torch.tensor([False]))
        self.assertEqual(out.item(), 60)    # 20 × 3 = 60

    # ── Encoder ────────────────────────────────────────────
    def test_encoder_shape_range(self):
        m = encoder.card_matrix
        self.assertEqual(m.ndim, 2)
        self.assertGreaterEqual(m.min().item(), 0.0)
        self.assertLessEqual(m.max().item(), 1.0 + 1e-6)

    # ── Neural network ─────────────────────────────────────
    def test_net_shapes(self):
        s = torch.randn(4, STATE_DIM)
        logits, v = net(s)
        self.assertEqual(logits.shape, (4, N_ACTIONS))
        self.assertEqual(v.shape, (4, 1))

    def test_net_mask(self):
        s    = torch.randn(1, STATE_DIM)
        mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
        mask[0, 7] = True
        logits, _ = net(s, mask)
        self.assertAlmostEqual(logits.softmax(-1)[0, 7].item(), 1.0, places=4)

    # ── Deck builder (size=60 — official format) ───────────
    def test_deck_smart_has_pokemon(self):
        self.assertIn('pokemon', build_deck_smart('R', size=60)['card_type'].values)

    def test_deck_smart_has_energy(self):
        self.assertIn('energy', build_deck_smart('R', size=60)['card_type'].values)

    def test_deck_smart_energy_count(self):
        deck = build_deck_smart('R', size=60)
        self.assertGreaterEqual((deck['card_type'] == 'energy').sum(), 4)

    def test_deck_smart_cost_compatible(self):
        deck = build_deck_smart('R', size=60)
        for _, row in deck[deck['card_type']=='pokemon'].iterrows():
            self.assertTrue(cost_compatible(str(row['Cost']), 'R', max_off=1),
                f"{row['Card Name']} has incompatible cost: {row['Cost']}")

    # ── Analyser ───────────────────────────────────────────
    def test_deck_analyser_keys(self):
        s = analyse_deck_smart(build_deck_smart('W', size=60))
        for k in ['avg_hp','avg_dmg','efficiency','SCORE']:
            self.assertIn(k, s)

    # ── Cost compatibility ─────────────────────────────────
    def test_cost_compatible_logic(self):
        self.assertTrue(cost_compatible('{R}{R}●', 'R', max_off=1))
        self.assertFalse(cost_compatible('{R}{P}{M}', 'R', max_off=1))

    # ── Mulligan simulation (real rules: 60 cards, 7-card hand) ──
    def test_mulligan_rate_bounds(self):
        rate_12 = simulate_mulligan_rate(12, deck_size=60, hand_size=7)
        self.assertGreaterEqual(rate_12, 0.0)
        self.assertLessEqual(rate_12, 1.0)
        # community floor: 12 basics in 60-card deck must be under 10%
        self.assertLess(rate, 0.10)

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(unittest.TestLoader().loadTestsFromTestCase(TestAll))
passed = result.testsRun - len(result.failures) - len(result.errors)
print(f"\n{'✓ ALL PASSED' if result.wasSuccessful() else '✗ FAILURES'} — {passed}/{result.testsRun} tests")
