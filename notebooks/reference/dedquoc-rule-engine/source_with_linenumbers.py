# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')
import pandas as pd
import numpy as np

# The path to the confirmed data file from your Step 1 log
en_card_path = '/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv'

print("--- 1. CHECK THE OVERVIEW OF ATTRIBUTE INFORMATION (COLUMNS)---")
try:
    df = pd.read_csv(en_card_path)
    print(f"Total number of cards in the database: {df.shape[0]} card.")
    print(f"Total number of attributes (Columns): {df.shape[1]}\n")
    
    # Display a list of columns along with their data types and the number of non-null values
    print(df.info())
    
except Exception as e:
    print(f"Error reading data file: {e}")

print("\n--- 2. DISPLAY THE FIRST 5 LINES OF DATA (PREVIEW) ---")
try:
    # Round the display and toggle long text columns for easier viewing.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head(5))
except Exception as e:
    pass

print("\n--- 3. SCAN FOR INFORMATION ON KEY CARDS IN THE DECK (TARGET SEARCH) ---")
try:
    # Filter and check information on core cards in the Mega Lucario ex strategy.
    target_ids = [673, 674, 675, 676, 677, 678, 1123, 1182]
    # Assume the column containing the ID is named 'cardId' or 'id' or 'Card ID'
    id_col = [col for col in df.columns if 'id' in col.lower() or 'card' in col.lower()][0]
    
    df_targets = df[df[id_col].isin(target_ids)]
    print(df_targets[[id_col, 'name', 'hp', 'stage', 'type'] if 'name' in df.columns else df_targets.columns[:5]])
except Exception as e:
    print(f"Unable to filter target details: {e}. Proceed to display all column names for your reference::")
    print(list(df.columns))
import pandas as pd

en_card_path = '/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv'
df = pd.read_csv(en_card_path)

# List of IDs of core cards in the Deck
core_ids = [673, 674, 675, 676, 677, 678, 1123, 1182]

print("--- ANALYSIS OF THE COMBAT INDEX OF THE CORE TEAM LINEUP ---")
df_core = df[df['Card ID'].isin(core_ids)]

# Filter important columns related to combat and abilities
cols_to_show = [
    'Card ID', 'Card Name', 'Stage (Pokémon)/Type (Energy and Trainer)', 
    'HP', 'Move Name', 'Cost', 'Damage', 'Effect Explanation'
]

# Hiển thị chi tiết
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)
display(df_core[cols_to_show].drop_duplicates().sort_values(by='Card ID'))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


en_card_path = '/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv'
df = pd.read_csv(en_card_path)


df_pokemon = df[df['HP'].notna()].copy()

def clean_damage(val):
    if pd.isna(val):
        return 0
  
    nums = re.findall(r'\d+', str(val))
    return int(nums[0]) if nums else 0

df_pokemon['Clean_Damage'] = df_pokemon['Damage'].apply(clean_damage)


df_pokemon['Is_Core'] = df_pokemon['Card ID'].isin(core_ids)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.set_theme(style="whitegrid")


sns.scatterplot(
    data=df_pokemon, 
    x='HP', 
    y='Clean_Damage', 
    hue='Is_Core', 
    palette={False: '#bdc3c7', True: '#e74c3c'},
    size='Is_Core',
    sizes={False: 20, True: 150},
    alpha=0.8,
    ax=axes[0]
)


core_highlights = df_pokemon[df_pokemon['Is_Core']].drop_duplicates(subset=['Card ID'])
for _, row in core_highlights.iterrows():
    axes[0].annotate(
        row['Card Name'], 
        (row['HP'], row['Clean_Damage']),
        textcoords="offset points", 
        xytext=(0,10), 
        ha='center', 
        fontsize=9, 
        weight='bold',
        color='#c0392b'
    )

axes[0].set_title('Power Matrix: Health (HP) vs. Damage', fontsize=14, weight='bold')
axes[0].set_xlabel('Pokémon Health (HP)', fontsize=12)
axes[0].set_ylabel('Skill Damage', fontsize=12)


def count_energy_cost(val):
    if pd.isna(val):
        return 0
    # Đếm số lượng ký tự nằm trong dấu ngoặc nhọn, ví dụ {F}{F}{F} -> 3
    return len(re.findall(r'\{.*?\}', str(val)))

df_pokemon['Energy_Cost_Count'] = df_pokemon['Cost'].apply(count_energy_cost)

sns.countplot(
    data=df_pokemon, 
    x='Energy_Cost_Count', 
    hue='Is_Core', 
    palette={False: '#95a5a6', True: '#f1c40f'},
    ax=axes[1]
)

axes[1].set_title('Energy Cost Distribution to Activate the Technique', fontsize=14, weight='bold')
axes[1].set_xlabel('Energy required (Number of particles)', fontsize=12)
axes[1].set_ylabel('Number of Cards / Skills', fontsize=12)
axes[1].legend(title='Core lineup', labels=['Other cards', 'Lucario Deck Cards'])

plt.tight_layout()
plt.show()

# 4. In thống kê mô tả nhanh
print("--- STRATEGIC POSITIONING PARAMETERS OF MEGA LUCARIO EX ACROSS THE META ---")
print(f"Highest HP in the entire meta: {df_pokemon['HP'].max()} HP (Mega Lucario ex của bạn đạt: 340.0 HP)")
print(f"Highest damage output in the entire meta: {df_pokemon['Clean_Damage'].max()} Damage (Mega Lucario ex đạt: 270 Damage)")
import pandas as pd

# Read standard data files from Kaggle memory.
en_card_path = '/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv'
df = pd.read_csv(en_card_path)

print("--- 🕵️‍♂️ Searching for Lucario's actual ID in the database. ---")
# Find all lines containing the word 'Lucario' or 'Riolu' regardless of case.
df_lucario = df[df['Card Name'].str.contains('Lucario|Riolu', case=False, na=False)]

if not df_lucario.empty:
    print(df_lucario[['Card ID', 'Card Name', 'HP', 'Move Name', 'Damage']].drop_duplicates().to_string())
else:
    print("❌ No cards with Lucario's name were found! Scanning the entire Pokémon Stage card list. 1...")
    # If you can't see it, print out the first 20 Pokémon Stage 1 cards to see how the ID structure increases.
    print(df[df['Stage (Pokémon)/Type (Energy and Trainer)'].str.contains('Stage 1', na=False)][['Card ID', 'Card Name']].head(20))
%%writefile main.py
## testing > has score init : Latest Score 600.0 V8 inside notebook outside public : 325.8
import os
import sys
from collections import defaultdict


# Mega Lucario ex Deck - Kaggle Production Ready Edition 
# Optimized & Bug-Fixed Version


# 1. PRIORITY FOR TOPPING UP HIGH-LEVEL LIBRARIES FROM KIYOTAH DATASET (SIM LOOP SPEED UP)
CUSTOM_CG_PATH = '/kaggle/input/datasets/kiyotah/cg-lib'
COMP_CG_PATH = '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/sample_submission'

if os.path.exists(CUSTOM_CG_PATH):
    if CUSTOM_CG_PATH not in sys.path:
        sys.path.insert(0, CUSTOM_CG_PATH)
else:
    if COMP_CG_PATH not in sys.path:
        sys.path.insert(0, COMP_CG_PATH)

# Proceed with a safe import from the found game engine.
from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

#2. AUTOMATICALLY READ YOUR MEGA LUCARIO EX DECK FROM YOUR ACTUAL DATASET
my_deck_path = "/kaggle/input/datasets/kiyotah/mega-lucario-ex-deck/deck.csv"
my_deck = []

if os.path.exists(my_deck_path):
    with open(my_deck_path, "r") as file:
        csv_lines = file.read().splitlines()
    my_deck = [int(line.strip()) for line in csv_lines if line.strip() != ""]
else:
    fallback_path = "deck.csv"
    if not os.path.exists(fallback_path):
        fallback_path = "/kaggle_simulations/agent/deck.csv"
    if os.path.exists(fallback_path):
        with open(fallback_path, "r") as file:
            my_deck = [int(line.strip()) for line in file.read().splitlines() if line.strip() != ""]

# Initialize a hidden global trading card database to prevent crashes
try:
    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
except Exception:
    card_table = {}

# Defining the Deck Card ID Constant
Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Gravity_Mountain = 1252
Basic_Fighting_Energy = 6

class AttackPlan:
    def __init__(self):
        self.attacker = -1
        self.target = -1
        self.attack_index = -1
        self.remain_hp = -1
        self.energy = False

# Safely initialize Agent state
agent_state = {
    "plan": AttackPlan(),
    "pre_turn": 0,
    "ability_used": False
}

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    # Function to extract card information safely, completely free from IndexError"""
    try:
        ps = obs.current.players[player_index]
        if area == AreaType.DECK: return obs.select.deck[index] if obs.select and obs.select.deck else None
        if area == AreaType.HAND: return ps.hand[index]
        if area == AreaType.DISCARD: return ps.discard[index]
        if area == AreaType.ACTIVE: return ps.active[index] if ps.active else None
        if area == AreaType.BENCH: return ps.bench[index]
        if area == AreaType.PRIZE: return ps.prize[index]
        if area == AreaType.STADIUM: return obs.current.stadium[index] if obs.current.stadium else None
        if area == AreaType.LOOKING: return obs.current.looking[index] if obs.current.looking else None
        return None
    except Exception:
        return None

def agent(obs_dict: dict) -> list[int]:
    # Main Agent handler function to interact with the Kaggle Simulation environment
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        # Fallback khẩn cấp nếu định dạng obs_dict lỗi từ phía môi trường
        return [0]

    # If the system requests a return of the deck at the beginning of the match (Initialization Phase)
    if obs.select is None:
        return my_deck if len(my_deck) == 60 else [6] * 60
        
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    my_prize = len(my_state.prize) if my_state.prize else 0

    # Reset state when starting a new turn
    if agent_state["pre_turn"] != state.turn:
        agent_state["pre_turn"] = state.turn
        agent_state["plan"] = AttackPlan()
        agent_state["ability_used"] = False
            
    discard_counts = defaultdict(int)
    if my_state.discard:
        for card in my_state.discard:
            discard_counts[card.id] += 1

    def safe_respond(indices: list[int] = None) -> list[int]:
        # Set the default value to empty [] to completely prevent crashes when calling without parameters """
        if indices is None:
            indices = []
        valid_indices = [i for i in indices if 0 <= i < len(select.option)]
        valid_indices = list(dict.fromkeys(valid_indices))
        
       # Add valid indexes if the Engine's minimum minCount has not been reached.
        if len(valid_indices) < select.minCount:
            for i in range(len(select.option)):
                if i not in valid_indices:
                    valid_indices.append(i)
                if len(valid_indices) == select.minCount:
                    break
                    
        # Cut back if the Engine's maximum maxCount is exceeded.
        if len(valid_indices) > select.maxCount:
            valid_indices = valid_indices[:select.maxCount]
            
        return valid_indices

    # --- MAIN PHASE: MAKING TACTICAL DECISIONS ---
    if context == SelectContext.MAIN:
        best_option_index = -1
        best_score = -100000

        for i, o in enumerate(select.option):
            current_score = 0
            
            # ATTACK (Highest priority action to terminate the decision chain)
            if o.type == OptionType.ATTACK:
                current_score += 20000
                active_pokemon = my_state.active[0] if (my_state.active and len(my_state.active) > 0) else None
                
                if active_pokemon:
                    if active_pokemon.id == Mega_Lucario_ex:
                        if o.attackId == 982:  
                            current_score += 60 * min(3, discard_counts[Basic_Fighting_Energy])
                        elif o.attackId == 983:  
                            current_score += 2000
                            if my_prize <= 2:  
                                current_score += 1000  
                    elif active_pokemon.id == Hariyama:
                        current_score += 500
                
                if current_score > best_score:
                    best_score = current_score
                    best_option_index = i

            # ENERGY TIE
            elif o.type == OptionType.ENERGY and best_score < 15000:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card and card.id == Basic_Fighting_Energy:
                    current_score += 5000
                    if o.inPlayArea == AreaType.ACTIVE:
                        current_score += 1000  
                
                if current_score > best_score:
                    best_score = current_score
                    best_option_index = i

            # EVOLVE
            elif o.type == OptionType.EVOLVE and best_score < 12000:
                current_score += 4000
                if current_score > best_score:
                    best_score = current_score
                    best_option_index = i

            # USING TRAINER CARDS
            elif o.type == OptionType.PLAY and best_score < 10000:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card:
                    if card.id in [Carmine, Lillie_Determination]:
                        current_score += 3000  
                    elif card.id in [Dusk_Ball, Premium_Power_Pro]:
                        current_score += 2500  
                    elif card.id == Boss_Orders:
                        current_score += 2000  
                    else:
                        current_score += 500
                
                if current_score > best_score:
                    best_score = current_score
                    best_option_index = i

        if best_option_index != -1:
            return safe_respond([best_option_index])

    # --- OTHER FALLBACKS: HANDLING OTHER REQUIRED CONTEXT TO AVOID TIMEOUT ---
    # Always return the first valid action if it falls into non-MAIN contexts (e.g., selecting the active Pokemon at the beginning)
    if select.minCount > 0:
        return safe_respond([0])
        
    return safe_respond([])
import tarfile
import os
import shutil

if os.path.exists('submission.tar.gz'):
    os.remove('submission.tar.gz')

# Đồng bộ hóa hạ tầng binary
source_cg_dir = '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg'
if os.path.exists('cg'): shutil.rmtree('cg')
shutil.copytree(source_cg_dir, 'cg')

# Đồng bộ hóa tệp cấu trúc bộ bài chuẩn mẫu từ cuộc thi
source_deck = '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv'
if os.path.exists(source_deck): shutil.copy(source_deck, 'deck.csv')

with tarfile.open("submission.tar.gz", "w:gz") as tar:
    tar.add('main.py', arcname='main.py')
    tar.add('deck.csv', arcname='deck.csv')
    tar.add('cg', arcname='cg')

print("Packaging process complete! Script has been optimized based on the actual ID.")
