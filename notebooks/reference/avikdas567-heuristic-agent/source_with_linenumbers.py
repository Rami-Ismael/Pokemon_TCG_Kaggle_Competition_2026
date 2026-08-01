import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import tarfile

import warnings
warnings.filterwarnings("ignore", message=".*missing from font.*")

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2

CARD_DATA_PATH = "/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv"
SAMPLE_DECK_PATH = "/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/deck.csv"

if not os.path.exists(CARD_DATA_PATH):
    CARD_DATA_PATH = "EN_Card_Data.csv"
    SAMPLE_DECK_PATH = "deck.csv"

cards_df = pd.read_csv(CARD_DATA_PATH)
deck_df = pd.read_csv(SAMPLE_DECK_PATH, header=None)

print(f"[INFO] Card Database loaded successfully. Total Records: {len(cards_df)}")
print(f"[INFO] Active User Deck verified. Total Items: {len(deck_df)}")
# Visualizing distribution of card classification categories
plt.figure(figsize=(12, 5))
stage_series = cards_df['Stage (Pokémon)/Type (Energy and Trainer)'].value_counts()
sns.barplot(x=stage_series.values, y=stage_series.index, hue=stage_series.index, palette="viridis", legend=False)
plt.title("Ecosystem Distribution: Card Type & Stage Ratios", fontsize=14, fontweight='bold')
plt.xlabel("Absolute Card Occurrence Frequency")
plt.ylabel("Classification Class")
plt.tight_layout()
plt.show()
# Analyzing Pokémon Card Type Affinity
pkmn_unique = cards_df[cards_df['Stage (Pokémon)/Type (Energy and Trainer)'].str.contains('Pokémon', na=False)].drop_duplicates(subset=['Card ID'])
plt.figure(figsize=(12, 5))
type_series = pkmn_unique['Type'].value_counts()
sns.barplot(x=type_series.index, y=type_series.values, hue=type_series.index, palette="plasma", legend=False)
plt.title("Pokémon Pool Density Across Elemental Archetypes", fontsize=14, fontweight='bold')
plt.xlabel("Elemental Flag Signifier")
plt.ylabel("Unique Card Volume")
plt.tight_layout()
plt.show()
# Metric distribution analysis for Pokémon Hit Points (HP)
plt.figure(figsize=(10, 5))
sns.histplot(data=pkmn_unique, x="HP", kde=True, color="darkviolet", bins=25, alpha=0.75)
plt.axvline(pkmn_unique['HP'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean HP: {pkmn_unique['HP'].mean():.1f}")
plt.title("Statistical Spectrum Analysis of Pokémon HP Parameters", fontsize=14, fontweight='bold')
plt.xlabel("Hit Points (HP Values)")
plt.ylabel("Frequency Probability Mass")
plt.legend()
plt.tight_layout()
plt.show()
def execute_deck_audit(deck_list, ref_df):
    """
    Validates full structural conformity parameters for a 60-card tournament deck.
    """
    issues = []
    if len(deck_list) != 60:
        issues.append(f"Card length deviation: expected exactly 60 cards, found {len(deck_list)}.")
        
    counts = pd.Series(deck_list).value_counts()
    name_frequency = {}
    contains_basic_pkmn = False
    
    for c_id, occurrences in counts.items():
        matches = ref_df[ref_df['Card ID'] == c_id]
        if matches.empty:
            issues.append(f"Anomalous Card ID {c_id}: absent from meta registry.")
            continue
        row = matches.iloc[0]
        c_name = row['Card Name']
        c_stage = str(row['Stage (Pokémon)/Type (Energy and Trainer)'])
        
        if 'Basic Pokémon' in c_stage:
            contains_basic_pkmn = True
        if 'Basic Energy' in c_stage:
            continue
            
        name_frequency[c_name] = name_frequency.get(c_name, 0) + occurrences
        
    for name, copies in name_frequency.items():
        if copies > 4:
            issues.append(f"Duplicate rule failure for '{name}': {copies} copies violates the max-4 limit.")
            
    if not contains_basic_pkmn:
        issues.append("Starting configuration check failure: zero Basic Pokémon available.")
        
    if not issues:
        print("🚀 Validation Audit Passed: Submission deck satisfies all structural tournament constraints.")
        return True
    else:
        print("❌ Validation Audit Failed!")
        for issue in issues:
            print(f"  -> {issue}")
        return False

execute_deck_audit(deck_df[0].tolist(), cards_df)
%%writefile main.py
import os
import random

def read_deck_csv() -> list[int]:
    """
    Robust asset loader for local and remote simulation directories.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
        
    with open(file_path, "r") as file:
        csv_lines = file.read().split("\n")
        
    deck = []
    for i in range(60):
        if csv_lines[i].strip():
            deck.append(int(csv_lines[i].strip()))
    return deck

def agent(obs_dict: dict) -> list[int]:
    """
    Grandmaster Strategic Heuristic Engine (Dictionary-Safe Version).
    Parses raw observation objects directly without external dependencies.
    """
    # Helper to access properties seamlessly whether obs_dict is a dict or object
    def safe_get(obj, key, default=None):
        if obj is None:
            return default
        if hasattr(obj, 'get'):
            return obj.get(key, default)
        return getattr(obj, key, default)

    select = safe_get(obs_dict, 'select')
    
    # Initialization/Configuration Phase: Return the 60-card deck list
    if select is None:
        return read_deck_csv()
        
    options_pool = safe_get(select, 'option', [])
    total_options = len(options_pool)
    
    min_req = safe_get(select, 'minCount', 1)
    max_req = safe_get(select, 'maxCount', 1)
    
    # Dynamically scale action counts to legal bounds
    target_count = min(max_req, total_options)
    
    scored_indices = []
    
    # Assign strategic utility values to legal options
    for idx, opt in enumerate(options_pool):
        score = 10  # Base utility floor
        opt_str = str(opt).lower()
        
        # Priority Layer 1: Evolution & Energy Acceleration
        if "evolve" in opt_str:
            score += 150
        if "attach" in opt_str:
            score += 100
        if "play_basic" in opt_str or "bench" in opt_str:
            score += 50
            
        # Priority Layer 2: Maximizing offensive damage capability
        if "attack" in opt_str:
            score += 80
            damage = safe_get(opt, 'damage', 0)
            if damage:
                score += int(damage)
                
        scored_indices.append((idx, score))
        
    # Sort choices in descending order of utility scores
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Extract unique option indexes up to target constraint limits
    selected_actions = [scored_indices[i][0] for i in range(target_count)]
    
    return selected_actions
# Write out local deck layout file configuration
deck_df.to_csv("deck.csv", index=False, header=False)

archive_target = "submission.tar.gz"
payload_manifest = ["main.py", "deck.csv"]

print(f"📦 Generating production tarball deployment archive: {archive_target}")
with tarfile.open(archive_target, "w:gz") as archive:
    for target_asset in payload_manifest:
        if os.path.exists(target_asset):
            archive.add(target_asset)
            print(f"  -> Verified & Packed asset: {target_asset}")
        else:
            print(f"  -> ⚠️ Critical Asset Missing: {target_asset}")

if os.path.exists(archive_target):
    sz = os.path.getsize(archive_target) / 1024
    print(f"\n🎉 Success! Deployable artifact compressed at: {os.path.abspath(archive_target)}")
    print(f"📂 Finished Archive Mass: {sz:.2f} KB")
