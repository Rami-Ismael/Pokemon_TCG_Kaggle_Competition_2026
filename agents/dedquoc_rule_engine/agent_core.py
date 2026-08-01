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
