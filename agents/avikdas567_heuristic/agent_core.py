"""avikdas567 heuristic agent, wired for benchmark_agents.py.
Source: notebooks/reference/avikdas567-heuristic-agent (verbatim main.py cell;
submission-packaging cells stripped, read_deck_csv() repointed to this dir's deck.csv).
NOTE: weak policy -- scores options by substring-matching str(option) ("attack",
"evolve", "attach"); included as a low/mid-tier real opponent, not a strong one.
Uses the sample Mega Lucario ex deck. Returns list[int] option indices.
"""
import os
import random

def read_deck_csv() -> list[int]:
    """
    Robust asset loader for local and remote simulation directories.
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
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
