from .manual_policy import DECK,choose
PLAYER_NAME='fixed-deck expert'
BUILD_ID='ptcg_fixed_deck_expert_hierarchical_rule_policy_v24_no_learning'
def agent(obs_dict:dict)->list[int]:
    return choose(obs_dict)
