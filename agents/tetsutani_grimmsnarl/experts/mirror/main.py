from .manual_policy import DECK,choose
PLAYER_NAME='public fixed-deck expert'
BUILD_ID='public_fixed_deck_expert'
def agent(obs_dict:dict)->list[int]:
    return choose(obs_dict)
