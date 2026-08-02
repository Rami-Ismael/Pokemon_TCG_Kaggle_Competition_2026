# main.py - PTCG AI Battle agent (self-contained, standard library only).
# Decision priority: lethal KO > escape when nearly KO'd > setup ability >
# evolve into attacker > attach energy > consistency trainer > play > chip attack.
import re

try:
    from cg.api import to_observation_class
except (ImportError, AttributeError):
    to_observation_class = None

LOW_HP_ABS = 50
LOW_HP_FRAC = 0.30
DMG_WEIGHT = 0.15
TRAINER_WEIGHTS = {
    "draw": 9, "professor": 10, "research": 6, "iono": 9, "arven": 8,
    "search": 8, "ball": 7, "energy": 4, "attach": 5, "accelerat": 6,
    "switch": 5, "rare candy": 8, "evolution": 5, "boss": 8, "gust": 8,
    "heal": 3, "potion": 3,
}
ABILITY_WEIGHTS = {"draw": 6, "search": 6, "knock": 8, "energy": 4, "extra": 3, "ability": 1}
DEFAULT_FLAGS = {"ko": True, "ability": True, "retreat": True, "trainer": True, "simple": False, "aggressive": False}
PRIMARY_TYPE = '{P}'
AGENT_DEBUG_LOGGING = False

# ---------- STRUCTURED DEBUG LOGGING ----------
def log(level, message, data=None):
    """Optional structured logger for tracing agent decisions.

    Disabled by default to keep inference fast and quiet in competition runtime.
    When AGENT_DEBUG_LOGGING is True, it prints compact decision diagnostics.
    """
    if not AGENT_DEBUG_LOGGING:
        return
    try:
        print(f"[{str(level).upper()}] {message}")
        if data is not None:
            print("  ", data)
    except Exception:
        pass


# ---------- ADAPTIVE STRATEGY EXTENSION ----------
OPPONENT_ARCHETYPE_UNKNOWN = 0
OPPONENT_ARCHETYPE_AGGRO = 1
OPPONENT_ARCHETYPE_CONTROL = 2
OPPONENT_ARCHETYPE_EVOLUTION = 3


def _detect_opponent_archetype(obs, history=None):
    """Infer the opponent archetype from public/visible observation signals.

    The detector is deliberately conservative: explicit test/runtime hints are
    respected first, then public clues such as active name, recent damage,
    opponent bench size, and attached energy are used. If evidence is weak, it
    returns UNKNOWN rather than hallucinating a matchup because apparently even
    card agents deserve epistemic humility.
    """
    explicit = _safe_get(obs, "opp_archetype", None)
    if explicit is None:
        explicit = _safe_get(obs, "opponent_archetype", None)
    if explicit is None:
        explicit = _safe_get(obs, "opponentArchetype", None)
    try:
        if explicit is not None:
            return int(explicit)
    except Exception:
        pass

    opp_active = _safe_get(obs, "opponentActive", None)
    if opp_active is None:
        opp_active = _safe_get(obs, "opponent_active", None)
    active_name = _safe_str(_safe_get(opp_active, "name", "")) if opp_active is not None else ""
    active_text = _option_text(opp_active)
    active_upper = (active_name + " " + active_text).upper()
    if any(tag in active_upper for tag in (" EX", "EX ", "-EX", " V", "V ", " GX", "GX ")):
        return OPPONENT_ARCHETYPE_AGGRO

    recent_damage = _find_num(obs, ["lastDamageTaken", "opponentLastAttackDamage", "damageTaken", "recentDamage"], depth=3)
    if isinstance(recent_damage, (int, float)) and recent_damage >= 120:
        return OPPONENT_ARCHETYPE_AGGRO

    opp_energy = _find_num(obs, ["opponentEnergyCount", "attachedEnergy", "energyCount", "energiesAttached"], depth=3)
    if isinstance(opp_energy, (int, float)) and opp_energy >= 3:
        return OPPONENT_ARCHETYPE_AGGRO

    visible_bench = _find_num(obs, ["opponentBenchCount", "oppBenchCount", "benchCount", "benchSize"], depth=3)
    if isinstance(visible_bench, (int, float)) and visible_bench >= 4:
        return OPPONENT_ARCHETYPE_EVOLUTION

    if history:
        try:
            hist_text = _safe_str(history).upper()
            hist_dmg = _extract_damage(hist_text)
            if hist_dmg >= 120:
                return OPPONENT_ARCHETYPE_AGGRO
            if "RARE CANDY" in hist_text or "EVOLVE" in hist_text:
                return OPPONENT_ARCHETYPE_EVOLUTION
        except Exception:
            pass
    return OPPONENT_ARCHETYPE_UNKNOWN


def _adjust_strategy(archetype):
    """Return runtime flags for option scoring. Values bind at decision time, not fit time."""
    if archetype == OPPONENT_ARCHETYPE_AGGRO:
        return {"ko": True, "ability": True, "retreat": True, "trainer": True,
                "simple": False, "aggressive": True}
    if archetype == OPPONENT_ARCHETYPE_CONTROL:
        return {"ko": True, "ability": True, "retreat": True, "trainer": True,
                "simple": False, "aggressive": False}
    if archetype == OPPONENT_ARCHETYPE_EVOLUTION:
        return {"ko": True, "ability": True, "retreat": True, "trainer": True,
                "simple": False, "aggressive": True}
    return dict(DEFAULT_FLAGS)


def _safe_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_str(x):
    try:
        if x is None:
            return ""
        if hasattr(x, "name") and not isinstance(x, (str, int, float)):
            return str(x.name)
        return str(x)
    except Exception:
        return ""


def _get_select(obs):
    return _safe_get(obs, "select", None)


def _get_options(select):
    opts = _safe_get(select, "option", None)
    if opts is None:
        opts = _safe_get(select, "options", None)
    try:
        return list(opts) if opts is not None else []
    except Exception:
        return []


def _option_type(option):
    if hasattr(option, "type"):
        t = option.type
    else:
        t = _safe_get(option, "type", None)
    if t is None:
        if hasattr(option, "name"):
            return str(option.name)
        return _safe_str(option)
    if hasattr(t, "name"):
        return str(t.name)
    return _safe_str(t)


def _select_type_name(select):
    return _safe_str(_safe_get(select, "type", "")).upper()


def _option_text(option):
    """Return a stable human-readable text representation for a legal option.

    Supports strings, dict-like payloads, enum-like objects, and opaque engine
    objects without assuming any single cg observation schema.
    """
    if option is None:
        return ""
    if isinstance(option, str):
        return option
    parts = []
    for k in ("type", "action_type", "name", "label", "text", "description", "move", "card", "pokemon"):
        v = _safe_get(option, k, None)
        sv = _safe_str(v).strip()
        if sv:
            parts.append(sv)
    if parts:
        return " ".join(parts)
    if hasattr(option, "name"):
        sv = _safe_str(getattr(option, "name", "")).strip()
        if sv:
            return sv
    try:
        s = str(option)
        # Avoid useless default object reprs when possible.
        if s and " object at 0x" not in s:
            return s
    except Exception:
        pass
    try:
        return repr(option)
    except Exception:
        return ""


def _extract_damage(text):
    """Extract maximum numeric damage, handling suffixes like +, ×, x, -, ~."""
    s = _safe_str(text)
    # Word boundaries fail before punctuation in strings like 120+ or 30×.
    # This pattern captures 2-3 digit damage numbers even when followed by
    # common TCG suffixes, ranges, or punctuation.
    nums = re.findall(r"(?<!\d)(\d{2,3})(?=[+\-×x~]?\s|$|[^\w])", s)
    nums_int = [int(n) for n in nums if n]
    if not nums_int:
        nums_int = [int(n) for n in re.findall(r"\b(\d{2,3})\b", s)]
    return max(nums_int) if nums_int else 0


def _option_damage(option):
    return _extract_damage(_option_text(option))


def _find_num(obj, keys, depth=4):
    """Find the first numeric value under preferred keys with bounded recursion.

    The search prioritizes likely scalar fields and only descends into complex
    containers up to a small depth to avoid crawling the entire engine object.
    """
    if depth < 0 or obj is None:
        return None
    keyset = tuple(keys or [])
    if isinstance(obj, dict):
        for k in keyset:
            v = obj.get(k, None)
            if isinstance(v, (int, float)):
                return v
        # common aliases before broad recursion
        for k, v in obj.items():
            if k in keyset and isinstance(v, str):
                try:
                    return float(v)
                except Exception:
                    pass
        scanned = 0
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)) or hasattr(v, "__dict__"):
                r = _find_num(v, keyset, depth - 1)
                if r is not None:
                    return r
                scanned += 1
                if scanned >= 24:
                    break
    elif isinstance(obj, (list, tuple)):
        for v in obj[:24]:
            r = _find_num(v, keyset, depth - 1)
            if r is not None:
                return r
    else:
        for k in keyset:
            v = getattr(obj, k, None)
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, str):
                try:
                    return float(v)
                except Exception:
                    pass
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            return _find_num(d, keyset, depth - 1)
    return None


def _active_hp_low(obs):
    cur = _safe_get(obs, "current", None)
    hp = _safe_get(cur, "hp", None) if cur is not None else None
    if not isinstance(hp, (int, float)):
        hp = _find_num(obs, ["hp", "currentHp", "remainingHp"])
    if not isinstance(hp, (int, float)):
        return False
    mx = _safe_get(cur, "maxHp", None) if cur is not None else None
    if isinstance(mx, (int, float)) and mx > 0:
        return hp <= LOW_HP_FRAC * mx
    return hp < LOW_HP_ABS


def _opponent_active_hp(obs):
    v = _find_num(obs, ["opponentHp", "opponent_hp", "defendingHp", "oppHp"])
    return v if isinstance(v, (int, float)) else None


def _is_end(option):
    """Detect terminal/pass-like options."""
    t = _option_text(option).upper()
    return "END" in t or "PASS" in t


def _is_attack(option):
    return "ATTACK" in _option_text(option).upper()


def _is_ability(option):
    return "ABILITY" in _option_text(option).upper()


def _is_evolve(option):
    return "EVOLVE" in _option_text(option).upper()


def _is_attach(option):
    return "ATTACH" in _option_text(option).upper()


def _is_retreat(option):
    t = _option_text(option).upper()
    return "RETREAT" in t or "SWITCH" in t


def _is_trainer(option):
    t = _option_text(option).upper()
    return "SUPPORTER" in t or "ITEM" in t or "TRAINER" in t


def _is_probable_ko(option, obs):
    if not _is_attack(option):
        return False
    opp = _opponent_active_hp(obs)
    dmg = _option_damage(option)
    return (opp is not None) and (dmg > 0) and (dmg >= opp)


def _trainer_score(text):
    t = _safe_str(text).lower()
    return sum(w for k, w in TRAINER_WEIGHTS.items() if k in t)


def _ability_score(text):
    t = _safe_str(text).lower()
    return sum(w for k, w in ABILITY_WEIGHTS.items() if k in t)


def _attack_score(option, obs):
    return _option_damage(option) * DMG_WEIGHT


def _score_option(option, obs, flags=None):
    flags = DEFAULT_FLAGS if flags is None else {**DEFAULT_FLAGS, **flags}
    text = _option_text(option)
    T = text.upper()
    if flags["simple"]:
        if _is_probable_ko(option, obs):
            return 950
        for i, a in enumerate(["ABILITY", "EVOLVE", "ATTACH", "RETREAT", "ATTACK", "END"]):
            if a in T:
                return 1000 - i * 100
        return 150
    is_attack = "ATTACK" in T
    is_retreat = "RETREAT" in T or "SWITCH" in T
    is_trainer = "SUPPORTER" in T or "ITEM" in T or "TRAINER" in T
    if flags["ko"] and is_attack and _is_probable_ko(option, obs):
        score = 900 + _attack_score(option, obs)
    elif flags["retreat"] and is_retreat and _active_hp_low(obs):
        score = 800
    elif "ABILITY" in T:
        score = (700 if flags["ability"] else 230) + _ability_score(text)
    elif "EVOLVE" in T:
        score = 600
    elif "ATTACH" in T:
        score = 500
    elif is_trainer:
        score = 400 + (_trainer_score(text) if flags["trainer"] else 0)
    elif "PLAY" in T:
        score = 300
    elif is_attack:
        score = 200 + _attack_score(option, obs)
    elif _is_end(option):
        score = 0
    else:
        score = 150
    # Aggressive mode must be strong enough to matter, not decorative confetti.
    # It favours attacking tempo against aggro/evolution archetypes while preserving lethal KO priority.
    if flags.get("aggressive", False):
        if is_attack:
            score += 400
        if "ATTACH" in T:
            score += 30
    if is_retreat and score < 700:
        score = min(score, 60)
    return score


def _basic_setup_score(option):
    """Score a Basic candidate for active setup using HP plus type affinity."""
    text = _option_text(option)
    nums = [int(n) for n in re.findall(r"\d+", text)]
    hp_or_damage = max(nums) if nums else 0
    type_bonus = 50 if PRIMARY_TYPE and PRIMARY_TYPE in text else 0
    return hp_or_damage + type_bonus


def _bench_score(option, primary_type=PRIMARY_TYPE):
    """Score a setup-bench candidate by durability, attack efficiency, type, and ability.

    score = HP*0.4 + damage-per-energy*0.5 + type bonus + ability bonus.
    HP and damage are parsed separately so a plain "90 HP Ability" is not
    mistaken for a 90-damage attack. Humanity survives one more regex incident.
    """
    text = _option_text(option)
    hp_match = re.findall(r"(?<!\d)(\d{2,3})\s*HP\b", text, flags=re.IGNORECASE)
    hp = max([int(n) for n in hp_match], default=0)
    if hp <= 0:
        nums = [int(n) for n in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", text)]
        hp = max(nums) if nums else 0
    dmg_match = re.findall(r"(?:ATTACK|DAMAGE|DMG|DOES)\D{0,24}(\d{2,3})(?=[+\-×x~]?\s|$|[^\w])", text, flags=re.IGNORECASE)
    dmg = max([int(n) for n in dmg_match], default=0)
    cost_symbols = re.findall(r"\{[^}]+\}", text)
    cost = len(cost_symbols) if cost_symbols else 0
    efficiency = dmg / (cost + 1)
    type_bonus = 12 if primary_type and primary_type in text else 0
    ability_bonus = 18 if "ABILITY" in text.upper() else 0
    return hp * 0.4 + efficiency * 0.5 + type_bonus + ability_bonus


def _setup_active_index(options):
    best_i, best_v = 0, -1.0
    for i, o in enumerate(options):
        v = float(_basic_setup_score(o))
        if v > best_v:
            best_v, best_i = v, i
    log("DEBUG", "setup active selected", {"index": best_i, "score": best_v})
    return best_i


def _setup_bench_indices(options, max_count, min_count=0, primary_type=PRIMARY_TYPE):
    if not options:
        return []
    try:
        max_count = int(max_count)
    except Exception:
        max_count = 1
    try:
        min_count = int(min_count)
    except Exception:
        min_count = 0
    scored = [(_bench_score(o, primary_type), i, _option_text(o)[:80]) for i, o in enumerate(options)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    k = min(max(max_count, 0), len(options))
    if k < min_count:
        k = min(min_count, len(options))
    if k <= 0:
        k = min(1, len(options))
    chosen = [i for _, i, _ in scored[:k]]
    log("DEBUG", "setup bench selected", {"chosen": chosen, "top": scored[:min(3, len(scored))]})
    return chosen

def _choose_action(select, obs):
    """Choose legal option index/indices for the current selection object."""
    opts = _get_options(select)
    if not opts:
        log("WARNING", "empty option list", {"select_type": _select_type_name(select)})
        return []  # never index into an empty option list
    stype = _select_type_name(select)
    sctx = _safe_str(_safe_get(select, "context", "")).upper()
    if "YES_NO" in stype or "YESNO" in stype:
        for i, o in enumerate(opts):
            if _safe_str(_option_type(o)).strip().upper() == "YES" or _option_text(o).strip().upper() == "YES":
                log("INFO", "yes/no selected YES", {"index": i})
                return [i]
        log("INFO", "yes/no fallback", {"index": 0})
        return [0]
    if "SETUP_ACTIVE" in sctx or "SETUP_ACTIVE" in stype:
        return [_setup_active_index(opts)]
    if "SETUP_BENCH" in sctx or "SETUP_BENCH" in stype:
        maxc = _safe_get(select, "maxCount", 1) or 1
        minc = _safe_get(select, "minCount", 0) or 0
        return _setup_bench_indices(opts, maxc, minc, PRIMARY_TYPE)
    try:
        archetype = _detect_opponent_archetype(obs)
        flags = _adjust_strategy(archetype)
        scored = sorted(((_score_option(o, obs, flags=flags) - i * 0.001, i, _option_text(o)[:80])
                         for i, o in enumerate(opts)), reverse=True)
        minc = _safe_get(select, "minCount", 1) or 1
        log("DEBUG", "main action scores", {"archetype": archetype, "top": scored[:min(3, len(scored))]})
        if isinstance(minc, int) and minc > 1:
            chosen = sorted(idx for _, idx, _ in scored[:min(minc, len(opts))])
            log("INFO", "multi-select action chosen", {"chosen": chosen})
            return chosen
        chosen = scored[0][1]
        log("INFO", "action chosen", {"index": chosen, "score": scored[0][0], "text": scored[0][2]})
        return [chosen]
    except Exception as e:
        log("ERROR", "error in _choose_action", {"error": repr(e), "options": [_option_text(o)[:80] for o in opts[:5]]})
        for i, o in enumerate(opts):
            if not _is_end(o):
                return [i]
        return [0]


def agent(obs_dict):
    """Competition entrypoint.

    Returns the 60-card deck when no selection is requested, otherwise returns a
    list of legal option index/indices. It never raises on malformed observations;
    in worst case it falls back to an empty selection or first non-END option.
    """
    obs = obs_dict
    try:
        if to_observation_class is not None and isinstance(obs_dict, dict) and "select" in obs_dict:
            try:
                converted = to_observation_class(obs_dict)
                if converted is not None:
                    obs = converted
            except Exception as e:
                log("WARNING", "to_observation_class failed", repr(e))
                obs = obs_dict
        if obs is None or _get_select(obs) is None:
            log("DEBUG", "no select found; returning deck")
            return list(DECK)
        return _choose_action(_get_select(obs), obs)
    except Exception as e:
        log("ERROR", "agent crashed", {"error": repr(e), "obs_type": type(obs_dict).__name__})
        try:
            if obs_dict is None or _safe_get(obs_dict, "select", None) is None:
                return list(DECK)
        except Exception:
            pass
        return []


DECK = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1156, 1156, 1156, 1156, 1135, 1135, 1135, 1083, 1083, 1083, 1083, 1125, 1125, 1125, 1125, 1132, 1132, 1132, 1132, 1169, 1169, 1169, 1169, 184, 184, 961, 961, 431, 431, 635, 635, 523, 523, 317, 317, 317, 317, 246, 111, 111, 223, 875, 875, 813, 967, 5, 184, 184, 971, 1088]
