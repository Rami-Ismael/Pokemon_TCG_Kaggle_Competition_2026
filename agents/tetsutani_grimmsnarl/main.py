from __future__ import annotations

import collections
import gzip
import json
import math
import pickle
import struct
from pathlib import Path

import policy_features as pf
from strategic_policy import agent as strategic_agent
from experts.mirror.main import agent as mirror_expert
from experts.tempo.main import agent as tempo_expert
from coalition_expert import agent as coalition_expert
from matchup_router import choose as route_choice, reset as route_reset
from human_controller import choose as human_choose
from residual_guard import choose as residual_choose, reset as residual_reset
from advisor_guard import choose as advisor_choose
from tactical_guard import choose as tactical_choose, reset as tactical_reset
from development_guard import choose as development_choose, reset as development_reset
from robustness_guard import choose as robustness_choose, reset as robustness_reset
import human_memory as human_memory

# Kaggle executes submitted source with ``exec`` and may not define __file__.
# Resolve the packaged directory without assuming normal module import semantics.
if "__file__" in globals():
    BASE = Path(globals()["__file__"]).resolve().parent
else:
    _base_candidates = (Path("/kaggle_simulations/agent"), Path.cwd())
    BASE = next(
        (candidate for candidate in _base_candidates
         if (candidate / "models/policy_ensemble.bin.gz").is_file()),
        Path.cwd(),
    )
EXPECTED_DECK = [7,7,7,7,7,7,7,7,7,7,104,104,112,112,112,112,646,646,646,646,647,647,647,648,648,648,860,860,1079,1079,1079,1080,1086,1086,1086,1086,1097,1097,1097,1122,1137,1152,1152,1152,1152,1182,1182,1219,1219,1219,1219,1227,1227,1227,1227,1231,1259,1259,1259,1259]
DECK = list(pf.DECK)
if DECK != EXPECTED_DECK:
    raise RuntimeError("fixed 60-card deck changed")

with gzip.open(BASE / "models/feature_schema.pkl.gz", "rb") as f:
    SCHEMA = pickle.load(f)
INTENT_TO_ID = {
    str(k): int(v)
    for k, v in json.loads((BASE / "final_schema.json").read_text(encoding="utf-8"))["intent_to_id"].items()
}

_RECORD = struct.Struct("<hBBBee")

def _load_ensemble(path: Path):
    raw = gzip.decompress(path.read_bytes())
    if raw[:4] != b"PTC2":
        raise ValueError("Unsupported policy asset")
    pos = 4
    model_count = raw[pos]
    pos += 1
    models = {}
    for _ in range(model_count):
        name_len = raw[pos]
        pos += 1
        name = raw[pos : pos + name_len].decode("ascii")
        pos += name_len
        tree_count = struct.unpack_from("<H", raw, pos)[0]
        pos += 2
        trees = []
        for _ in range(tree_count):
            node_count = raw[pos]
            pos += 1
            nodes = []
            for _ in range(node_count):
                nodes.append(_RECORD.unpack_from(raw, pos))
                pos += _RECORD.size
            trees.append(nodes)
        models[name] = trees
    if pos != len(raw):
        raise ValueError("Policy asset has trailing bytes")
    return models

MODELS = _load_ensemble(BASE / "models/policy_ensemble.bin.gz")
_HISTORY = []


def _reset():
    global _HISTORY
    _HISTORY = []
    try: route_reset()
    except Exception: pass
    try: human_memory.reset()
    except Exception: pass
    try: residual_reset()
    except Exception: pass
    try: tactical_reset()
    except Exception: pass
    try: development_reset()
    except Exception: pass
    try: robustness_reset()
    except Exception: pass
    for fn in (mirror_expert, tempo_expert, coalition_expert):
        try: fn({})
        except Exception: pass


def _route(context: int):
    if context == 0:
        return "main"
    if context == 7:
        return "c7"
    if context in (3, 5, 8):
        return "low"
    if context in (13, 15, 16, 21, 22, 40, 43):
        return "mid"
    if context in (1, 2, 4, 27, 30, 34, 37, 38, 41):
        return "easy"
    return None


def _transform(row):
    maps = SCHEMA["category_maps"]
    values = []
    for name in SCHEMA["features"]:
        value = row.get(name, 0)
        if name in maps:
            try:
                value = value.item()
            except Exception:
                pass
            value = maps[name].get(value, -1)
        try:
            values.append(float(value))
        except Exception:
            values.append(float("nan"))
    return values


def _score(values, trees):
    total = 0.0
    for nodes in trees:
        node_index = 0
        while True:
            feature, flags, left, right, threshold, leaf_value = nodes[node_index]
            if flags & 1:
                total += leaf_value
                break
            value = values[feature]
            if math.isnan(value):
                node_index = left if flags & 2 else right
            else:
                node_index = left if value <= threshold else right
    return total


def _legal(action, select, option_count):
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", 0) or 0)
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(isinstance(index, int) and 0 <= index < option_count for index in action)
    )


def _model_action(obs):
    select = obs.get("select") or {}
    options = select.get("option") or []
    option_count = len(options)
    selected_count = min(option_count, int(select.get("maxCount", 0) or 0))
    if selected_count <= 0:
        return [], []

    context = int(select.get("context", -1) if select.get("context") is not None else -1)
    model_name = _route(context)
    if model_name is None:
        return None, [pf.semantic(obs, option) for option in options]

    state = pf.base_state(obs, _HISTORY)
    semantics = [pf.semantic(obs, option) for option in options]
    semantic_keys = [
        (item["type"], item["source_id"], item["target_id"], item["attack_id"], item["area"], item["inplay_area"])
        for item in semantics
    ]
    counts = collections.Counter(semantic_keys)
    seen = collections.Counter()
    previous = _HISTORY[-1] if _HISTORY else {}
    scores = []

    for position, (option, item, key) in enumerate(zip(options, semantics, semantic_keys)):
        duplicate_rank = seen[key]
        seen[key] += 1
        row, _ = pf.option_row(obs, state, option, position, item, counts[key], duplicate_rank)
        values = _transform(row)
        values.extend(
            [
                float(INTENT_TO_ID.get(pf.intent_text(item), -1)),
                float(previous.get("type", -1)),
                float(previous.get("source_id", 0)),
                float(previous.get("target_id", 0)),
                float(previous.get("attack_id", 0)),
                float(len(_HISTORY)),
            ]
        )
        scores.append(_score(values, MODELS[model_name]))

    order = sorted(range(option_count), key=lambda index: (-scores[index], index))
    return sorted(order[:selected_count]), semantics


def agent(obs):
    global _HISTORY
    if not obs or obs.get("select") is None:
        _reset()
        try:
            strategic_agent(obs)
        except Exception:
            pass
        return list(DECK)

    select = obs.get("select") or {}
    options = select.get("option") or []
    option_count = len(options)

    try:
        fallback = strategic_agent(obs)
    except Exception:
        fallback = []

    try:
        action, semantics = _model_action(obs)
    except Exception:
        action, semantics = None, [pf.semantic(obs, option) for option in options]

    chosen = fallback if action is None else action
    try:
        mirror_action = list(mirror_expert(obs))
    except Exception:
        mirror_action = []
    try:
        tempo_action = list(tempo_expert(obs))
    except Exception:
        tempo_action = []
    try:
        baseline_route = route_choice(obs, chosen, mirror_action, tempo_action)
    except Exception:
        baseline_route = chosen
    if baseline_route is None or not _legal(baseline_route, select, option_count):
        baseline_route = chosen
    try:
        memory_state = human_memory.update(obs)
    except Exception:
        memory_state = {}
    if memory_state.get("profile") == "grass_fast" and float(memory_state.get("confidence", 0.0)) >= 0.45:
        try:
            coalition_action = list(coalition_expert(obs))
        except Exception:
            coalition_action = []
    else:
        coalition_action = []
    try:
        routed = human_choose(obs, {
            "model": action if action is not None else [],
            "strategic": fallback,
            "mirror": mirror_action,
            "tempo": tempo_action,
            "coalition": coalition_action,
            "baseline_route": baseline_route,
        })
    except Exception:
        routed = None
    if routed is not None and _legal(routed, select, option_count):
        chosen = routed
    try:
        advised_action = advisor_choose(obs, chosen)
    except Exception:
        advised_action = None
    if advised_action is not None and _legal(advised_action, select, option_count):
        chosen = advised_action
    try:
        residual_action = residual_choose(obs, chosen)
    except Exception:
        residual_action = None
    if residual_action is not None and _legal(residual_action, select, option_count):
        chosen = residual_action
    try:
        tactical_action = tactical_choose(obs, chosen)
    except Exception:
        tactical_action = None
    if tactical_action is not None and _legal(tactical_action, select, option_count):
        chosen = tactical_action
    try:
        development_action = development_choose(obs, chosen)
    except Exception:
        development_action = None
    if development_action is not None and _legal(development_action, select, option_count):
        chosen = development_action
    try:
        robustness_action = robustness_choose(obs, chosen)
    except Exception:
        robustness_action = None
    if robustness_action is not None and _legal(robustness_action, select, option_count):
        chosen = robustness_action
    if not _legal(chosen, select, option_count):
        chosen = fallback
    if not _legal(chosen, select, option_count):
        chosen = list(range(min(option_count, int(select.get("maxCount", 0) or 0))))

    _HISTORY = (_HISTORY + [semantics[index] for index in chosen if 0 <= index < option_count])[-8:]
    return chosen
