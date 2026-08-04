"""Tests for the deterministic-future features (notes/feature_ablation_candidates.md).

Covers the pure arithmetic (energy matcher, damage model, KO turns), the
feature-column plumbing, encoder output shapes on real episodes, and that
the model consumes/ignores the extra tensors per its config -- including
the legacy-checkpoint path (no feature fields -> extras ignored).

Run directly: uv run python -m pytest tests/test_il_features.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data" / "external" / "cg-lib"))

import pytest  # noqa: E402
import torch  # noqa: E402

from pokemon_tcg import il_dataset as ds  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    GLOBAL_FEATURE_SPECS,
    N_EXTRA_GLOBAL,
    N_EXTRA_OPT,
    OPT_FEATURE_SPECS,
    _damage_vs,
    _energy_deficit,
    _ko_turns,
    _static_db,
    encode_observation,
    feature_columns,
    iter_decisions,
    resolve_split_dir,
)
from pokemon_tcg.il_model import PTCGILConfig, PTCGImitationPolicy  # noqa: E402

C, G, F, W, P, D, R, TR = 0, 1, 2, 3, 5, 7, 10, 11  # EnergyType shorthand


def test_energy_deficit_exact_and_colorless():
    assert _energy_deficit([F, F], [F, F]) == 0
    assert _energy_deficit([F, C], [F, W]) == 0      # colorless takes leftover
    assert _energy_deficit([F, C], [F]) == 1         # colorless unmet
    assert _energy_deficit([F, F], [W, W]) == 2      # wrong color
    assert _energy_deficit([], []) == 0
    assert _energy_deficit([C, C], [G, D]) == 0      # colorless takes anything


def test_energy_deficit_wildcards():
    assert _energy_deficit([F], [R]) == 0            # rainbow matches any color
    assert _energy_deficit([P], [TR]) == 0           # team rocket matches psychic
    assert _energy_deficit([D], [TR]) == 0           # ...and darkness
    assert _energy_deficit([F], [TR]) == 1           # ...but nothing else
    # colored requirement consumes same-type before wildcard, leaving the
    # wildcard for the colorless requirement
    assert _energy_deficit([F, C], [F, R]) == 0


def _find(pred, seq):
    for x in seq:
        if pred(x):
            return x
    return None


def test_damage_vs_weakness_and_resistance():
    card_db, attack_db = _static_db()
    assert card_db and attack_db, "static card DB must load via cg lib"
    atk = _find(lambda a: (a.damage or 0) >= 50, attack_db.values())
    # attacker whose energyType matches some defender's weakness
    attacker = _find(
        lambda c: c.attacks and c.hp > 0, card_db.values()
    )
    assert atk is not None and attacker is not None
    weak_def = _find(
        lambda c: c.weakness is not None and c.weakness == attacker.energyType,
        card_db.values(),
    )
    plain_def = _find(
        lambda c: c.hp > 0 and c.weakness != attacker.energyType
        and c.resistance != attacker.energyType,
        card_db.values(),
    )
    assert _damage_vs(atk, attacker, plain_def) == atk.damage
    if weak_def is not None:
        assert _damage_vs(atk, attacker, weak_def) == 2 * atk.damage
    res_def = _find(
        lambda c: c.resistance is not None and c.resistance == attacker.energyType
        and c.weakness != attacker.energyType,
        card_db.values(),
    )
    if res_def is not None:
        assert _damage_vs(atk, attacker, res_def) == max(atk.damage - 30, 0)


def test_ko_turns():
    assert _ko_turns(60, 30) == 2
    assert _ko_turns(70, 30) == 3     # ceil
    assert _ko_turns(60, 0) == ds._KO_TURNS_CAP   # can't attack yet
    assert _ko_turns(1000, 30) == ds._KO_TURNS_CAP  # capped
    assert _ko_turns(0, 30) == 0
    assert _ko_turns(None, 30) == 0


def test_feature_columns_order_and_unknown():
    cols = feature_columns(["ko_race"], GLOBAL_FEATURE_SPECS)
    assert cols == list(range(GLOBAL_FEATURE_SPECS["ko_race"]))
    off = GLOBAL_FEATURE_SPECS["ko_race"]
    cols2 = feature_columns(["prize_race"], GLOBAL_FEATURE_SPECS)
    assert cols2 == list(range(off, off + GLOBAL_FEATURE_SPECS["prize_race"]))
    # order of request does not change column identity
    assert set(feature_columns(["prize_race", "ko_race"], GLOBAL_FEATURE_SPECS)) == set(
        cols + cols2
    )
    with pytest.raises(KeyError):
        feature_columns(["nope"], GLOBAL_FEATURE_SPECS)
    assert sum(GLOBAL_FEATURE_SPECS.values()) == N_EXTRA_GLOBAL
    assert sum(OPT_FEATURE_SPECS.values()) == N_EXTRA_OPT


def _some_encoded_rows(n=200):
    rows = []
    for obs, label, exclude, _meta in iter_decisions(resolve_split_dir("train"), max_episodes=5):
        feats = encode_observation(obs, exclude=exclude)
        if feats is not None:
            rows.append(feats)
            if len(rows) >= n:
                break
    assert rows, "expected decisions in the first train episodes"
    return rows


def test_encoder_emits_extra_tensors_with_sane_ranges():
    rows = _some_encoded_rows()
    saw_nonzero_global = saw_nonzero_opt = False
    for feats in rows:
        eg, eo = feats["extra_global"], feats["extra_opt"]
        assert eg.shape == (N_EXTRA_GLOBAL,)
        assert eo.shape == (ds.MAX_OPTIONS, N_EXTRA_OPT)
        assert torch.isfinite(eg).all() and torch.isfinite(eo).all()
        assert (eg >= 0).all() and (eg <= 2.0).all()
        assert (eo >= 0).all() and (eo <= 2.0).all()
        saw_nonzero_global |= bool((eg != 0).any())
        saw_nonzero_opt |= bool((eo != 0).any())
    assert saw_nonzero_global, "global features never fired on real data"
    assert saw_nonzero_opt, "option features never fired on real data"


def test_attack_tactical_ko_flag_consistent_with_visible_hp():
    """Wherever attack_tactical fired, kos_now must equal dmg >= visible opp hp."""
    card_db, attack_db = _static_db()
    checked = 0
    for obs, label, exclude, _meta in iter_decisions(resolve_split_dir("train"), max_episodes=20):
        select = obs.get("select") or {}
        cur = obs.get("current") or {}
        me = cur["players"][cur["yourIndex"]]
        opp = cur["players"][1 - cur["yourIndex"]]
        opp_active = (opp.get("active") or [None])[0]
        feats = encode_observation(obs, exclude=exclude)
        if feats is None or opp_active is None:
            continue
        for i, o in enumerate(select.get("option", [])[: ds.MAX_OPTIONS]):
            if o.get("type") != int(ds._OT_ATTACK) or o.get("attackId") is None:
                continue
            row = feats["extra_opt"][i]
            dmg_norm, kos_now = float(row[0]), float(row[1])
            if dmg_norm == 0.0:
                continue  # variable-damage attack or unknown attacker: no claim
            expected = dmg_norm * ds._DMG_NORM >= (opp_active.get("hp") or 0) - 1e-6
            assert kos_now == (1.0 if expected else 0.0)
            checked += 1
    assert checked > 0, "no ATTACK options with damage found in sample"


def _tiny_config(**kw):
    return PTCGILConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, **kw)


def _batch_from_rows(rows, n=4):
    keys = [k for k in rows[0] if k != "n_real_options"]
    batch = {k: torch.stack([r[k] for r in rows[:n]]) for k in keys}
    batch["label"] = torch.zeros(min(n, len(rows)), dtype=torch.long)
    return batch


def test_model_consumes_and_ignores_extras_per_config():
    rows = _some_encoded_rows(8)
    batch = _batch_from_rows(rows)

    legacy = PTCGImitationPolicy(_tiny_config())          # no feature fields
    out = legacy(**batch)                                  # extras present, ignored
    assert out["logits"].shape[0] == batch["label"].shape[0]
    assert legacy.extra_global_proj is None and legacy.extra_opt_proj is None

    full = PTCGImitationPolicy(
        _tiny_config(
            global_features=list(GLOBAL_FEATURE_SPECS),
            opt_features=list(OPT_FEATURE_SPECS),
        )
    )
    out2 = full(**batch)
    assert out2["loss"] is not None and torch.isfinite(out2["loss"])
    # a model that consumes features must fail loudly without them
    stripped = {k: v for k, v in batch.items() if k not in ("extra_global", "extra_opt")}
    with pytest.raises(ValueError):
        full(**stripped)
    # changing an enabled feature column must change the logits (wiring live)
    full.eval()
    with torch.no_grad():
        base_logits = full(**batch)["logits"]
        perturbed = dict(batch)
        perturbed["extra_global"] = batch["extra_global"] + 0.5
        assert not torch.allclose(full(**perturbed)["logits"], base_logits)


def test_config_roundtrip(tmp_path):
    cfg = _tiny_config(global_features=["ko_race"], opt_features=["attack_tactical"])
    model = PTCGImitationPolicy(cfg)
    model.save_pretrained(tmp_path)
    loaded = PTCGImitationPolicy.from_pretrained(tmp_path)
    assert loaded.config.global_features == ["ko_race"]
    assert loaded.config.opt_features == ["attack_tactical"]
    assert loaded.extra_global_proj is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
