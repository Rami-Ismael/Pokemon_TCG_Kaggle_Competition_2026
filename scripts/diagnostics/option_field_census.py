"""Per-SelectContext census of which raw `Option` fields the encoder drops --
and, crucially, which of them VARY between the options of a single decision.

A dropped field only matters if it differs across the legal options of the same
row: that is exactly the case where two distinct actions encode identically and
the policy can only tell them apart by slot index. This script measures, per
context:

  rows / multi-option rows / rows whose options collide bitwise under the
  current encoder, and for each dropped field (`energyIndex`, `count`,
  `toolIndex`, `serial`, `cardId`) how often it varies inside a colliding group.

No model is loaded -- this is a property of the encoder + corpus only, so it is
cheap and deterministic. Use it to bound headroom BEFORE paying for a training
arm (see notes/experiments/2026-08-04-energy-option-identity.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    _get_card,
    _resolve_option_refs,
    encode_observation,
    iter_decisions,
    iter_episode_decisions,
    resolve_split_dir,
)

sys.path.insert(0, str(config.PROJECT_ROOT / "data" / "external" / "cg-lib"))
from cg.api import SelectContext, SelectData, State  # noqa: E402
from cg.utils import to_dataclass  # noqa: E402

# Option fields _encode_options never reads (verified by grep, 2026-08-04).
DROPPED_FIELDS = ["energyIndex", "count", "toolIndex", "serial", "cardId"]

# Bitwise option-slot signature uses every per-option tensor the encoder emits.
OPT_KEYS = [
    "opt_type",
    "opt_attack",
    "opt_special",
    "opt_scalar",
    "opt_ref_card_id",
    "opt_ref_scalar",
    "extra_opt",
]


def hub_decisions(split: str, max_episodes: int | None):
    """Stream decisions from the Hub shards (deterministic; local dirs are stubs)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from pokemon_tcg.il_dataset import _resolve_hub_shards

    repo = config.HF_EPISODES_REPO
    n = 0
    for rel in _resolve_hub_shards(repo, split, days=None):
        pf = pq.ParquetFile(hf_hub_download(repo, rel, repo_type="dataset"))
        for rb in pf.iter_batches(batch_size=8, columns=["episode_json"]):
            for raw in rb.to_pydict()["episode_json"]:
                if max_episodes is not None and n >= max_episodes:
                    print(f"episodes consumed: {n}")
                    return
                n += 1
                yield from iter_episode_decisions(json.loads(raw), episode_id=n)
    print(f"episodes consumed: {n}")


def opt_signature(feats, i: int):
    parts = []
    for k in OPT_KEYS:
        v = feats[k]
        parts.append(tuple(v[i].flatten().tolist()) if v.dim() > 1 else (v[i].item(),))
    return tuple(parts)


def energy_type_of(state, select, option, my_index: int):
    """Resolve `option.energyIndex` against the target's attached `energies`.

    Returns (energy_type | None, n_attached | None). The attached-energy list is
    public board state on both sides, so this reads nothing private.
    """
    ei = option.energyIndex
    if ei is None:
        return None, None
    ref1, _ = _resolve_option_refs(option, my_index)
    if ref1 is None:
        return None, None
    card = _get_card(state, select, *ref1)
    ens = list(getattr(card, "energies", None) or [])
    if not ens:
        return None, 0
    return (ens[ei] if 0 <= ei < len(ens) else None), len(ens)


def _bootstrap_ci(per_ep_rows: Counter, per_ep_sep: Counter, n_boot: int = 2000):
    """95% CI on the separable-row fraction, resampling whole EPISODES.

    Rows inside one episode are correlated (same board, same two players), so a
    row-level binomial CI understates the interval -- cluster on the episode.
    """
    import random

    eps = list(per_ep_rows)
    if not eps:
        return 0.0, 0.0
    rng = random.Random(config.RANDOM_SEED)
    fracs = []
    for _ in range(n_boot):
        pick = [eps[rng.randrange(len(eps))] for _ in eps]
        num = sum(per_ep_sep[e] for e in pick)
        den = sum(per_ep_rows[e] for e in pick)
        fracs.append(num / den if den else 0.0)
    fracs.sort()
    return fracs[int(0.025 * n_boot)], fracs[int(0.975 * n_boot)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--max-episodes", type=int, default=200)
    ap.add_argument(
        "--data-source", choices=["hub", "local"], default="hub",
        help="local split folders may be pruned stubs (ADR-001) -- hub is the default",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model-dir", type=Path, default=None,
        help="if given, also report the model's top-1 on the separable rows -- "
             "turns the OPTIMISTIC ceiling into the ACTUAL one",
    )
    args = ap.parse_args()

    model = None
    if args.model_dir is not None:
        import torch

        from pokemon_tcg.device import resolve_device
        from pokemon_tcg.il_model import PTCGImitationPolicy

        device = resolve_device(None)
        model = PTCGImitationPolicy.from_pretrained(args.model_dir).to(device).eval()
        print(f"scoring separable rows with {args.model_dir} on {device}")

    if args.data_source == "hub":
        stream = hub_decisions(args.split, args.max_episodes)
    else:
        stream = iter_decisions(resolve_split_dir(args.split), args.max_episodes)

    # per-context accumulators
    rows = Counter()
    multi = Counter()
    collide_rows = Counter()          # >=1 bitwise-identical pair of real options
    full_collide = Counter()          # ALL real options identical
    n_colliding_opts = Counter()      # options sitting in a collision group
    varies = defaultdict(Counter)     # ctx -> field -> rows where it varies in-group
    etype_varies = Counter()          # resolved energy TYPE differs inside a group
    etype_same = Counter()            # resolves, but all the same type (genuine tie)
    etype_unresolved = Counter()
    label_separable = Counter()       # label's group differs SEMANTICALLY (type/count/card)
    label_arbitrary = Counter()       # label's group differs only by raw index/serial
    sep_model_correct = Counter()     # of those, ones the model ALREADY gets right
    sep_rows_by_ctx = defaultdict(list)
    examples: dict[int, list] = defaultdict(list)
    # per-episode (rows, semantically-separable rows) for an episode-cluster bootstrap
    per_ep_rows: Counter = Counter()
    per_ep_sep: Counter = Counter()

    for obs, label, exclude, _meta in stream:
        feats = encode_observation(obs, exclude=exclude)
        if feats is None:
            continue
        ctx = int(feats["select_context"].item())
        n_real = int(feats["n_real_options"])
        rows[ctx] += 1
        per_ep_rows[_meta.episode_id] += 1
        if n_real < 2:
            continue
        multi[ctx] += 1

        groups: dict[tuple, list[int]] = {}
        for i in range(n_real):
            groups.setdefault(opt_signature(feats, i), []).append(i)
        colliding = [idxs for idxs in groups.values() if len(idxs) > 1]
        if not colliding:
            continue
        collide_rows[ctx] += 1
        n_colliding_opts[ctx] += sum(len(g) for g in colliding)
        if len(groups) == 1:
            full_collide[ctx] += 1

        state: State = to_dataclass(obs["current"], State)
        select: SelectData = to_dataclass(obs["select"], SelectData)
        my_index = state.yourIndex
        opts = select.option[:n_real]

        row_varies = set()
        kinds = set()
        label_sep_semantic = False
        label_sep_arbitrary = False
        for g in colliding:
            sep_by_field = False
            for f in DROPPED_FIELDS:
                if len({getattr(opts[i], f, None) for i in g}) > 1:
                    row_varies.add(f)
                    sep_by_field = True
            ets = [energy_type_of(state, select, opts[i], my_index)[0] for i in g]
            if any(e is None for e in ets):
                kind = "unresolved"
            elif len(set(ets)) > 1:
                kind = "differ"
            else:
                kind = "same"
            kinds.add(kind)
            if label < n_real and label in g:
                # SEMANTIC: the game state actually differs between these options
                # -- resolved energy type, how many are moved, or which card.
                semantic = (
                    kind == "differ"
                    or len({getattr(opts[i], "count", None) for i in g}) > 1
                    or len({getattr(opts[i], "cardId", None) for i in g}) > 1
                )
                label_sep_semantic = label_sep_semantic or semantic
                # ARBITRARY: only a raw slot number / physical-card serial differs,
                # so the recorded label is one of several interchangeable indices
                # and no feature can predict WHICH index was logged.
                label_sep_arbitrary = label_sep_arbitrary or (
                    sep_by_field and not semantic
                )

        for f in row_varies:
            varies[ctx][f] += 1
        # one label per row: a real difference anywhere beats a tie beats unresolved
        if "differ" in kinds:
            etype_varies[ctx] += 1
        elif "same" in kinds:
            etype_same[ctx] += 1
        else:
            etype_unresolved[ctx] += 1
        if label_sep_semantic:
            label_separable[ctx] += 1
            sep_rows_by_ctx[ctx].append(_meta.episode_id)
            per_ep_sep[_meta.episode_id] += 1
            if model is not None:
                import torch

                batch = {k: v.unsqueeze(0).to(model.device)
                         for k, v in feats.items() if k != "n_real_options"}
                with torch.no_grad():
                    top1 = int(model(**batch)["logits"][0].argmax())
                if top1 == label:
                    sep_model_correct[ctx] += 1
        elif label_sep_arbitrary:
            label_arbitrary[ctx] += 1

        if len(examples[ctx]) < 3:
            examples[ctx].append(
                {
                    "label": label,
                    "n_real": n_real,
                    "groups": [list(g) for g in colliding],
                    "options": [
                        {k: getattr(o, k, None) for k in
                         ["type", "area", "index", "playerIndex", *DROPPED_FIELDS,
                          "inPlayArea", "inPlayIndex"]}
                        for o in opts
                    ],
                }
            )

    total_rows = sum(rows.values())
    name = {int(m): m.name for m in SelectContext}
    print(f"\ntotal decision rows: {total_rows}\n")
    hdr = (f"{'ctx':>4} {'name':<24} {'rows':>7} {'%all':>6} {'multi':>7} "
           f"{'collide':>8} {'full':>6} {'etypeDIFF':>10} {'etypeSAME':>10} "
           f"{'sepSEM':>7} {'sepARB':>7}")
    print(hdr)
    print("-" * len(hdr))
    for ctx in sorted(rows, key=lambda c: -collide_rows[c]):
        if not collide_rows[ctx]:
            continue
        print(f"{ctx:>4} {name.get(ctx, '?'):<24} {rows[ctx]:>7} "
              f"{100 * rows[ctx] / total_rows:>5.2f}% {multi[ctx]:>7} "
              f"{collide_rows[ctx]:>8} {full_collide[ctx]:>6} "
              f"{etype_varies[ctx]:>10} {etype_same[ctx]:>10} "
              f"{label_separable[ctx]:>7} {label_arbitrary[ctx]:>7}")

    print("\ndropped field varies INSIDE a bitwise-identical group (rows):")
    fhdr = f"{'ctx':>4} {'name':<24} " + " ".join(f"{f:>12}" for f in DROPPED_FIELDS)
    print(fhdr)
    print("-" * len(fhdr))
    for ctx in sorted(rows, key=lambda c: -collide_rows[c]):
        if not collide_rows[ctx]:
            continue
        print(f"{ctx:>4} {name.get(ctx, '?'):<24} "
              + " ".join(f"{varies[ctx][f]:>12}" for f in DROPPED_FIELDS))

    sep = sum(label_separable.values())
    arb = sum(label_arbitrary.values())
    lo, hi = _bootstrap_ci(per_ep_rows, per_ep_sep)
    print(f"\nSEMANTIC ceiling -- label sits in a collision group whose members "
          f"differ in energy type / count / cardId:")
    print(f"  {sep} / {total_rows} = {100 * sep / max(total_rows, 1):.3f}% of all "
          f"decisions   95% CI [{100 * lo:.3f}%, {100 * hi:.3f}%] "
          f"(episode-cluster bootstrap, {len(per_ep_rows)} episodes)")
    print("  ^ OPTIMISTIC ceiling on top-1 gain: assumes the feature fixes every "
          "such row\n    AND that the model currently gets all of them wrong.")
    print(f"\nARBITRARY (unlearnable) -- label's group differs only by raw "
          f"energyIndex / serial:")
    print(f"  {arb} / {total_rows} = {100 * arb / max(total_rows, 1):.3f}% of all "
          f"decisions -- no feature can predict which interchangeable index was logged")

    if model is not None:
        ok = sum(sep_model_correct.values())
        wrong = sep - ok
        print(f"\nACTUAL ceiling -- of the {sep} semantically-separable rows the "
              f"model already gets {ok} right ({100 * ok / max(sep, 1):.1f}%).")
        print(f"  Only the remaining {wrong} could improve: "
              f"{100 * wrong / max(total_rows, 1):.3f}% of all decisions.")
        print(f"\n{'ctx':>4} {'name':<24} {'sepSEM':>7} {'modelOK':>8} {'wrong':>6} "
              f"{'ctxRows':>8} {'sep/ctx':>8}")
        for ctx in sorted(label_separable, key=lambda c: -label_separable[c]):
            print(f"{ctx:>4} {name.get(ctx, '?'):<24} {label_separable[ctx]:>7} "
                  f"{sep_model_correct[ctx]:>8} "
                  f"{label_separable[ctx] - sep_model_correct[ctx]:>6} "
                  f"{rows[ctx]:>8} "
                  f"{100 * label_separable[ctx] / max(rows[ctx], 1):>7.1f}%")

    args.out.write_text(json.dumps(
        {
            "total_rows": total_rows,
            "ceiling_semantic": {"rows": sep, "frac": sep / max(total_rows, 1),
                                 "ci95": [lo, hi], "n_episodes": len(per_ep_rows)},
            "ceiling_arbitrary": {"rows": arb, "frac": arb / max(total_rows, 1)},
            "per_ctx": {
                str(c): {
                    "name": name.get(c, "?"),
                    "rows": rows[c], "multi": multi[c],
                    "collide_rows": collide_rows[c], "full_collide": full_collide[c],
                    "n_colliding_opts": n_colliding_opts[c],
                    "etype_differ": etype_varies[c], "etype_same": etype_same[c],
                    "etype_unresolved": etype_unresolved[c],
                    "label_sep_semantic": label_separable[c],
                    "label_sep_arbitrary": label_arbitrary[c],
                    "sep_model_correct": sep_model_correct[c],
                    "field_varies": dict(varies[c]),
                }
                for c in sorted(rows)
            },
            "examples": {str(c): v for c, v in examples.items()},
        },
        indent=1, default=str,
    ))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
