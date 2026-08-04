"""Why do distinct options collide? Split collisions into "same card" vs
"card reference did not resolve".

`_encode_options` gives every option a card identity via `_resolve_option_refs`
-> `_get_card`. When `_get_card` returns None (unknown area, empty
`select.deck`, out-of-range index) the slot's card id and card scalars stay
zero, so two options pointing at *different* cards encode identically. That is a
different failure from two options pointing at two copies of the same card,
which is a genuine tie.

Reports, per context and per AreaType: colliding option slots whose ref
resolved, vs failed to resolve, plus whether `select.deck` was populated.
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
    iter_episode_decisions,
)

sys.path.insert(0, str(config.PROJECT_ROOT / "data" / "external" / "cg-lib"))
from cg.api import AreaType, SelectContext, SelectData, State  # noqa: E402
from cg.utils import to_dataclass  # noqa: E402

OPT_KEYS = ["opt_type", "opt_attack", "opt_special", "opt_scalar",
            "opt_ref_card_id", "opt_ref_scalar", "extra_opt"]


def hub_decisions(split: str, max_episodes: int | None):
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


def sig(feats, i):
    parts = []
    for k in OPT_KEYS:
        v = feats[k]
        parts.append(tuple(v[i].flatten().tolist()) if v.dim() > 1 else (v[i].item(),))
    return tuple(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--max-episodes", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    area_name = {int(a): a.name for a in AreaType}
    ctx_name = {int(c): c.name for c in SelectContext}

    rows = Counter()
    collide_rows = Counter()
    # ctx -> "resolved" | "unresolved" | "no_ref": colliding option slots
    kind_by_ctx = defaultdict(Counter)
    # area -> resolved / unresolved counts over colliding slots
    area_kind = defaultdict(Counter)
    deck_present = Counter()   # DECK-area colliding slots, select.deck populated?
    # rows whose collision would VANISH if every ref resolved to its true card
    fixable_rows = Counter()

    for obs, label, exclude, _m in hub_decisions(args.split, args.max_episodes):
        feats = encode_observation(obs, exclude=exclude)
        if feats is None:
            continue
        ctx = int(feats["select_context"].item())
        n_real = int(feats["n_real_options"])
        rows[ctx] += 1
        if n_real < 2:
            continue
        groups: dict[tuple, list[int]] = {}
        for i in range(n_real):
            groups.setdefault(sig(feats, i), []).append(i)
        colliding = [g for g in groups.values() if len(g) > 1]
        if not colliding:
            continue
        collide_rows[ctx] += 1

        state: State = to_dataclass(obs["current"], State)
        select: SelectData = to_dataclass(obs["select"], SelectData)
        my_index = state.yourIndex
        opts = select.option[:n_real]
        has_deck = bool(select.deck)

        row_fixable = False
        for g in colliding:
            true_ids = []
            for i in g:
                o = opts[i]
                ref1, _ = _resolve_option_refs(o, my_index)
                if ref1 is None:
                    kind_by_ctx[ctx]["no_ref"] += 1
                    true_ids.append(None)
                    continue
                card = _get_card(state, select, *ref1)
                area = ref1[0]
                k = "resolved" if card is not None else "unresolved"
                kind_by_ctx[ctx][k] += 1
                area_kind[int(area) if area is not None else -1][k] += 1
                if area == AreaType.DECK:
                    deck_present["deck_populated" if has_deck else "deck_empty"] += 1
                true_ids.append(getattr(card, "id", None) if card is not None else "UNRESOLVED")
            # would resolving the refs separate this group?
            if len({t for t in true_ids}) > 1:
                row_fixable = True
        if row_fixable:
            fixable_rows[ctx] += 1

    total = sum(rows.values())
    print(f"\ntotal decision rows: {total}\n")
    hdr = (f"{'ctx':>4} {'name':<24} {'rows':>7} {'collide':>8} "
           f"{'resolved':>9} {'unresolv':>9} {'no_ref':>7} {'sepByCard':>10}")
    print(hdr)
    print("-" * len(hdr))
    for ctx in sorted(rows, key=lambda c: -collide_rows[c]):
        if not collide_rows[ctx]:
            continue
        k = kind_by_ctx[ctx]
        print(f"{ctx:>4} {ctx_name.get(ctx, '?'):<24} {rows[ctx]:>7} "
              f"{collide_rows[ctx]:>8} {k['resolved']:>9} {k['unresolved']:>9} "
              f"{k['no_ref']:>7} {fixable_rows[ctx]:>10}")

    print("\ncolliding option slots by AreaType of ref1:")
    for a in sorted(area_kind):
        k = area_kind[a]
        tot = k["resolved"] + k["unresolved"]
        print(f"  {area_name.get(a, a):<14} resolved {k['resolved']:>7}  "
              f"unresolved {k['unresolved']:>7}  ({100 * k['unresolved'] / max(tot, 1):.1f}% unresolved)")
    print(f"\nDECK-area colliding slots: {dict(deck_present)}")

    fx = sum(fixable_rows.values())
    print(f"\nrows whose collision would be separated by correct card resolution: "
          f"{fx} / {total} = {100 * fx / max(total, 1):.2f}% of all decisions")

    args.out.write_text(json.dumps({
        "total_rows": total,
        "per_ctx": {str(c): {"name": ctx_name.get(c, "?"), "rows": rows[c],
                             "collide_rows": collide_rows[c],
                             "kinds": dict(kind_by_ctx[c]),
                             "sep_by_card": fixable_rows[c]}
                    for c in sorted(rows)},
        "area_kind": {area_name.get(a, str(a)): dict(k) for a, k in area_kind.items()},
        "deck_present": dict(deck_present),
    }, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
