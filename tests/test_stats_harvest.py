"""The rollout-stats harvest must not drop samples when pufferl swaps its dict.

pufferl.train() does `self.stats = defaultdict(list)` every update, so each
evaluate() returns brand-new list objects. A harvest that tracks only a
consumed LENGTH across that boundary loses any key whose fresh list is no
longer than its stale count -- which is every SPARSE key, i.e. exactly the
per-matchup cells. Measured on g4_treatment_s42 before the fix: 3 opponent keys
captured ~100% of episodes, 29 deck keys 59%, 841 matchup keys 20%.

The rates stayed roughly unbiased (drops are by arrival order, not outcome) but
n was understated, which matters because n is what decides whether a cell is
readable at all.

This reimplements both harvests over a scripted stats stream and asserts the
fixed one is exact. `uv run python tests/test_stats_harvest.py` also works.
"""
from __future__ import annotations

from collections import defaultdict


def _buggy(stream):
    """Length-only bookkeeping: the original."""
    seen, n = defaultdict(int), defaultdict(int)
    for stats in stream:
        for k, v in stats.items():
            if len(v) < seen[k]:
                seen[k] = 0
            fresh, seen[k] = v[seen[k]:], len(v)
            n[k] += len(fresh)
    return dict(n)


def _fixed(stream):
    """Identity-keyed bookkeeping: hold the list so `is` is sound."""
    prev, consumed, n = {}, defaultdict(int), defaultdict(int)
    for stats in stream:
        for k, v in stats.items():
            fresh = v[consumed[k]:] if prev.get(k) is v else v[:]
            prev[k], consumed[k] = v, len(v)
            n[k] += len(fresh)
    return dict(n)


def _stream():
    """Ten update cycles. Every cycle pufferl hands back a NEW dict of NEW
    lists: a dense key with many samples, a sparse key with exactly one."""
    truth = defaultdict(int)
    out = []
    for _ in range(10):
        stats = {"wr_mirror": [1.0] * 40, "wrm_a~b": [1.0]}
        for k, v in stats.items():
            truth[k] += len(v)
        out.append(stats)
    return out, dict(truth)


def test_fixed_harvest_is_exact():
    stream, truth = _stream()
    assert _fixed(stream) == truth, "fixed harvest lost samples"


def test_buggy_harvest_drops_the_sparse_key():
    """Pinned so the regression is visible, not just fixed."""
    stream, truth = _stream()
    bad = _buggy(stream)
    assert bad["wrm_a~b"] < truth["wrm_a~b"], (
        "the length-only bug no longer reproduces; if the harvest changed, "
        "re-derive whether this guard is still needed")
    assert bad["wrm_a~b"] == 1, f"expected 1 of 10 sparse samples, got {bad}"


def test_delta_is_taken_when_the_same_list_grows():
    """Within one update the list object is reused and only grows -- taking
    everything again would double-count."""
    shared = {"wr_mirror": [1.0, 1.0]}
    n1 = _fixed([shared])
    shared["wr_mirror"].extend([1.0, 1.0])
    prev, consumed, n = {}, defaultdict(int), defaultdict(int)
    same = [1.0, 1.0]
    stats = {"wr_mirror": same}
    for _ in range(2):
        for k, v in stats.items():
            fresh = v[consumed[k]:] if prev.get(k) is v else v[:]
            prev[k], consumed[k] = v, len(v)
            n[k] += len(fresh)
        same.append(1.0)
    assert n1 == {"wr_mirror": 2}
    assert n["wr_mirror"] == 3, f"double-counted a growing list: {n}"


if __name__ == "__main__":
    test_fixed_harvest_is_exact(); print("fixed harvest exact: OK")
    test_buggy_harvest_drops_the_sparse_key(); print("regression pinned: OK")
    test_delta_is_taken_when_the_same_list_grows(); print("no double-count: OK")
