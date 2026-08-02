"""Test wrapper around the agent performance benchmark.

The full tournament is expensive (hundreds of engine games), so this test runs
a *small* benchmark and asserts the harness works end-to-end:
  * every configured agent loads and self-plays a full legal game,
  * the win matrix is produced with non-negative counts,
  * every agent has a recorded overall win rate.

Run the heavy tournament separately with:
    uv run python scripts/benchmark_agents.py --games 20

Requires no pytest: it can also be executed directly with `uv run python tests/test_agent_benchmark.py`.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "benchmark_agents.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("_test_benchmark_agents", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    spec.loader.exec_module(mod)
    return mod


class _FakeEnv:
    """Mimics kaggle_environments' env.run() return shape for play_match."""

    def __init__(self, reward_seat0: float, reward_seat1: float):
        self._rewards = (reward_seat0, reward_seat1)

    def run(self, agents):
        return [[{"reward": self._rewards[0], "status": "DONE"},
                  {"reward": self._rewards[1], "status": "DONE"}]]


def test_play_match_self_play_does_not_double_count_wins():
    """agent_a is agent_b (true self-play): seat 0 always wins these fake
    games, so exactly one of a_wins/b_wins should increment per game -- not
    both, as the old identity-based attribution did."""
    bench = _load_benchmark_module()
    same_agent = lambda obs: None  # noqa: E731 - identity is what matters here

    a_wins, b_wins, draws, _ = bench.play_match(
        same_agent, same_agent,
        env_factory=lambda: _FakeEnv(reward_seat0=1.0, reward_seat1=0.0),
        pairs=2,
    )
    assert draws == 0
    assert a_wins + b_wins == 4, f"expected 4 decisive games, got a_wins={a_wins} b_wins={b_wins}"


def _run_small_benchmark(out_dir: Path) -> tuple[dict, Path]:
    """Invoke scripts/benchmark_agents.py with a tiny game count and parse its JSON.

    Writes into `out_dir`, NOT into reports/. This test used to run with the
    script's defaults, which meant a `pytest` invocation silently overwrote the
    tracked reports/agent_benchmark.json and folded its own 2-game noise into
    the standing reports/glicko_ratings.json -- ratings that are supposed to
    compound over real tournament history. Worse, it raced any real benchmark
    running concurrently: both processes wrote the same two files, and whoever
    finished last won. Redirecting both outputs keeps the test hermetic.
    """
    out_path = out_dir / "agent_benchmark.json"
    glicko_path = out_dir / "glicko_ratings.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--games", "2",
         "--out", str(out_path), "--glicko-path", str(glicko_path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"benchmark script failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    assert out_path.exists(), f"benchmark did not write {out_path}"
    return json.loads(out_path.read_text()), glicko_path


def test_agents_load_and_play(tmp_path):
    """Every agent participates and the win matrix is well-formed."""
    data, glicko_path = _run_small_benchmark(tmp_path)
    agents = data["agents"]
    assert agents, "no agents benchmarked"

    # Counts are non-negative and games played per ordered pair match.
    for a in agents:
        for b in agents:
            if a == b:
                continue
            assert data["games"][a][b] >= 0
            assert data["wins"][a][b] >= 0
            assert data["wins"][a][b] <= data["games"][a][b]

    # Every agent got an overall win-rate entry.
    for a in agents:
        assert a in data["overall_win_pct"]
        assert 0.0 <= data["overall_win_pct"][a] <= 100.0

    # Every agent got a Glicko-1 rating + rating deviation, and the ratings
    # file was persisted so future runs build on this run's history. (Persisted
    # to the tmp path above, so the real reports/glicko_ratings.json history is
    # untouched -- the persistence *mechanism* is what's under test.)
    assert glicko_path.exists(), f"benchmark did not write {glicko_path}"
    persisted = json.loads(glicko_path.read_text())
    for a in agents:
        assert a in data["glicko"], f"{a} missing from glicko report"
        assert a in persisted, f"{a} missing from persisted glicko ratings"
        for src in (data["glicko"][a], persisted[a]):
            assert src["rd"] > 0
            assert 0 < src["rating"] < 4000  # sane bound, not a tight assertion

        # GXE is only in the report (not persisted glicko_ratings.json), and
        # is always a well-formed win-probability percentage.
        assert 0.0 <= data["glicko"][a]["gxe"] <= 100.0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as _tmp:
        data, _ = _run_small_benchmark(Path(_tmp))
    print("benchmark ran OK for:", data["agents"])
    print("overall win %:", data["overall_win_pct"])
