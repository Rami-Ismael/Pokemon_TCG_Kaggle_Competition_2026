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

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "benchmark_agents.py"


def _run_small_benchmark() -> dict:
    """Invoke scripts/benchmark_agents.py with a tiny game count and parse its JSON."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--games", "2"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"benchmark script failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    out_path = REPO / "reports" / "agent_benchmark.json"
    assert out_path.exists(), "benchmark did not write reports/agent_benchmark.json"
    return json.loads(out_path.read_text())


def test_agents_load_and_play():
    """Every agent participates and the win matrix is well-formed."""
    data = _run_small_benchmark()
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
    # file was persisted so future runs build on this run's history.
    glicko_path = REPO / "reports" / "glicko_ratings.json"
    assert glicko_path.exists(), "benchmark did not write reports/glicko_ratings.json"
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
    data = _run_small_benchmark()
    print("benchmark ran OK for:", data["agents"])
    print("overall win %:", data["overall_win_pct"])
