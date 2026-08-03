# Fallback-site inventory (2026-08-03)

Every site in OUR code where a failure is silently converted into a legal-but-
model-free action (or a silent degradation). Compiled by sweeping for bare
`except`, `except Exception`, silent default returns, and default-swap
patterns before any counter was written; the tracker in
`agents/il_agent/agent_core.py` (`PTCG_FALLBACK_TRACK=1`) measures the
il_agent rows. Line numbers are post-instrumentation.

A fallback that fires silently is worse than a crash: the agent "runs" and
nothing distinguishes a weak model from a safety net answering 40% of
decisions. Rates, not counts: read every number against `decisions`.

## il_agent submission path (instrumented — reason key in parens)

| Site | Catches / condition | Returns / effect |
|---|---|---|
| `agents/il_agent/agent_core.py:66` | `except Exception` on torch/pokemon_tcg import | `_ML_AVAILABLE=False` → EVERY decision `_safe_choice` (`model_unavailable:no_ml_stack`) |
| `agents/il_agent/agent_core.py:71-74` | explicit `IL_MODEL_DIR` doesn't exist | silently resolves to `models/il_agent` — the WRONG checkpoint for a sweep (`model_dir_redirect` signal; defeated this session's first negative control) |
| `agents/il_agent/agent_core.py:189` | `except Exception` in `_load_model` | `_model=None` → every decision `_safe_choice` (`model_unavailable:load_failed`; traceback kept as `model_load_error`) |
| `agents/il_agent/agent_core.py:298` | `select.option` empty | `[]` (`no_legal_actions`) |
| `agents/il_agent/agent_core.py:301` | >48 options (`MAX_OPTIONS`) | `_safe_choice` (`too_many_options`) |
| `agents/il_agent/agent_core.py:246` | 5s decision deadline exceeded | `_safe_choice` (`step_timeout`) |
| `agents/il_agent/agent_core.py:252` | `encode_observation` returned None | `_safe_choice` (`encode_none`) |
| `agents/il_agent/agent_core.py:261` | model argmax out of range / already-picked | `_safe_choice` (`model_action_illegal` — structurally 0 when masking is right; CI-gated by `tests/test_fallback_tracking.py`) |
| `agents/il_agent/agent_core.py:276` | fewer picks than `minCount` | `_safe_choice` (`min_count_unmet`) |
| `agents/il_agent/agent_core.py:319` | outer `except Exception` in `agent()` | `_safe_choice` or `[]` (`policy_exception:<class>`, traceback kept; nested variant `policy_exception_nested:<class>`) |
| `src/pokemon_tcg/il_dataset.py:85` | `_clamp_id`: id outside trained vocab | silently mapped to shared OOV embedding row (`unknown_card` / `unknown_attack` signals — the Mega Lucario/Riolu zero-prevalence problem surfacing at inference) |
| `src/pokemon_tcg/il_model.py:139-145` | `.clamp()` defense-in-depth on all id tensors | same class as above, invisible past the encoder check |
| (nowhere) | NaN logits | argmax still yields an in-range index — garbage-but-legal choice (`nan_logits` signal; count-only, behavior unchanged) |

`s2_arms/*/agent_core.py` (9 wrappers): all of the above via a private module
instance; diag hooks re-exported per-arm. Note the `model_dir_redirect` row —
an arm checkpoint missing in a worktree silently benchmarks Stage-1 weights.

## Other agents of ours (own coarse `_DIAG` or none)

| Site | Catches | Returns |
|---|---|---|
| `agents/grunt/agent_core.py:261` | `except Exception` around whole policy | first-k legal indices; nested `:272` → `[]`. No counters. |
| `scripts/_proto_agent.py:458` | search error (printed to stdout) | `None` → heuristic |
| `scripts/_proto_agent.py:465` | obs parse failure | `[0]` — a guess, not even `_safe_choice`. No counters. |
| `scripts/_proto_agent.py:480` | `except Exception` in `agent()` | first-k legal. No counters. |
| `agents/mega_lucario/agent_core_improved.py:970/987/1013` | search/parse/heuristic failures | own `_DIAG` (4 reasons), harness-collected |
| `agents/improved_probabilistic/main.py:105` | deck.csv unreadable / wrong length | silently swaps to built-in `DECK` |
| `agents/improved_probabilistic/main.py:714/772/853/866/892` | engine-search / parse / heuristic failures | own `_DIAG`, harness-collected |
| `scripts/benchmark_agents.py:460` | crashed agent (`reward=None`) | scored as loss, `[WARN]` printed (visible — fine) |

Public opponents (kiyotah_*, tb_*, etc.) have their own fallback style; not
ours to instrument. `mechi22_alakazam` ships its own `_DIAG`.

## Standing rule

The tracker OBSERVES these sites. Never add a new try/except while extending
it — a counter that manufactures fallbacks to count defeats its purpose.
