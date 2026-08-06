import os
import sys

# Kaggle's harness loads this file via exec(code_object, env) (see
# kaggle_environments/agent.py's get_last_callable), not a normal script
# run or import -- exec()'d code has no `__file__` in its namespace unless
# the caller injects it, which Kaggle's harness doesn't. A bare
# `os.path.abspath(__file__)` crashes with NameError there even though it
# works fine under `python main.py` or `import main` locally (both of
# which DO set __file__ normally). Confirmed against a real failed
# submission's traceback: the extracted bundle always lives at
# /kaggle_simulations/agent/, so that's used directly instead of deriving
# the path from __file__ at all.
if os.path.exists("/kaggle_simulations/agent"):
    _BUNDLE_DIR = "/kaggle_simulations/agent"
else:
    try:
        _BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _BUNDLE_DIR = os.getcwd()

sys.path.insert(0, _BUNDLE_DIR)

# Load THIS bundle's agent_core.py under a unique module name instead of
# `import agent_core`: in the benchmark harness several bundles coexist in
# one process, and a bare import would silently reuse whichever bundle's
# `agent_core` hit sys.modules first -- this arm could end up running a
# different model than the one shipped beside it.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "il_agent_v2_core", os.path.join(_BUNDLE_DIR, "agent_core.py")
)
agent_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_core)

with open(os.path.join(_BUNDLE_DIR, "deck.csv")) as f:
    agent_core.my_deck = [
        int(x) for x in f.read().splitlines() if x.strip()
    ][:60]

agent = agent_core.agent

# Re-export the fallback-diagnostic hooks (like the s2 wrappers do) so
# fallback_diagnostic.py / benchmark_agents.py can read this arm's counters
# despite the unique-name agent_core load above.
for _hook in ("diag_snapshot", "diag_first", "diag_reset"):
    if hasattr(agent_core, _hook):
        globals()[_hook] = getattr(agent_core, _hook)
