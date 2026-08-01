# Kiyotah — "Reinforcement Learning and MCTS Sample Code" (host's notebook)

The original Kaggle notebook is **one code cell of 689 lines**. Jupyter shows no line
numbers, so the walkthrough notebooks' "read lines N–M" instructions are un-followable
there.

- `reinforcement-learning-and-mcts-sample-code.ipynb` — host's original (single 689-line cell).
- `source_with_linenumbers.py` — that same cell exported **verbatim** as a plain `.py`
  file. Open it in any editor → the real line numbers are 1:1 with the notebook's cell,
  so "lines 389-442" is literally lines 389-442 here.

## Verified line map (walkthrough → numbered file)

| Walkthrough says            | File lines | What's there                                    |
|-----------------------------|------------|-------------------------------------------------|
| `03`: "lines 389-442"       | 389-442    | `create_node()` candidate enumeration + `mcts_agent()` start |
| `05`: "lines 196-241"       | 196-241    | `get_encoder_input()` — assembles the 24 words  |
| `05`: "lines 7-36"          | 7-36       | constants (`SEARCH_COUNT`, `card_count`, …)     |
| `05`: "lines 57-96"         | 57-96       | `DecoderLayer` + `MyModel` shell                |
| `05`: "lines 122-152"      | 122-152    | `SparseVector` (`add` / `add_pos` / `add_single` / `word_start`) |
| train loop loss/backprop    | 643-689    | batch build, mask, `loss_dec.sum()/BATCH_SIZE`  |

Regenerate anytime with:

```python
import json, pathlib
nb = pathlib.Path("reinforcement-learning-and-mcts-sample-code.ipynb").read_text()
cells = [c for c in json.loads(nb)["cells"] if c["cell_type"] == "code"]
src = "\n".join("".join(c["source"]) for c in cells)
pathlib.Path("source_with_linenumbers.py").write_text(src + "\n")
```
