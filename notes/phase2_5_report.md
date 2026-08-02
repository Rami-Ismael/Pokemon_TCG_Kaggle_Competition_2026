# Phase 2.5 — Where the Corpus Lives — CLOSED

Date: 2026-08-02.

## Step 1 measurement (the only step needed)

Ran the actual production `encode_observation()` pipeline (`scripts/measure_encoded_size.py`)
over the entire train day — all 4,554 episodes, 822,637 rows, not a sample:

| | value |
|---|---:|
| Raw JSON | 20.0 GiB/day |
| Encoded parquet (zstd) | **0.022 GiB/day** |
| Compression ratio | 892.7x |
| Rows/episode | 180.6 (matches the previously-documented 181.3 figure) |
| Extrapolated to the full 41+ day corpus | **~0.92 GiB total** |

Disk: 35 GiB free on the volume holding `data/` (`df -h`), vs. <1 GiB needed
for the entire published corpus once encoded.

## Verdict

**It fits locally, no remote storage needed.** Per the prompt's own Step 1
rule, this is the expected/most-likely outcome, and it isn't close — the
whole 41+ day, ~210k-episode corpus encodes to under 1 GiB, roughly 38x
smaller than free disk space even before deleting the raw JSON.

Step 2 (pick a remote backend) does not apply. This also closes out the
Hugging Face `datasets`-library streaming discussion from earlier in this
session: the ergonomics argument was real, but the workload it would have
served (a corpus too large for local disk) doesn't exist here. Skipped per
explicit direction.

## Not yet done (optional follow-up, not blocking)

- Actually encode train + eval to parquet and switch `train_il.py`/`ILDataset`
  to read from it instead of raw JSON. Not required by the gate (the
  question was "where does the corpus live," and the answer is "locally,
  as-is or encoded, either works at this size") but would speed up future
  training runs' cold-start and dataloading if it becomes a bottleneck.
- If more days of the corpus get downloaded later (Phase 3's data-quality-
  vs-volume question may want them), re-run `scripts/measure_encoded_size.py`
  on the larger pull to confirm the trend holds before assuming it does.
