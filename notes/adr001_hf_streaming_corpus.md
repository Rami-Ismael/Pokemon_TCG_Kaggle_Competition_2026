# ADR-001: Host the behavior-cloning episode corpus on Hugging Face Hub and stream it during training

**Status:** Accepted (implemented 2026-08-03)
**Date:** 2026-08-03
**Deciders:** Rami Ismael

## Context

Stage 1 of the Metamon-style plan (behavior cloning, per *Human-Level Competitive
Pokémon via Scalable Offline Reinforcement Learning with Transformers*) trains on
downloaded Kaggle leaderboard episodes. The full episode archive is ~800 GB of raw
JSON. The training machine is a laptop with 512 GB total disk and ~16 GB free;
`data/episodes/` currently holds ~3 days of episodes = **60 GB across 14,254 files**
(~4.2 MB per episode). The full corpus cannot live on this machine in raw form,
and it keeps growing as the competition plays new episodes.

The proposed solution is to upload the corpus to Hugging Face Hub and stream it
during training with `datasets.IterableDataset`. This ADR evaluates that plan and
pins down the details that make it cheap and safe instead of slow and expensive.

### Measured facts (verified 2026-08-03)

**Compression.** One representative episode (`splits/train-2026-07-01/83174254.json`,
5,103,431 bytes) compresses to **105,794 bytes at zstd-3 (48×)** and **51,821 bytes
at zstd-19 (98×)**. Kaggle replay JSON is mostly repeated state structure, which is
why the ratio is extreme. At 50–80× effective ratio inside Parquet:

> **800 GB raw ≈ 10–16 GB compressed.**

This one fact reframes the whole problem. The corpus is not "800 GB that needs a
big cloud plan" — it is ~15 GB that needs a sane container format.

**Network** (measured 2026-08-03 with `networkQuality -s`): uplink **78.9 Mbps**
(~9.9 MB/s), downlink **252.6 Mbps** (~31.6 MB/s), idle latency 55 ms. Concretely:
the full compressed corpus (~16 GB) uploads in ~27 min of saturated uplink (spread
over the rolling ingest anyway), and a compressed epoch streams in ~8 min if
network-bound — training is MPS-bound, so the actual draw is far lower. Raw form
for comparison: ~22 h of saturated uplink to upload 800 GB once, ~7 h of downlink
per epoch to stream it. Uplink responsiveness measured "Low" (~400 ms under load):
saturating uploads will make interactive use laggy while they run, which is fine
for unattended ingest but worth knowing.

**Hugging Face limits and pricing** (fetched from `huggingface.co/docs/hub/storage-limits`
and `/pricing`, 2026-08-03):

| Plan | Private storage | Public storage | Price |
|---|---|---|---|
| Free | **100 GB** | "best-effort", expects community-useful content | $0 |
| PRO | **1 TB** included, then $18/TB/mo | up to 10 TB included | **$9/mo** |
| Team | 1 TB/seat | 12 TB base | $20/user/mo |
| Public storage add-on | — | $12/TB/mo | paid plans only |

Repo hygiene rules: keep **<100k files per repo**, **<10k entries per folder**,
files ideally well under 200 GB (hard cap 500 GB/file); Parquet or WebDataset are
the recommended formats for large datasets; HF explicitly asks that large *public*
datasets exist for community reuse.

**Streaming API** (from `huggingface.co/docs/datasets/stream`): `load_dataset(...,
streaming=True)` yields an `IterableDataset` that never downloads the full set to
disk; `.shuffle(buffer_size=N)` shuffles both the shard order and a rolling example
buffer; `.set_epoch(e)` reseeds per epoch; a `DataLoader` with `num_workers=w`
splits the dataset's `num_shards` across workers (so shard count must be ≥ worker
count); `state_dict()/load_state_dict()` (and torchdata's `StatefulDataLoader`)
give mid-epoch checkpoint resume. Local files stream through the identical API.

**Current loader.** [il_dataset.py](../src/pokemon_tcg/il_dataset.py) is already a
torch `IterableDataset` that reads raw episode JSONs from per-day split folders,
extracts decisions on the fly (`iter_decisions` → `encode_observation`), and
shuffles through a small buffer. Only the *file source* needs to change; the
decision-extraction and encoding logic — which is still evolving (vocab sizes,
features, DECLINE handling) — stays at training time.

### Constraints

- ~16 GB free local disk; no room for new raw corpora (this also forced Stage 3 to on-policy PPO).
- Corpus grows daily; whatever we choose must absorb growth without re-architecture.
- Training runs **exclusively on the laptop** (MPS) — Kaggle notebooks and cloud
  GPUs are not part of the plan. Data loading must not starve MPS, but its
  throughput is modest.
- Episodes are Kaggle competition replays (they contain other competitors' games).
  Redistributing them publicly is both against the spirit of HF's public-storage
  policy and unwise competitively. **The repo must be private.**
- Budget: user is willing to pay for any HF plan — but should not pay for more than needed.
- The corpus is **competition-scoped**: it exists to train Stages 1–3 and will be
  deleted once the competition is over. Long-term archival and portability to other
  training environments are explicitly *not* goals; storage only needs to be cheap
  for the competition's remaining months, and teardown should be planned from the
  start. Losslessness still matters *during* the competition — once local raw files
  are pruned, the Hub copy is the only training copy.

## Decision

Adopt the Hugging Face plan, with three amendments that change its cost by ~50×:

1. **Upload compressed episode shards, never raw JSON files.** Pack episodes into
   zstd-compressed Parquet shards (one row = one episode: `episode_id`, `day`,
   `agents`, `episode_json` string holding the exact file bytes — packing is
   lossless and reversible), ~250–500 MB per shard, laid out per split day
   (`train/day=2026-07-01/shard-000.parquet`, `eval/…`). Target ≥64 shards so
   `DataLoader` workers parallelize. This keeps the repo at ~15 GB / dozens of
   files instead of 800 GB / ~190k files (which would violate the 100k-file and
   10k-per-folder limits outright).
2. **Private dataset repo** (`<user>/ptcg-episodes`), which fits in the **free
   100 GB private tier** with ample headroom for the rest of the competition.
   PRO ($9/mo) is an optional
   comfort upgrade (1 TB private, private-repo dataset viewer) — not a requirement.
   No public add-on, no Team plan.
3. **Rolling local buffer.** Because raw episodes only fit locally for ~3 days at a
   time: download day → pack to Parquet → `upload_large_folder` (resumable) →
   verify → delete raw JSONs. Keep the compressed shards cached locally while disk
   allows (they're small); the Hub copy is the source of truth and backup.

Training streams with `load_dataset(..., streaming=True)` (or from the local shard
cache with identical code), flat-mapping episode rows through the existing
`iter_decisions`/`encode_observation` path.

## Options Considered

### Option A (chosen): Compress episodes ~80×, store them in a private Hugging Face repo, stream them into training

Today every episode is its own `.json` file on the laptop, and the training
script opens those files one by one — which is why the whole corpus has to fit
on local disk, and 800 GB cannot. Option A changes where the bytes live, not
how training works:

1. Each day's episodes are bundled into a couple of compressed Parquet files.
   Measured on a real day: 21.5 GB of JSONs became 267 MB (80.6×), and the
   original files are reconstructable byte-for-byte, so nothing is lost.
2. Those files are uploaded to a Hugging Face dataset repository only this
   account can see — effectively a free 100 GB private cloud drive that the
   `datasets` library can read from directly.
3. The bulky raw `.json` files are then deleted from the laptop — only after
   the verify step proves the packed copy is identical.
4. At training time the loader iterates the repo with
   `load_dataset(..., streaming=True)`: it downloads a few MB of compressed
   data at a time, decompresses it in memory, and hands each episode to the
   existing decision-extraction/encoding code — like streaming a video
   instead of downloading the whole movie first. The model still just
   receives batches of tensors; the laptop never holds more than a few MB of
   the corpus at once. Keeping the compressed files locally too (~10 GB for
   everything) skips the network entirely.

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — one packing script, one loader change |
| Cost | **$0/mo** (free tier), $9/mo PRO optional |
| Scalability | Excellent — corpus can grow ~5× before even PRO matters |
| Team familiarity | High — `datasets` already used; loader is already an IterableDataset |

**Pros:** off-laptop durability/backup; corpus portable to any future cloud GPU box;
per-epoch network cost is the *compressed* size (~15 GB/epoch, minutes on broadband);
checkpointable streaming; dataset viewer (with PRO) for debugging.
**Cons:** training gains a network dependency (mitigated by local shard cache and
resume); packing step adds a moving part; private repo needs `HF_TOKEN` in the env.

### Option B: HF Hub with raw episode JSONs (the plan as originally stated)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low to set up, painful to live with |
| Cost | $9/mo PRO (800 GB < 1 TB private) — affordable, but see cons |
| Scalability | Fails on file-count limits before storage limits |
| Team familiarity | High |

**Pros:** no packing step; zero format decisions.
**Cons:** ~190k files exceeds the 100k/repo recommendation and the 10k-entries-per-folder
cap, so it needs restructuring anyway; initial upload of 800 GB ≈ **~22 h** of
saturated uplink at the measured 79 Mbps (vs ~27 min compressed); and the killer —
**streaming bandwidth scales with stored bytes**: every epoch would pull 800 GB
(~7 h at the measured 253 Mbps downlink, and 40 epochs ≈ 32 TB of traffic — past
any ISP data cap). Compression divides all of that by ~50.

### Option C: External SSD (2 TB, ~$100–180 one-time)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Lowest — loader unchanged |
| Cost | ~$150 one-time |
| Scalability | Fine to ~2 TB raw, then stuck again |
| Team familiarity | High |

**Pros:** fastest reads; no network dependency; no subscriptions.
**Cons:** single copy, no backup — a mid-competition drive failure costs weeks of
rate-limited Kaggle re-download that the training timeline can't absorb; still
smart to compress anyway — and once compressed, the corpus fits on the *internal*
disk, making the SSD redundant. Also a worse fit for a disposable corpus: hardware
outlives the need. Reasonable as a supplement, not the architecture.

### Option D: Kaggle Datasets + train on Kaggle notebook GPUs

**Pros:** free storage next to the competition; free T4/P100 hours; datasets mount
as local disk (no streaming needed).
**Cons:** 12 h session limit and ephemeral environments break long BC runs and the
existing local MPS pipeline; dataset size/versioning limits are tighter and less
transparent than HF; keeps data on the platform competitors watch. Moot regardless:
training happens exclusively on the laptop (user decision), so this option's main
draw — free Kaggle GPU hours — would go unused. Rejected.

### Option E: Local-only compressed shards (no cloud at all)

The compression insight alone dissolves the stated blocker: ~15 GB of zstd Parquet
fits on the laptop. **Pros:** $0, no network. **Cons:** no backup (a mid-competition
disk failure costs weeks of rate-limited Kaggle re-download), and no growth headroom
on a 16 GB-free disk. Rejected as the primary store, but Option A includes it as the
local cache layer for free.

## Trade-off Analysis

- **The format decision dominates the vendor decision.** Compressed-vs-raw changes
  storage 50×, upload time 50×, and per-epoch bandwidth 50×. Any option is bad with
  raw JSON; nearly every option is fine with compressed shards. Hence the amendments.
- **Private-free vs PRO:** the corpus (~15 GB now, ~1 GB compressed per 3 days of
  growth) stays under 100 GB through the end of the competition, and the repo is
  deleted afterward — worst-case total cost is **$0**, or a few months × $9 if PRO
  comforts are wanted. Pay $9/mo only
  if you want the private dataset viewer or cross 80 GB. The user's willingness to
  pay "any plan" buys nothing beyond PRO — Team/Enterprise and storage add-ons
  solve problems this corpus does not have.
- **Episode-level rows vs pre-encoded decision rows:** storing raw episode JSON per
  row preserves total reprocessing freedom while the encoder still evolves (the
  Metamon lesson: they re-parse replays every time the obs space changes). CPU cost
  of `json.loads` + encoding at train time is parallelized across DataLoader
  workers; if it ever bottlenecks MPS, emit a pre-encoded decision-level Parquet as
  a derived artifact (v2) — cheap because the source of truth is complete.
- **Streaming vs cache:** with the corpus this small, Hub streaming is the
  durability/growth story; day-to-day training will usually hit the local shard
  cache at SSD speed. The code path is identical either way. With training pinned
  to the laptop, the Hub's role is durability plus overflow for whatever exceeds
  the ~16 GB of free disk — it never feeds a remote machine.

## Consequences

- The corpus becomes durable and laptop-independent **for the duration of the
  competition** — a laptop or disk failure mid-competition no longer loses the
  training data. It is competition-scoped by design: when the competition ends, the
  HF repo and local cache are deleted and all storage obligations end. Until then,
  the only deletions are local raw files whose bytes are verified inside uploaded
  shards (raw JSONs stay recoverable byte-for-byte from the shards if ever needed).
- Raw episodes stop accumulating locally; disk pressure drops from "blocking" to
  "irrelevant" (~15 GB cache, prunable).
- Training gains `HF_TOKEN` + network as soft dependencies; mid-epoch resume via
  `StatefulDataLoader` and `datasets` retries must be wired in.
- The held-out-day split discipline (splits.json's "distinct days, never shuffle
  across them") must be preserved in the repo layout — per-day files under separate
  `train/` and `eval/` prefixes, loaded as separate splits, exactly as today.
- Watch-item: `shuffle()` quality now depends on shard order + buffer sizes
  (episode-level buffer ~100, decision-level buffer ~10k) instead of a global file
  sort; verify loss curves match a local-run baseline on the 3-day subset.
- Watch-item: deleting local raw JSONs after upload is irreversible — the pipeline
  must verify the uploaded shard (row count + checksum) before deletion.

## Action Items

1. [x] `scripts/pack_episodes.py` (done 2026-08-03): pack/unpack/verify implemented.
   Measured on train-2026-07-01: 5,266 episodes, 21.47 GB → 2 shards, 266.5 MB
   (**80.6×**) in 44 s (487 MB/s); verify passed — all rows hash-checked, id set
   exact, 50-row byte-identical sample; unpack reconstructs originals bit-for-bit.
2. [x] Private repo `Rami/ptcg-episodes` (done 2026-08-03): all three days
   uploaded and Hub-verified (every row's sha256 re-checked by streaming the
   shards back down, id sets exact, byte-identical samples) — 07-01 (2 shards,
   266.5 MB), 07-26 (4 shards, 253.0 MB, 84.9×), 07-27 (packed+uploaded the
   same day). ⚠️ one upload ran at only ~3.5 Mbps effective (579 s / 253 MB)
   vs the 57 Mbps measured uplink — single sample, confounded by concurrent
   traffic; watch, don't panic, uploads are ~daily 250 MB either way.
3. [x] Rolling ingest loop (done 2026-08-03): `scripts/ingest_episodes.py` —
   `status` / `run` (pack → local verify → upload → **hub verify**) /
   `download` / `backfill-manifest` / `verify-hub`. The download stage uses the
   official kaggle CLI's episode support (`team-submissions` → `episodes` →
   `replay`), verified to return files **byte-identical** to the existing
   archive (episode 88453216). Raw-folder deletion is deliberately NOT
   automated: the script prints what became safe to delete after hub-verify
   and stops there (concurrent sessions read `data/episodes/`).
   Per-episode rating fields for NEW days are approximated from each
   submission's current `publicScore` (the CLI does not expose the historical
   per-episode ratings the original manifest had); blank means unknown.
4. [x] Streaming source (done 2026-08-03): `ShardILDataset` in
   [il_dataset.py](../src/pokemon_tcg/il_dataset.py) — reads episode rows from
   the Hub (per-shard `hf_hub_download` into the standard HF cache: first pass
   pays network, later passes read at SSD speed) or a local shard dir, shares
   `iter_episode_decisions`/`encode_observation` with the raw-JSON path,
   splits shards across DataLoader workers, reshuffles per `set_epoch`.
   [train_il.py](../scripts/train_il.py) gained `--data-source {auto,local,hub}`,
   `--hub-days`, `--num-workers`; hub schedule length comes from parquet
   footers. `StatefulDataLoader` mid-epoch resume: deferred (torchdata is not
   a dependency; runs are restartable at pass granularity as before).
5. [x] Equality + smoke (2026-08-03): with shuffling off, the Hub-shard stream
   and the local raw-JSON stream produce IDENTICAL example sequences (30
   episodes → 4,324 examples, same labels, same feature checksums);
   tests/test_privacy_no_leak.py + test_il_pipeline.py pass (15/15), full
   suite 24/24 after the worktree models/ symlink fix.
   Measured throughput (default 192h/6L model, batch 64, MPS, 150 steps,
   steady-state windows, shards pre-cached):
   | source | workers | steps/s |
   |---|---|---|
   | local raw JSON (old path) | 0 | 11.35 |
   | hub shards | 0 | 11.4 |
   | hub shards | 4 | **14.1** (+24%) |
   | hub shards | 6 | 14.15 (plateau) |
   Cached shard streaming costs nothing vs raw local files; 4 workers is the
   recommended default; beyond 4 the bottleneck is MPS compute + collation,
   not shard parallelism. (The earlier "56% data-wait" single-worker note is
   partially, not fully, recoverable via workers.)
6. [ ] Revisit pricing only on triggers: >80 GB private used → PRO ($9/mo); encoder
   stabilized and loader CPU-bound → emit derived decision-level Parquet (v2).
7. [ ] Post-competition teardown: once training no longer needs the corpus, delete
   the private HF repo and the local shard cache, and cancel PRO if it was enabled.

### Addendum (2026-08-03 evening, post-merge session)

- Hub inventory re-verified independently against splits.json: 07-01 = 5,266
  rows, 07-26 = 4,554, eval 07-27 = 4,430 — all exact. `train/day=2026-08-03`
  (partial day, 2 episodes) packed, verified, uploaded; the raw pair kept
  locally (ingest will extend that day).
- Parity re-confirmed at tensor level: 24 restored train-2026-07-26 episodes →
  4,439 decisions, every feature tensor and label `torch.equal` between the
  raw-folder path and `ShardILDataset` over the Hub.
- **`--data-source auto` hardened**: the old check was `split_dir.exists()`,
  but raw dirs now hold pruned-target symlinks (train-combined union: 9,820
  links, 24 resolvable) or partial test-restored samples (24/4,554) — auto
  would have picked "local" and silently trained on the readable fraction
  while scheduling for the full count. Auto now requires BOTH splits' raw
  folders to be complete per splits.json, counting only symlink-resolvable
  files, and prints why it reroutes to hub.
- **Hub source now follows splits.json**: `--train-split`/`--eval-split`
  resolve to Hub day lists via `il_dataset.split_meta` (union splits use their
  `dates` field), instead of defaulting to every day under `train/` — new
  ingest days no longer leak into the corpus definition silently. `--hub-days`
  remains as an explicit override.
- `--dry-run` now redirects a default `--out` to `models/il_agent_dryrun`
  (a smoke test once overwrote the deployed `models/il_agent`; restored from
  `models/il_agent_3ep`, same step-38562 checkpoint).
- First full streamed training run: `runs/hfstream-combined-3ep` —
  train_combined (9,820 episodes, days 07-01+07-26 from the Hub), eval 07-27
  streamed, 3 epochs / 83,454 steps, 11.38 steps/s observed at step 1k
  (matches the no-worker row of the throughput table; run predates the
  `--num-workers 4` recommendation).
