import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pokemon_tcg import config
from pokemon_tcg.il_dataset import iter_episode_decisions, encode_observation, _resolve_hub_shards
sys.path.insert(0, str(config.PROJECT_ROOT/"data"/"external"/"cg-lib"))
from cg.api import SelectContext
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
split, maxep = sys.argv[1], int(sys.argv[2])
ctx=collections.Counter(); nopts=collections.Counter(); n=0; rows=0
for rel in _resolve_hub_shards(config.HF_EPISODES_REPO, split, days=None):
    if n>=maxep: break
    pf=pq.ParquetFile(hf_hub_download(config.HF_EPISODES_REPO, rel, repo_type="dataset"))
    for rb in pf.iter_batches(batch_size=8, columns=["episode_json"]):
        for raw in rb.to_pydict()["episode_json"]:
            if n>=maxep: break
            n+=1
            for obs,label,exc,meta in iter_episode_decisions(json.loads(raw)):
                f=encode_observation(obs, exclude=exc)
                if f is None: continue
                c=int(f["select_context"].item()); ctx[c]+=1; rows+=1
                if c==30: nopts[int(f["n_real_options"])]+=1
        if n>=maxep: break
print(f"split={split}  episodes consumed={n}  rows={rows}")
tot=sum(ctx.values())
print(f"ctx 30 (DISCARD_ENERGY): {ctx[30]} rows = {ctx[30]/tot:.3%} of corpus")
print(f"  n_opts breakdown: {sorted(nopts.items())}  forced share={nopts[1]/max(1,sum(nopts.values())):.1%}")
print("\ntop 12 contexts by volume:")
for c,k in ctx.most_common(12):
    nm = SelectContext(c).name if c < len(SelectContext) else str(c)
    print(f"  {c:>3} {nm:<28} {k:>7} {k/tot:>7.2%}")
print(f"\nctx 30 rank: {[c for c,_ in ctx.most_common()].index(30)+1} of {len(ctx)} contexts present")
