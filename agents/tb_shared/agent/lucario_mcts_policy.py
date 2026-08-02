"""Submission wrapper: RL+MCTS checkpoint + LucarioScorer legal fallback."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from agent.agent import HeuristicScorer, load_deck
from agent.lucario_policy import LucarioScorer, is_lucario_deck

DEFAULT_SEARCH_COUNT = 12


class LucarioMCTSScorer(HeuristicScorer):
    def __init__(
        self,
        deck_path: str | None = None,
        model_path: str | None = None,
        meta_path: str | None = None,
        rng=None,
    ) -> None:
        super().__init__(rng=rng)
        deck_ids = load_deck(deck_path) if deck_path else []
        if is_lucario_deck(deck_ids):
            self._fallback = LucarioScorer(rng=rng, deck_path=deck_path)
        else:
            from agent.rule_core import RuleCoreScorer

            self._fallback = RuleCoreScorer(rng=rng, deck_path=deck_path)
        self._deck = deck_ids or []
        self._ready = False
        self._rt = None
        self._model = None
        self._load_model(model_path, meta_path)

    def _load_model(self, model_path: str | None, meta_path: str | None) -> None:
        try:
            import torch
            from agent import lucario_mcts_runtime as rt

            search_count = int(os.environ.get("LUC_SUBMIT_SEARCH_COUNT", DEFAULT_SEARCH_COUNT))
            rt.SEARCH_COUNT = max(1, search_count)

            cfg = {}
            if meta_path and Path(meta_path).is_file():
                meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
                cfg = meta.get("config") or meta
            for key, attr in (
                ("LUC_D_MODEL", "D_MODEL"),
                ("LUC_HEADS", "NUM_HEADS"),
                ("LUC_D_FF", "D_FF"),
                ("LUC_ENC_LAYERS", "ENC_LAYERS"),
                ("LUC_DEC_LAYERS", "DEC_LAYERS"),
            ):
                if key in cfg:
                    setattr(rt, attr, int(cfg[key]))

            d_model = rt.D_MODEL
            device = torch.device("cpu")
            model = rt.MyModel(d_model, rt.NUM_HEADS, rt.D_FF, rt.ENC_LAYERS, rt.DEC_LAYERS).to(device)

            root = Path(__file__).resolve().parents[1]
            candidates = []
            if model_path:
                candidates.append(Path(model_path))
            candidates.extend([
                root / "agent" / "models" / "lucario_model_best.pth",
                root / "rl_mcts_field" / "lucarioex_v1" / "model_best.pth",
            ])
            ckpt = next((p for p in candidates if p.is_file()), None)
            if ckpt is None:
                return

            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state)
            model.eval()

            if not self._deck:
                self._deck = list(rt.LUCARIO_DECK)

            self._rt = rt
            self._model = model
            self._torch = torch
            self._ready = True
        except Exception:
            self._ready = False

    def choose(self, obs_dict, select, current, options):
        if not options:
            return []
        if not self._ready or self._model is None or self._rt is None:
            return self._fallback.choose(obs_dict, select, current, options)
        try:
            with self._torch.inference_mode():
                selected, _ = self._rt.mcts_agent(
                    obs_dict, self._deck, self._model, opp_deck=self._deck,
                )
            return selected
        except Exception:
            return self._fallback.choose(obs_dict, select, current, options)


def build_lucario_mcts_scorer(
    deck_path: str | None = None,
    model_path: str | None = None,
    meta_path: str | None = None,
    rng=None,
) -> LucarioMCTSScorer:
    return LucarioMCTSScorer(
        deck_path=deck_path,
        model_path=model_path,
        meta_path=meta_path,
        rng=rng,
    )
