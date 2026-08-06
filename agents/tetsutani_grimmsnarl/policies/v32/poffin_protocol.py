from __future__ import annotations

"""Handwritten Poffin completion rule for a Snorunt-only result."""
import os
from . import validated_rule_profile as profile


class ManualPoffinProtocol:
    RULE = "TAKE_SNORUNT_WHEN_BASELINE_EMPTY"

    def __init__(self) -> None:
        raw = os.environ.get("PTCG_V32_POFFIN_RULES", "ALL").strip()
        self.enabled = raw in ("", "ALL") or self.RULE in {x for x in raw.split(",") if x}

    def resolve(self, view: dict, selected: list[int]) -> list[int]:
        if selected or not self.enabled:
            return selected
        opts = view.get("opts") or []
        available = [int(s.get("source_id", 0) or 0) for s in opts]
        bench_free = max(
            0,
            int((view.get("me") or {}).get("bench_max", 5) or 5)
            - len((view.get("me") or {}).get("bench") or []),
        )
        if profile.SNORUNT not in available or bench_free <= 0:
            return selected
        take = min(
            2,
            int(view.get("max", 0) or 0),
            bench_free,
            available.count(profile.SNORUNT),
        )
        return profile._select_ids(view, [profile.SNORUNT] * take, max_count=take)
