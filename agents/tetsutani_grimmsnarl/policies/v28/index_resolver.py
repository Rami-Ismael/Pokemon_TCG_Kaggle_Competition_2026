from __future__ import annotations

from .resource_ledger import action_signature


def resolve_index(options: list[dict], selected: list[int], legacy: list[int]) -> list[int]:
    """Resolve a strategic action into legal option indices.

    If the hierarchy selected the same semantic action as the legacy policy, retain
    the legacy index. This preserves validated serial/board-order tie-breaks. For a
    new strategic action, the hierarchy already supplies a concrete legal index.
    """
    if len(selected) == 1 and len(legacy) == 1:
        if action_signature(options[selected[0]]) == action_signature(options[legacy[0]]):
            return legacy
    return selected
