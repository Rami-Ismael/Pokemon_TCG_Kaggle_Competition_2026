"""Shared cg SearchBegin wrapper for finish-mode lethal verification."""

from __future__ import annotations

import json
import time


def try_cg_search(
    obs_dict: dict,
    options: list,
    *,
    budget_ms: float = 200,
    audit_log: str | None = None,
) -> list[int] | None:
    """Return best option index from cg search_* if engine provides begin_input."""
    if not options:
        return None
    begin_input = obs_dict.get("search_begin_input", "")
    if not begin_input:
        return None
    try:
        from cg.sim import Battle, lib  # type: ignore

        ptr = Battle.battle_ptr
        if ptr is None:
            return None
        deadline = time.monotonic() + budget_ms / 1000.0
        n_opts = len(options)
        ctypes = __import__("ctypes")
        idx_arr = (ctypes.c_int * n_opts)(*range(n_opts))
        out_idx = (ctypes.c_int * 1)(0)
        out_score = (ctypes.c_int * 1)(0)
        out_depth = (ctypes.c_int * 1)(0)
        out_nodes = (ctypes.c_int * 1)(0)
        out_time = (ctypes.c_int * 1)(0)
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        search_ptr = lib.SearchBegin(
            ptr,
            begin_input.encode("ascii"),
            remaining_ms,
            idx_arr,
            out_idx,
            out_score,
            out_depth,
            out_nodes,
            out_time,
            n_opts,
        )
        if not search_ptr:
            return None
        handle = int(search_ptr, 16) if isinstance(search_ptr, str) else 0
        try:
            step = lib.SearchStep(ptr, handle, out_idx, 1)
            if step:
                data = json.loads(step.decode()) if isinstance(step, bytes) else {}
                pick = data.get("index", out_idx[0])
                if isinstance(pick, int) and 0 <= pick < n_opts:
                    return [pick]
            if 0 <= out_idx[0] < n_opts:
                return [out_idx[0]]
        finally:
            lib.SearchEnd(ptr)
            if handle:
                lib.SearchRelease(ptr, handle)
    except Exception:
        if audit_log:
            _audit(audit_log, "finish_search_exception")
        return None
    return None


def _audit(path: str, event: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event}) + "\n")
    except Exception:
        pass
