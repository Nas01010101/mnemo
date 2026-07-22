"""TenetStore — a LangGraph `BaseStore` backed by Tenet's bi-temporal MemoryCore.

LangGraph's `BaseStore` (langgraph.store.base) is the drop-in long-term-memory
interface every serious store (InMemoryStore, Postgres/Redis stores, LangMem)
implements: an abstract `batch`/`abatch` over four Op types —
`GetOp` / `PutOp` / `SearchOp` / `ListNamespacesOp` — with `get`/`put`/`search`/
`delete`/`list_namespaces` (+ their `a*` async twins) provided concretely on the
base class in terms of `batch`/`abatch`. This module implements exactly that
Op contract (verified against the installed `langgraph` 1.0.5 /
`langgraph-checkpoint` 3.0.1 source, not from memory) so Tenet can be dropped
into any LangGraph agent as its `store=`.

What's different from every other BaseStore implementation: `put()` does not
overwrite. A re-put of the same `(namespace, key)` with a different value
SUPERSEDES the prior value (invalidated, not deleted) — the same bi-temporal
mechanism `Tenet.ingest()` uses for conversational facts. History survives; it's
just not exposed through `BaseStore`'s synchronous-snapshot API (LangGraph has
no time-travel verb). `search(query=...)` also annotates each hit with Tenet's
learned-dynamics confidence (`p_valid` — see dynamics.py) that the fact is
still current, since `SearchItem` has no separate metadata slot to carry it.

Mapping:
  (namespace, key)              -> Tenet skey = "::".join((*namespace, key))
  put(value)                    -> MemoryCore.store(json.dumps(value), key=skey)
                                    (same key + different value ⇒ supersede)
  get                           -> exact current-value lookup by skey
  search(query=...)             -> MemoryCore.recall(query) filtered to the
                                    namespace prefix, each hit's value carries
                                    "_tenet_p_valid" when the dynamics model has
                                    an opinion
  search(no query)              -> plain namespace-prefix listing, newest first
  delete / put(value=None)      -> archived (Tenet's soft-delete; excluded from
                                    every read path here, kept for forget_sweep
                                    bookkeeping rather than a hard row delete)

Install: pip install 'tenet-memory[langgraph] @ git+https://github.com/Nas01010101/tenet.git'
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Iterable

try:
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        Item,
        ListNamespacesOp,
        MatchCondition,
        Op,
        PutOp,
        Result,
        SearchItem,
        SearchOp,
    )
except ImportError as e:  # pragma: no cover - exercised via scripts/test_langgraph_store.py
    raise ImportError(
        "TenetStore needs langgraph: pip install 'tenet-memory[langgraph] @ git+https://github.com/Nas01010101/tenet.git'"
    ) from e

from .. import memory as _memory

_SEP = "::"
_SOURCE = "langgraph_store"  # tags rows written through this adapter, so search/list
                              # never mixes in unrelated Tenet.ingest() fact rows that
                              # happen to share the DB.


def _validate_parts(namespace: tuple[str, ...], key: str) -> None:
    for part in (*namespace, key):
        if _SEP in part:
            raise ValueError(
                f"namespace/key part {part!r} contains the reserved separator "
                f"{_SEP!r} (Tenet joins namespace+key into one skey with it)"
            )


def _skey(namespace: tuple[str, ...], key: str) -> str:
    _validate_parts(namespace, key)
    return _SEP.join((*namespace, key))


def _split_skey(skey: str) -> tuple[tuple[str, ...], str]:
    parts = skey.split(_SEP)
    return tuple(parts[:-1]), parts[-1]


def _does_match(condition: "MatchCondition", ns: tuple[str, ...]) -> bool:
    """Same semantics as langgraph.store.memory.InMemoryStore's `_does_match`:
    "*" matches any single namespace element at that position."""
    path = condition.path
    if len(ns) < len(path):
        return False
    if condition.match_type == "prefix":
        pairs = zip(ns, path)
    elif condition.match_type == "suffix":
        pairs = zip(reversed(ns), reversed(path))
    else:  # pragma: no cover - base class only emits prefix/suffix today
        raise ValueError(f"unsupported match type: {condition.match_type}")
    return all(p == "*" or n == p for n, p in pairs)


def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    if operator == "$eq":
        return value == op_value
    if operator == "$ne":
        return value != op_value
    if operator == "$gt":
        return value is not None and float(value) > float(op_value)
    if operator == "$gte":
        return value is not None and float(value) >= float(op_value)
    if operator == "$lt":
        return value is not None and float(value) < float(op_value)
    if operator == "$lte":
        return value is not None and float(value) <= float(op_value)
    raise ValueError(f"unsupported filter operator: {operator}")


def _matches_filter(value: dict, filt: dict) -> bool:
    """Top-level-field filter, matching BaseStore's documented $eq/$ne/$gt/$gte/
    $lt/$lte operators plus plain equality. Simplification vs the reference
    InMemoryStore: only top-level `value` keys are matched (no dotted/nested
    paths) — sufficient for the flat dicts this adapter round-trips."""
    for k, want in filt.items():
        got = value.get(k)
        if isinstance(want, dict) and any(str(op).startswith("$") for op in want):
            if not all(_apply_operator(got, op, ov) for op, ov in want.items()):
                return False
        elif got != want:
            return False
    return True


class TenetStore(BaseStore):
    """LangGraph `BaseStore` backed by a Tenet `MemoryCore`.

    Pass an existing `MemoryCore` to share a store already in use elsewhere in
    the process, or a `db_path` to open/create one:

        store = TenetStore(db_path="data/agent.db")
        store.put(("users", "alex"), "prefs", {"theme": "dark"})
        store.get(("users", "alex"), "prefs").value  # -> {"theme": "dark"}
        store.put(("users", "alex"), "prefs", {"theme": "light"})  # supersedes
        store.search(("users", "alex"), query="ui preferences")

    `index=`/`ttl=` on `put`/`aput`: Tenet embeds every stored item unconditionally
    (that's its storage substrate, not an opt-in feature), so `index=False` is
    accepted for signature compatibility but has no effect. TTL is unsupported
    (`supports_ttl` stays `False`, inherited default) — the base class raises
    before `batch` is ever called if a caller passes one.
    """

    def __init__(
        self,
        core: "_memory.MemoryCore | None" = None,
        *,
        db_path: str | None = None,
        now=time.time,
    ):
        if core is not None and db_path is not None:
            raise ValueError("pass either core= or db_path=, not both")
        self.core = core if core is not None else (
            _memory.MemoryCore(db_path, now=now) if db_path else _memory.MemoryCore(now=now)
        )

    # ---- BaseStore's only required surface --------------------------------
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return [self._dispatch(op) for op in ops]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # MemoryCore's I/O (sqlite + the embedding call) is synchronous/blocking —
        # there's no native asyncio path to hook into, so offload to a thread
        # rather than block the caller's event loop.
        return await asyncio.to_thread(self.batch, list(ops))

    def _dispatch(self, op: Op) -> Result:
        if isinstance(op, GetOp):
            return self._get(op)
        if isinstance(op, PutOp):
            return self._put(op)
        if isinstance(op, SearchOp):
            return self._search(op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(op)
        raise TypeError(f"unsupported op: {type(op).__name__}")  # pragma: no cover

    # ---- ops ----------------------------------------------------------------
    def _get(self, op: GetOp) -> Item | None:
        skey = _skey(op.namespace, op.key)
        row = self.core.db.execute(
            "SELECT * FROM memories WHERE skey=? AND source=? AND kind='fact' "
            "AND archived=0 AND expired_at IS NULL",
            (skey, _SOURCE),
        ).fetchone()
        return self._row_to_item(row) if row is not None else None

    def _put(self, op: PutOp) -> None:
        if op.value is None:  # PutOp(value=None) is how BaseStore.delete() spells delete
            self.core.db.execute(
                "UPDATE memories SET archived=1 WHERE skey=? AND source=? AND archived=0",
                (_skey(op.namespace, op.key), _SOURCE),
            )
            self.core.db.commit()
            return None
        skey = _skey(op.namespace, op.key)
        text = json.dumps(op.value, sort_keys=True)  # sort_keys: identical values
        self.core.store(text, key=skey, kind="fact", source=_SOURCE)  # dedupe as a restatement
        return None

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        if op.query:
            return self._search_semantic(op, prefix)
        return self._search_listing(op, prefix)

    def _search_semantic(self, op: SearchOp, prefix: tuple[str, ...]) -> list[SearchItem]:
        # Over-fetch: recall() ranks the WHOLE store, then we filter to this
        # namespace prefix, so we need more than limit+offset raw hits to still
        # have `limit` left after filtering.
        hits = self.core.recall(op.query, k=max((op.limit + op.offset) * 4, 40))
        out: list[SearchItem] = []
        for h in hits:
            if h.kind != "fact" or h.source != _SOURCE or not h.key:
                continue
            ns, key = _split_skey(h.key)
            if ns[: len(prefix)] != prefix:
                continue
            try:
                value = json.loads(h.text)
            except json.JSONDecodeError:  # pragma: no cover - defensive only
                continue
            if op.filter and not _matches_filter(value, op.filter):
                continue
            if h.confidence is not None:
                # SearchItem carries no metadata slot beyond `value`/`score` in the
                # real contract — this is the only place p_valid can ride.
                value = {**value, "_tenet_p_valid": h.confidence}
            out.append(SearchItem(
                namespace=ns, key=key, value=value,
                created_at=_ts(h.created_at), updated_at=_ts(h.last_access),
                score=h.score,
            ))
        return out[op.offset: op.offset + op.limit]

    def _search_listing(self, op: SearchOp, prefix: tuple[str, ...]) -> list[SearchItem]:
        rows = self.core.db.execute(
            "SELECT * FROM memories WHERE kind='fact' AND source=? AND archived=0 "
            "AND expired_at IS NULL AND skey IS NOT NULL ORDER BY created_at DESC",
            (_SOURCE,),
        ).fetchall()
        out: list[SearchItem] = []
        for row in rows:
            ns, key = _split_skey(row["skey"])
            if ns[: len(prefix)] != prefix:
                continue
            item = self._row_to_item(row)
            if op.filter and not _matches_filter(item.value, op.filter):
                continue
            out.append(SearchItem(
                namespace=item.namespace, key=item.key, value=item.value,
                created_at=item.created_at, updated_at=item.updated_at, score=None,
            ))
        return out[op.offset: op.offset + op.limit]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        rows = self.core.db.execute(
            "SELECT DISTINCT skey FROM memories WHERE kind='fact' AND source=? "
            "AND archived=0 AND expired_at IS NULL AND skey IS NOT NULL",
            (_SOURCE,),
        ).fetchall()
        namespaces = {_split_skey(row["skey"])[0] for row in rows}
        if op.match_conditions:
            namespaces = {ns for ns in namespaces
                          if all(_does_match(c, ns) for c in op.match_conditions)}
        if op.max_depth is not None:
            namespaces = {ns[: op.max_depth] for ns in namespaces}
        return sorted(namespaces)[op.offset: op.offset + op.limit]

    # ---- helpers ------------------------------------------------------------
    def _row_to_item(self, row) -> Item:
        namespace, key = _split_skey(row["skey"])
        return Item(
            value=json.loads(row["text"]), key=key, namespace=namespace,
            created_at=_ts(row["created_at"]), updated_at=_ts(row["last_access"]),
        )

    def close(self) -> None:
        self.core.close()


def _ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)
