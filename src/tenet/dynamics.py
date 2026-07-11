"""Fact dynamics — the drift/staleness model over the bi-temporal belief store.

The ledger is a training set for *how facts change*: every superseded fact is an
observed lifetime (valid_at -> invalid_at), every current fact a right-censored one.
From these we learn, per key class (the attribute part of "subject::attribute"):

  1. a CHANGE-RATE posterior — conjugate Gamma(a, b) over an exponential lifetime
     rate; the posterior-predictive survival is Lomax:

         S(dt | class) = (b_post / (b_post + dt)) ** a_post
         a_post = a0 + n_supersessions,   b_post = b0 + total_exposure_seconds

     so P(a fact is still valid) is *learned* from that attribute's own churn:
     "residence" learns a slow hazard, "mood" a fast one, automatically.

  2. a RIPPLE matrix — P(class B superseded within a window | class A superseded),
     counted from co-supersession events. When A changes, correlated keys' survival
     drops: the store knows what it *no longer knows* and can surface it.

Everything is closed-form, LLM-free, and fits in one SQL scan. Refit is cheap;
MemoryCore refits lazily and at every forget_sweep().
"""
from __future__ import annotations

import time
from collections import defaultdict

# Weak prior with mean lifetime ~90d but only ~2 weeks of pseudo-exposure, so a
# handful of real observations dominates it (a churny key learns fast).
_B0 = 14 * 24 * 3600.0
_A0 = 0.15  # a0/b0 = prior mean change rate ≈ 1/93d
_RIPPLE_WINDOW_S = 7 * 24 * 3600.0   # B superseded within 7d of A => co-change event
_RIPPLE_MIN_P = 0.34                 # below this, ignore the correlation
_RIPPLE_GAMMA = 1.0                  # strength of the survival penalty on ripple


def key_class(skey: str | None) -> str | None:
    """'user::residence' -> 'residence' (the attribute is the dynamics unit)."""
    if not skey:
        return None
    return skey.rsplit("::", 1)[-1].strip().lower() or None


class Dynamics:
    """Fitted fact-lifetime model. Build with `fit(rows, now)`; query with
    `p_valid(skey, age_s)` and `ripple_bump(skey, now)`."""

    def __init__(self):
        self._post: dict[str, tuple[float, float]] = {}   # class -> (a_post, b_post)
        self._ripple: dict[str, dict[str, float]] = {}    # A -> {B: P(B|A)}
        self._recent_super: dict[str, float] = {}         # class -> last supersession ts

    # ---- fitting ---------------------------------------------------------
    @classmethod
    def fit(cls, rows, now: float | None = None) -> "Dynamics":
        """rows: iterable of sqlite Rows/dicts with skey, valid_at, invalid_at
        (keyed fact rows only, superseded AND current)."""
        now = now if now is not None else time.time()
        d = cls()
        events: dict[str, int] = defaultdict(int)
        exposure: dict[str, float] = defaultdict(float)
        super_ts: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            k = key_class(r["skey"])
            if k is None:
                continue
            if r["invalid_at"] is not None:                       # observed lifetime
                events[k] += 1
                exposure[k] += max(0.0, r["invalid_at"] - r["valid_at"])
                super_ts[k].append(r["invalid_at"])
            else:                                                 # censored (current)
                exposure[k] += max(0.0, now - r["valid_at"])
        for k in exposure:
            d._post[k] = (_A0 + events[k], _B0 + exposure[k])
        # ripple: P(B superseded within window | A superseded), Laplace-smoothed
        classes = [k for k in super_ts if super_ts[k]]
        for a in classes:
            n_a = len(super_ts[a])
            for b in classes:
                if a == b:
                    continue
                co = sum(
                    1 for ta in super_ts[a]
                    if any(0 <= tb - ta <= _RIPPLE_WINDOW_S for tb in super_ts[b] if tb != ta or a < b)
                )
                p = (co + 1) / (n_a + 2)
                if p >= _RIPPLE_MIN_P and co >= 1:
                    d._ripple.setdefault(a, {})[b] = p
            d._recent_super[a] = max(super_ts[a])
        return d

    # ---- queries ---------------------------------------------------------
    def p_valid(self, skey: str | None, age_s: float, now: float | None = None) -> float:
        """Posterior-predictive P(fact of this key is still the current truth),
        given it was asserted age_s seconds ago. Includes the ripple penalty."""
        k = key_class(skey)
        a, b = self._post.get(k, (_A0, _B0))
        s = (b / (b + max(0.0, age_s))) ** a
        if k is not None and now is not None:
            bump = self.ripple_bump(k, now)
            if bump > 0:
                s = s ** (1.0 + _RIPPLE_GAMMA * bump)
        return float(s)

    def ripple_bump(self, klass: str, now: float) -> float:
        """Sum of P(this class changed | recently-changed correlated class),
        over classes superseded within the ripple window before `now`."""
        bump = 0.0
        for a, targets in self._ripple.items():
            if a == klass:
                continue
            ta = self._recent_super.get(a)
            if ta is not None and 0 <= now - ta <= _RIPPLE_WINDOW_S:
                bump += targets.get(klass, 0.0)
        return bump

    def expected_lifetime_days(self, skey: str | None) -> float | None:
        """Posterior mean lifetime for the key's class (None if a<=1 undefined)."""
        a, b = self._post.get(key_class(skey), (_A0, _B0))
        return (b / max(a - 1.0, 1e-9)) / 86400.0 if a > 1.0 else None
