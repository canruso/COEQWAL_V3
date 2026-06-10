"""Single source of truth for CalSim3 column-name grammar and variable matching.

Every COEQWAL column encodes the same grammar:

    Part B (MultiIndex level 1):  ``<base>_s####``        e.g. ``S_SHSTA_s0020``
    flattened (dashboard):        ``<SOURCE>_<base>_s####_<C>_<D>_<E>_<F>_<UNIT>``
                                  e.g. ``CALSIM_S_SHSTA_s0020_STORAGE_1MON_L2020A_PER-AVER_TAF``

This module is the ONLY place that knows that grammar. It does exact, parse-based
matching of a variable to its column(s) -- it never substring-matches, so a base
name that is a prefix of another (``S_SHSTA`` vs ``S_SHSTALEVEL1DV``;
``DEL_CVP_TOTAL`` vs ``DEL_CVP_TOTAL_N``) can never collide.

Design rules:
  * Pure: depends only on ``re`` / ``numpy``. Imports no other coeqwal module, so it
    can be a dependency of both ``metrics`` and ``cqwlutils`` without a cycle, and can
    be vendored byte-for-byte into the standalone dashboard.
  * No silent fallback: ``col_for`` RAISES on an ambiguous match instead of silently
    returning the first column. Over-matching surfaces immediately.

Variable names passed in are always the BARE base (``"S_SHSTA"``, ``"DEL_SWP_TOTA"``) --
never a trailing-underscore form. The grammar boundary (``_s####``) is supplied here.
"""

import re

import numpy as np

# Known Part-A source prefixes seen in flattened dashboard column names.
SOURCES = ("CALSIM", "CALCULATED", "MANUAL-ADD", "IWFM")

_SCEN = r"s\d+"
_PARTB_RE = re.compile(rf"^(?P<base>.+)_(?P<scen>{_SCEN})$")          # greedy: takes the LAST _s####
_SRC_RE = re.compile(r"^(?P<src>" + "|".join(SOURCES) + r")_(?P<rest>.+)$")
_FLAT_RE = re.compile(rf"^(?P<base>.+?)_(?P<scen>{_SCEN})(?:_(?P<tail>.*))?$")  # non-greedy: first _s####


def sid(n):
    """Canonical scenario id. Accepts int (20), str ('20' or 's0020') -> 's0020'."""
    if isinstance(n, str):
        m = re.fullmatch(r"s?0*(\d+)", n.strip())
        if not m:
            raise ValueError(f"varmatch.sid: not a scenario id: {n!r}")
        n = int(m.group(1))
    return f"s{int(n):04d}"


def parse_column(col):
    """Parse a column into ``{source, base, scenario, unit}`` or ``None``.

    Handles both a 7-level MultiIndex tuple and a flattened string. Returns ``None``
    for columns that are not ``<base>_s####`` variable columns (e.g. ``WaterYear``).
    """
    if isinstance(col, tuple):
        pb = parse_partb(col[1])
        if pb is None:
            return None
        base, scen = pb
        unit = str(col[6]).strip() if len(col) > 6 and col[6] is not None else None
        return {"source": str(col[0]).strip(), "base": base, "scenario": scen, "unit": unit}
    return _parse_flat(str(col))


def parse_partb(part_b):
    """``'<base>_s####'`` -> ``(base, 's####')``; ``None`` if no scenario suffix."""
    m = _PARTB_RE.match(str(part_b).strip())
    if not m:
        return None
    return m.group("base").strip(), m.group("scen")


def _parse_flat(s):
    s = s.strip()
    src = None
    m = _SRC_RE.match(s)
    if m:
        src, s = m.group("src"), m.group("rest")
    m = _FLAT_RE.match(s)
    if not m:
        return None
    tail = m.group("tail")
    unit = tail.rsplit("_", 1)[-1] if tail else None
    return {"source": src, "base": m.group("base").strip(), "scenario": m.group("scen"),
            "unit": unit.strip() if unit else None}


def base_of(col):
    r = parse_column(col)
    return r["base"] if r else None


def scenario_of(col):
    r = parse_column(col)
    return r["scenario"] if r else None


def unit_of(col):
    r = parse_column(col)
    return r["unit"] if r else None


def _unit_eq(a, b):
    if b is None:
        return True
    return a is not None and str(a).strip().upper() == str(b).strip().upper()


def mask_for(columns, var, unit=None):
    """Boolean mask selecting every column belonging to base ``var`` (any scenario),
    optionally filtered by ``unit``. Exact base match. Also accepts an exact full
    instance name (``"S_SHSTA_s0154"``) to select that single scenario's column --
    still exact equality, never substring."""
    var = str(var).strip()
    return np.array([
        (r := parse_column(c)) is not None and _unit_eq(r["unit"], unit)
        and (r["base"] == var or f"{r['base']}_{r['scenario']}" == var)
        for c in columns
    ])


def mask_for_many(columns, vars_, unit=None):
    """Union mask over several bases and/or full instance names (exact).
    Replaces ``str.contains('|'.join(...))``."""
    want = {str(v).strip() for v in vars_}
    return np.array([
        (r := parse_column(c)) is not None and _unit_eq(r["unit"], unit)
        and (r["base"] in want or f"{r['base']}_{r['scenario']}" in want)
        for c in columns
    ])


def col_for(columns, var, scenario, unit=None):
    """The single column for (base ``var``, ``scenario``[, ``unit``]).

    Returns ``None`` if absent. RAISES ``ValueError`` on >1 match (ambiguity) -- never
    silently returns the first, so over-matching is caught at the call site.
    """
    var = str(var).strip()
    scen = sid(scenario)
    matches = [
        c for c in columns
        if (r := parse_column(c)) is not None
        and r["base"] == var and r["scenario"] == scen and _unit_eq(r["unit"], unit)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"varmatch.col_for: ambiguous match for var={var!r} scenario={scen!r} "
            f"unit={unit!r} -> {len(matches)} columns: {matches[:4]}"
        )
    return matches[0] if matches else None


def build_index(columns):
    """``{(base, scenario, unit): column}`` for O(1) exact lookup. Raises on collision."""
    idx = {}
    for c in columns:
        r = parse_column(c)
        if r is None:
            continue
        key = (r["base"], r["scenario"], r["unit"])
        if key in idx:
            raise ValueError(f"varmatch.build_index: duplicate key {key}: {idx[key]} vs {c}")
        idx[key] = c
    return idx
