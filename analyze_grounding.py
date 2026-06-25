"""Offline check of the smart-gate premise against REAL grounding-fire contexts.

Reads /tmp/grounding_contexts.jsonl (captured by the TOT_GROUNDING_LOG
instrumentation across many real physics problems). For every context where the
CURRENT grounding check fired, it:
  1. categorizes WHY (suffix/alias mismatch, numeric value, descriptive value,
     or genuinely-unrelated), and
  2. runs a PROTOTYPE "smart" predicate (suffix/alias normalization +
     numeric-value + descriptive-exempt) to measure how many fires it would
     ELIMINATE (recovered false negatives),
  3. runs a hand-built set of genuinely-hallucinated BCs through the smart
     predicate to confirm it STILL flags real garbage (not advisory-in-disguise).

This tests whether the proposed fix is empirically justified BEFORE implementing
it in the system.
"""
import json
import re
import sys

from skills import _text_symbols, _normalize_boundary_condition_key_text, _boundary_condition_value_text

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/grounding_contexts.jsonl"

# tokens that mark a "phase" of a quantity, not a distinct variable
_PHASE_SUFFIXES = {"initial", "final", "init", "fin", "start", "end", "max", "min",
                   "0", "1", "2", "f", "i", "o", "left", "right", "top", "bottom",
                   "in", "out", "surface", "mid", "exit", "hole"}


def _norm_token(tok: str) -> str:
    """Reduce a symbol to its base: lowercase, drop phase suffix parts, strip trailing digits."""
    parts = [p for p in re.split(r"[_\s]+", str(tok).lower()) if p]
    parts = [p for p in parts if p not in _PHASE_SUFFIXES]
    parts = [re.sub(r"\d+$", "", p) or p for p in parts]
    return "".join(parts)


def _norm_set(symbols):
    out = set()
    for s in symbols:
        n = _norm_token(s)
        if n:
            out.add(n)
        # also keep the per-part bases (e.g. "v_initial" -> {"v"})
        for p in re.split(r"[_\s]+", str(s).lower()):
            p = re.sub(r"\d+$", "", p)
            if p and p not in _PHASE_SUFFIXES:
                out.add(p)
    return out


def _is_numeric(value_text: str) -> bool:
    t = str(value_text).strip()
    if not t:
        return False
    try:
        float(t)
        return True
    except (TypeError, ValueError):
        return bool(re.match(r"^[-+]?\d", t))  # starts with a number (e.g. "20 C")


def _key_from_violation(msg: str):
    m = re.search(r"not grounded in equations or known variables:\s*(.+)$", msg)
    if not m:
        return None
    key = m.group(1).strip()
    # axis form is "axis = value"; take the axis
    if "=" in key:
        key = key.split("=", 1)[0].strip()
    return key


def smart_grounded(key_text: str, value_text: str, eq_norm: set) -> bool:
    """Proposed predicate: True = grounded (do NOT flag)."""
    # 1. descriptive (non-numeric value) -> annotation, exempt
    if not _is_numeric(value_text):
        return True
    # 2. normalized-name match against equation symbols (handles suffix/alias)
    norm_key, has_label = _normalize_boundary_condition_key_text(key_text)
    key_syms = _norm_set(_text_symbols(norm_key.split("=", 1)[0]))
    if key_syms & eq_norm:
        return True
    # 3. numeric value but base var absent everywhere -> genuinely ungrounded
    return False


def main():
    try:
        rows = [json.loads(l) for l in open(LOG) if l.strip()]
    except FileNotFoundError:
        print(f"no log at {LOG}")
        return
    grounding_fires = []
    for r in rows:
        eqs = r.get("equations", [])
        kv = r.get("known_vars", [])
        bcs = r.get("boundary_conditions", {})
        eq_syms = set()
        for e in eqs:
            eq_syms.update(_text_symbols(str(e)))
        eq_syms.update(str(k) for k in kv)
        eq_norm = _norm_set(eq_syms)
        for v in r.get("violations", []):
            if "not grounded" not in v:
                continue
            key = _key_from_violation(v)
            if key is None:
                continue
            val = bcs.get(key, bcs.get(key + " ", ""))
            # value may be under the raw key incl. '=' ; best effort
            if not val:
                for bk, bv in bcs.items():
                    if bk.split("=", 1)[0].strip() == key:
                        val = bv
                        break
            grounding_fires.append({"key": key, "value": str(val), "eq_norm": eq_norm,
                                    "numeric": _is_numeric(str(val))})

    n = len(grounding_fires)
    if not n:
        print(f"contexts loaded: {len(rows)}; grounding fires: 0")
        return
    recovered = sum(1 for g in grounding_fires if smart_grounded(g["key"], g["value"], g["eq_norm"]))
    numeric = sum(1 for g in grounding_fires if g["numeric"])
    descriptive = n - numeric
    still = n - recovered

    print(f"contexts loaded: {len(rows)}")
    print(f"grounding fires (current): {n}")
    print(f"  by value type: descriptive(non-numeric)={descriptive}  numeric={numeric}")
    print(f"smart predicate ELIMINATES: {recovered}/{n} ({recovered/n:.0%})  -- recovered false negatives")
    print(f"smart predicate STILL flags: {still}/{n}")
    print("\nsample STILL-flagged (candidate true positives -- inspect these are really ungrounded):")
    shown = 0
    for g in grounding_fires:
        if not smart_grounded(g["key"], g["value"], g["eq_norm"]):
            print(f"  key={g['key']!r} value={g['value']!r}")
            shown += 1
            if shown >= 12:
                break

    # adversarial negatives: genuinely hallucinated numeric BCs unrelated to equations
    print("\n=== adversarial check: smart predicate must STILL flag genuine garbage ===")
    adversarial = [
        ("psi", "0", ["F = m*a", "d = v0**2/(2*a)"]),
        ("phantom_axis", "5", ["E = 0.5*k*x**2"]),
        ("zeta", "0", ["v = omega*r", "a = v**2/r"]),
        ("chi_boundary", "3.0", ["P*V = n*R*T"]),
    ]
    caught = 0
    for key, val, eqs in adversarial:
        en = _norm_set(set().union(*[set(_text_symbols(e)) for e in eqs]))
        flagged = not smart_grounded(key, val, en)
        caught += flagged
        print(f"  {key}={val} vs {eqs}: {'FLAGGED (good)' if flagged else 'MISSED (bad)'}")
    print(f"adversarial retention: {caught}/{len(adversarial)} genuine-garbage BCs still flagged")

    print("\nVERDICT: smart gate is justified iff it recovers a large fraction of real fires "
          "AND retains adversarial detection.")


if __name__ == "__main__":
    main()
