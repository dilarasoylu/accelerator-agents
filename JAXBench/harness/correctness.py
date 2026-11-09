"""Correctness checking for kernel outputs.

Default scheme (recommended): a per-output, magnitude-scaled absolute tolerance plus a dtype-keyed
relative tolerance, and a small outlier budget, instead of a single flat ``atol = rtol = 1e-2`` required
on 100% of elements. A flat ``1e-2`` is at once too loose for small-magnitude outputs and too strict for
large-magnitude reductions, and it ignores that bf16 and fp32 round very differently.

    pass(leaf)  iff  mean( |ref - test| <= atol + rtol * |ref| )  >=  REQUIRED_FRACTION
    atol = C * u(dtype) * max|ref|        # absolute floor, scaled to the output magnitude
    rtol = RTOL_BY_DTYPE[dtype]           # bf16 0.05, fp32 1e-5

where ``u`` is the unit roundoff of the reference output's dtype. The constants mirror SOL-ExecBench
(``C = 3``, 98% within tolerance, dtype-keyed rtol).

Back-compat: pass ``strict=True`` (or set ``MDS_STRICT_ALLCLOSE=1``, or pass explicit ``atol``/``rtol``)
to get the legacy flat ``1e-2`` / 100%-strict ``np.allclose`` behavior.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

# Legacy flat tolerance (KernelBench) -- used only on the strict / back-compat path.
ATOL = 1e-2
RTOL = 1e-2

# Recommended scheme constants.
C = 3.0                      # magnitude constant for the absolute floor (matches SOL-ExecBench)
REQUIRED_FRACTION = 0.98     # >= 98% of elements within tolerance (2% outlier budget)
RTOL_BY_DTYPE = {"bfloat16": 0.05, "float16": 0.05, "float32": 1e-5, "float64": 1e-9}
# Unit roundoff u = 2^-(stored_mantissa_bits + 1).
U_BY_DTYPE = {"bfloat16": 2.0 ** -8, "float16": 2.0 ** -11, "float32": 2.0 ** -24, "float64": 2.0 ** -53}


def _dtype_name(x):
    """Reference dtype name (before the float32 compare cast); defaults to float32 if unknown."""
    try:
        return jnp.dtype(x.dtype).name
    except Exception:
        return "float32"


def _leaf_pass(r_np, t_np, dtype_name, strict, atol, rtol):
    """(passed, max_diff, within_fraction) for one output leaf (both already float32)."""
    max_diff = float(np.max(np.abs(r_np - t_np))) if r_np.size else 0.0

    # A NaN/Inf in the candidate where the reference is finite is always a failure.
    bad = ~np.isfinite(t_np) & np.isfinite(r_np)
    if np.any(bad):
        return False, float("inf"), 0.0

    if strict:
        ok = bool(np.allclose(r_np, t_np, atol=(atol if atol is not None else ATOL),
                              rtol=(rtol if rtol is not None else RTOL)))
        return ok, max_diff, (1.0 if ok else 0.0)

    u = U_BY_DTYPE.get(dtype_name, U_BY_DTYPE["float32"])
    rt = RTOL_BY_DTYPE.get(dtype_name, RTOL_BY_DTYPE["float32"])
    a = C * u * float(np.max(np.abs(r_np))) if r_np.size else 0.0
    within = np.abs(r_np - t_np) <= (a + rt * np.abs(r_np))
    frac = float(np.mean(within)) if within.size else 1.0
    return (frac >= REQUIRED_FRACTION), max_diff, frac


def check_correctness(ref_output, test_output, atol=None, rtol=None, strict=None):
    """Compare reference and test outputs after converting to float32.

    Handles single tensors, tuples, and pytree outputs. By default uses the recommended
    magnitude-scaled / dtype-keyed / >=98% scheme; see the module docstring for the strict path.

    Returns a dict with keys: correct (bool), max_diff (float), within_fraction (float),
    reason (str: 'ok' | 'output count mismatch...' | 'shape mismatch...' | 'values differ...').
    """
    # strict if explicitly requested, if the env flag is set, or if a caller pins atol/rtol
    # (an explicit tolerance means "use exactly this", i.e. the legacy allclose contract).
    if strict is None:
        strict = (os.environ.get("MDS_STRICT_ALLCLOSE", "") == "1") or (atol is not None) or (rtol is not None)

    ref_flat = jax.tree.leaves(ref_output) if isinstance(ref_output, (tuple, list)) else [ref_output]
    test_flat = jax.tree.leaves(test_output) if isinstance(test_output, (tuple, list)) else [test_output]

    if len(ref_flat) != len(test_flat):
        return {
            'correct': False,
            'max_diff': float('inf'),
            'within_fraction': 0.0,
            'reason': f'output count mismatch: {len(ref_flat)} vs {len(test_flat)}',
        }

    max_diff = 0.0
    min_fraction = 1.0
    for r, t in zip(ref_flat, test_flat):
        dtype_name = _dtype_name(r)
        r_np = np.array(r, dtype=np.float32)
        t_np = np.array(t, dtype=np.float32)

        if r_np.shape != t_np.shape:
            return {
                'correct': False,
                'max_diff': float('inf'),
                'within_fraction': 0.0,
                'reason': f'shape mismatch: {r_np.shape} vs {t_np.shape}',
            }

        passed, diff, frac = _leaf_pass(r_np, t_np, dtype_name, strict, atol, rtol)
        max_diff = max(max_diff, diff)
        min_fraction = min(min_fraction, frac)
        if not passed:
            return {
                'correct': False,
                'max_diff': max_diff,
                'within_fraction': frac,
                'reason': (f'values differ (within {frac:.4f} < {REQUIRED_FRACTION})'
                           if not strict else 'values differ'),
            }

    return {
        'correct': True,
        'max_diff': max_diff,
        'within_fraction': min_fraction,
        'reason': 'ok',
    }
