"""TOPPRA subprocess worker (runs inside the dedicated numpy<2 venv).

The main SDK runs on numpy>=2, where toppra's native solver hard-aborts on
Windows. This worker is launched by :mod:`_toppra_bridge` using the
``.toppra_env`` interpreter (numpy<2 + toppra + ecos), reads a single JSON
request from ``stdin`` and writes a single JSON response to ``stdout``.

Protocol
--------
Request  (stdin, one JSON object)::

    {
      "path":      [[j0, j1, ...], ...],   # required, N x n_joints
      "ctrl_freq": 0.05,                    # optional
      "max_vels":  [...] | null,            # optional
      "max_accs":  [...] | null             # optional
    }

Response (stdout, one JSON object)::

    {"ok": true,  "frames": [[...], ...]}   # M x n_joints
    {"ok": false, "error": "..."}

This module deliberately has **no dependency on the SDK**; it only needs
``numpy`` and the sibling :mod:`_toppra_interp` wrapper (which imports
``toppra``/``ecos``).
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw)

        import numpy as np
        from _toppra_interp import PiecewisePolyTOPPRA

        path = np.asarray(req["path"], dtype=float)
        if path.ndim != 2:
            raise ValueError(f"path must be 2-D (N, n_joints); got {path.shape}")

        frames = PiecewisePolyTOPPRA().interpolate_by_max_spdacc(
            path=path,
            ctrl_freq=float(req.get("ctrl_freq", 0.05)),
            max_vels=req.get("max_vels"),
            max_accs=req.get("max_accs"),
            toggle_debug=False,
        )
        out = {"ok": True, "frames": np.asarray(frames, dtype=float).tolist()}
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()
        return 0
    except Exception as exc:  # noqa: BLE001 - report every failure as JSON
        err = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()}
        sys.stdout.write(json.dumps(err))
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
