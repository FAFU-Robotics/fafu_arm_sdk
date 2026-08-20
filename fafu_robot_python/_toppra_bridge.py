"""Bridge to run TOPPRA in a dedicated numpy<2 subprocess.

toppra's native solver hard-aborts on Windows under numpy>=2 (see
``_toppra_interp`` / ``move_jntspace_path`` notes). The main SDK runs on
numpy>=2, so real time-optimal TOPPRA is offloaded to a sibling virtual
environment (``.toppra_env``, numpy<2 + toppra + ecos) via
:mod:`_toppra_worker`, exchanging JSON over ``stdin``/``stdout``.

Public API
----------
``BRIDGE_AVAILABLE``
    True when a usable ``.toppra_env`` interpreter was found.
``find_bridge_python()``
    Locate the numpy<2 interpreter (env var override + default location).
``interpolate_by_max_spdacc(path, ctrl_freq, max_vels, max_accs, timeout)``
    Time-parametrise ``path`` and return an ``(M, n_joints)`` numpy array.
    Same semantics as
    ``wrs.motion.trajectory.piecewisepoly_toppra.PiecewisePolyTOPPRA``.

The interpreter path can be overridden with the ``FAFU_TOPPRA_PYTHON``
environment variable (absolute path to a python.exe / python).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "_toppra_worker.py")


def find_bridge_python() -> Optional[str]:
    """Return the numpy<2 interpreter path, or ``None`` if not found."""
    override = os.environ.get("FAFU_TOPPRA_PYTHON")
    if override and os.path.exists(override):
        return override
    candidates = [
        os.path.join(_HERE, ".toppra_env", "Scripts", "python.exe"),  # Windows
        os.path.join(_HERE, ".toppra_env", "bin", "python"),          # POSIX
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


BRIDGE_AVAILABLE = find_bridge_python() is not None


def interpolate_by_max_spdacc(
    path,
    ctrl_freq: float = 0.05,
    max_vels: Optional[List[float]] = None,
    max_accs: Optional[List[float]] = None,
    timeout: float = 30.0,
) -> np.ndarray:
    """Run TOPPRA in the numpy<2 subprocess and return interpolated frames.

    Parameters mirror
    ``PiecewisePolyTOPPRA.interpolate_by_max_spdacc``. ``path`` is an
    ``(N, n_joints)`` array in radians (or any consistent unit).

    Raises
    ------
    RuntimeError
        If the bridge interpreter is missing, the subprocess fails, times
        out, or TOPPRA reports an error.
    """
    py = find_bridge_python()
    if py is None:
        raise RuntimeError(
            "TOPPRA bridge interpreter not found. Create the venv at "
            "'.toppra_env' (numpy<2 + toppra + ecos) or set FAFU_TOPPRA_PYTHON."
        )
    if not os.path.exists(_WORKER):
        raise RuntimeError(f"TOPPRA worker script missing: {_WORKER}")

    path_arr = np.asarray(path, dtype=float)
    req = {
        "path": path_arr.tolist(),
        "ctrl_freq": float(ctrl_freq),
        "max_vels": None if max_vels is None else list(np.asarray(max_vels, float)),
        "max_accs": None if max_accs is None else list(np.asarray(max_accs, float)),
    }

    try:
        proc = subprocess.run(
            [py, _WORKER],
            input=json.dumps(req),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"TOPPRA bridge timed out after {timeout}s") from exc

    if proc.returncode != 0 and not proc.stdout.strip():
        # Hard crash / import failure with no JSON payload.
        raise RuntimeError(
            f"TOPPRA bridge subprocess failed (code {proc.returncode}): "
            f"{proc.stderr.strip()[:500]}"
        )

    try:
        resp = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"TOPPRA bridge returned non-JSON output: {proc.stdout[:300]!r} "
            f"(stderr: {proc.stderr.strip()[:300]})"
        ) from exc

    if not resp.get("ok"):
        raise RuntimeError(f"TOPPRA bridge error: {resp.get('error')}")

    return np.asarray(resp["frames"], dtype=float)


class PiecewisePolyTOPPRA(object):
    """Duck-typed drop-in matching the wrs / vendored interpolator API.

    Lets callers use ``pwp.PiecewisePolyTOPPRA().interpolate_by_max_spdacc(...)``
    uniformly whether ``pwp`` is wrs, the in-process wrapper, or this bridge.
    """

    def interpolate_by_max_spdacc(
        self,
        path,
        ctrl_freq: float = 0.05,
        max_vels: Optional[List[float]] = None,
        max_accs: Optional[List[float]] = None,
        toggle_debug: bool = False,
    ) -> np.ndarray:
        return interpolate_by_max_spdacc(
            path, ctrl_freq=ctrl_freq, max_vels=max_vels, max_accs=max_accs
        )
