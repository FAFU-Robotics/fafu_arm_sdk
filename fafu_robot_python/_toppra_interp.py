"""Vendored TOPPRA piecewise-polynomial interpolator.

This is a lightweight, self-contained copy of ``wrs``'s
``wrs.motion.trajectory.piecewisepoly_toppra.PiecewisePolyTOPPRA`` so the
Fafu SDK can time-parametrise joint-space paths **without** depending on the
whole ``wrs`` framework.

It only needs the standalone ``toppra`` package (``pip install toppra``),
which pulls in numpy/scipy. The public API (``interpolate_by_max_spdacc``)
is byte-for-byte compatible with the ``wrs`` original so
``move_jntspace_path`` can use either transparently.

Original author: weiwei (Weiwei Wan, wrs project).
"""

from __future__ import annotations

import math

import numpy as np

import sys

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

# toppra's default ``seidel`` solver wrapper has an int32/int64 buffer-dtype
# bug on Windows (``expected 'INT_t' but got 'long'``): under numpy<2 it
# raises a clean ValueError, but under numpy>=2 it corrupts memory and hard
# aborts the process. Use the ``ecos`` conic solver instead on Windows
# (requires ``pip install ecos``); leave the fast default elsewhere.
_SOLVER_WRAPPER = "ecos" if sys.platform == "win32" else "seidel"


class PiecewisePolyTOPPRA(object):
    """TOPPRA-based time-optimal piecewise-polynomial interpolator.

    Drop-in replacement for ``wrs.motion.trajectory.piecewisepoly_toppra``.
    """

    def __init__(self):
        pass

    def _remove_duplicate(self, path):
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        new_path.append(path[-1])
        return new_path

    def interpolate_by_max_spdacc(
        self,
        path,
        ctrl_freq=0.005,
        max_vels=None,
        max_accs=None,
        toggle_debug=False,
    ):
        """Time-parametrise ``path`` under joint velocity/accel limits.

        Parameters
        ----------
        path : array_like, shape (N, n_joints)
            Waypoints to traverse in order.
        ctrl_freq : float
            Output sample period in seconds.
        max_vels, max_accs : array_like, optional
            Per-joint velocity/acceleration limits. Defaults to
            ``2*pi/3`` rad/s and ``pi`` rad/s^2 respectively.
        toggle_debug : bool
            When True, plot the resulting profiles with matplotlib.

        Returns
        -------
        numpy.ndarray, shape (M, n_joints)
            Densely sampled configurations along the optimal trajectory.
        """
        path = self._remove_duplicate(path)
        self._path_array = np.array(path)
        self._n_pnts, _ = self._path_array.shape
        if max_vels is None:
            max_vels = [math.pi * 2 / 3] * path[0].shape[0]
        if max_accs is None:
            max_accs = [math.pi] * path[0].shape[0]
        max_vels = np.asarray(max_vels)
        max_accs = np.asarray(max_accs)
        # seed time intervals from the slowest joint in each segment
        time_intervals = []
        for i in range(self._n_pnts - 1):
            pose_diff = abs(path[i + 1] - path[i])
            tmp_time_interval = np.max(pose_diff / max_vels)
            time_intervals.append(tmp_time_interval)
        time_intervals = np.array(time_intervals)
        x = [0]
        tmp_total_x = 0
        for i in range(len(time_intervals)):
            tmp_time_interval = time_intervals[i]
            x.append(tmp_time_interval + tmp_total_x)
            tmp_total_x += tmp_time_interval
        interpolated_path = ta.SplineInterpolator(x, path)
        pc_vel = constraint.JointVelocityConstraint(max_vels)
        pc_acc = constraint.JointAccelerationConstraint(max_accs)
        instance = algo.TOPPRA(
            [pc_vel, pc_acc], interpolated_path,
            solver_wrapper=_SOLVER_WRAPPER,
        )
        jnt_traj = instance.compute_trajectory()
        duration = jnt_traj.duration
        ts = np.linspace(0, duration, math.ceil(duration / ctrl_freq))
        interpolated_confs = jnt_traj.eval(ts)
        if toggle_debug:
            import matplotlib.pyplot as plt

            interpolated_spds = jnt_traj.evald(ts)
            interpolated_accs = jnt_traj.evaldd(ts)
            fig, axs = plt.subplots(3, figsize=(10, 30))
            fig.tight_layout(pad=0.7)
            axs[0].plot(ts, interpolated_confs, "o")
            axs[1].plot(ts, interpolated_spds)
            for ys in max_vels:
                axs[1].axhline(y=ys)
                axs[1].axhline(y=-ys)
            axs[2].plot(ts, interpolated_accs)
            for ys in max_accs:
                axs[2].axhline(y=ys)
                axs[2].axhline(y=-ys)
            plt.show()
        return interpolated_confs
