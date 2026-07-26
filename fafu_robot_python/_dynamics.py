"""Pure rigid-body dynamics and kinematics support.

The high-level controller owns hardware state and safety policy.  This module
only owns a Pinocchio model and deterministic numerical operations, so it can
be tested and maintained without serial or motor-driver dependencies.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

if __package__:
    from ._api_types import FrictionParams
else:  # pragma: no cover - legacy direct-module import
    from _api_types import FrictionParams

try:
    import pinocchio as pin  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pin = None  # type: ignore


def resolve_urdf_path(
    package_dir: str,
    urdf_path: Optional[str],
) -> Optional[str]:
    """Resolve an explicit URDF or the first vendored description."""

    if urdf_path:
        return urdf_path if os.path.exists(urdf_path) else None

    description_dir = os.path.join(package_dir, "fafu_robot_description")
    if not os.path.isdir(description_dir):
        return None
    for filename in sorted(os.listdir(description_dir)):
        candidate = os.path.join(description_dir, filename)
        if filename.endswith(".urdf") and os.path.isfile(candidate):
            return candidate
    return None


def friction_compensation(
    velocity: Iterable[float],
    params: FrictionParams,
) -> np.ndarray:
    """Return Coulomb plus viscous friction torque in Nm."""

    vel = np.asarray(list(velocity), dtype=float)
    fc = np.asarray(params.fc, dtype=float)
    fv = np.asarray(params.fv, dtype=float)
    if fc.shape != vel.shape or fv.shape != vel.shape:
        raise ValueError(
            f"friction fc/fv length {fc.shape}/{fv.shape} != velocity "
            f"length {vel.shape}"
        )
    threshold = float(params.vel_threshold)
    if (
        not np.all(np.isfinite(vel))
        or not np.all(np.isfinite(fc))
        or not np.all(np.isfinite(fv))
        or not np.isfinite(threshold)
        or threshold < 0.0
    ):
        raise ValueError(
            "friction velocity, coefficients, and threshold must be finite; "
            "threshold must be non-negative"
        )

    viscous = fv * vel
    full = fc * np.sign(vel) + viscous
    return np.where(np.abs(vel) < threshold, viscous, full)


class DynamicsModel:
    """Thread-safe numerical model for one robot configuration."""

    def __init__(
        self,
        model,
        *,
        gravity_vector: Iterable[float],
        eef_frame: Optional[str],
    ) -> None:
        self.model = model
        self.data = model.createData()
        self.gravity_vector = np.asarray(list(gravity_vector), dtype=float)
        if self.gravity_vector.shape != (3,):
            raise ValueError("gravity_vec must have exactly 3 elements")
        if not np.all(np.isfinite(self.gravity_vector)):
            raise ValueError("gravity_vec must contain only finite values")

        self.eef_frame_id, self.eef_frame_name = self._resolve_eef_frame(
            eef_frame
        )
        self._lock = threading.RLock()

    @classmethod
    def load(
        cls,
        urdf_path: str,
        *,
        num_joints: int,
        gravity_vector: Iterable[float],
        eef_frame: Optional[str] = None,
    ) -> "DynamicsModel":
        if pin is None:
            raise RuntimeError(
                "pinocchio is not installed; install it from conda-forge "
                "(recommended) or pip before calling setup_dynamics()"
            )
        try:
            model = pin.buildModelFromUrdf(urdf_path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Pinocchio model from {urdf_path!r}: {exc}"
            ) from exc
        if model.nq != num_joints or model.nv != num_joints:
            raise RuntimeError(
                f"URDF has nq={model.nq}, nv={model.nv}, but controller has "
                f"{num_joints} joints; only fixed-base 1-DoF joint chains "
                "are supported"
            )
        return cls(
            model,
            gravity_vector=gravity_vector,
            eef_frame=eef_frame,
        )

    def _resolve_eef_frame(self, requested: Optional[str]) -> Tuple[int, str]:
        candidates = []
        if requested:
            candidates.append(requested)
        candidates.append("tool_link")
        try:
            candidates.append(self.model.names[self.model.njoints - 1])
        except Exception:
            pass

        for name in candidates:
            if self.model.existFrame(name):
                return self.model.getFrameId(name), name

        frame_id = self.model.nframes - 1
        return frame_id, self.model.frames[frame_id].name

    @staticmethod
    def rotation_matrix(
        rotation,
        *,
        is_euler: bool,
        is_radians: bool,
    ) -> np.ndarray:
        if pin is None:  # pragma: no cover - guarded by model construction
            raise RuntimeError("pinocchio is not installed")
        if rotation is None:
            return np.eye(3)
        value = np.asarray(rotation, dtype=float)
        if is_euler or value.shape == (3,):
            rpy = value.reshape(3)
            if not is_radians:
                rpy = np.deg2rad(rpy)
            return np.asarray(
                pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2]),
                dtype=float,
            )
        if value.shape != (3, 3):
            raise ValueError(
                "rot must be a 3x3 rotation matrix or a length-3 RPY triple"
            )
        return value

    def _forward_se3(self, q_rad: np.ndarray):
        pin.forwardKinematics(self.model, self.data, q_rad)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.eef_frame_id]

    def forward_kinematics(self, q_rad: Iterable[float]) -> Dict[str, object]:
        q = np.asarray(list(q_rad), dtype=float)
        with self._lock:
            pose = self._forward_se3(q)
            position = np.asarray(pose.translation, dtype=float).copy()
            rotation = np.asarray(pose.rotation, dtype=float).copy()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return {
            "position": position,
            "rotation": rotation,
            "rpy": np.asarray(pin.rpy.matrixToRpy(rotation), dtype=float),
            "transform": transform,
            "q": q,
        }

    def cartesian_waypoints(
        self,
        start_q: Iterable[float],
        target_position: Iterable[float],
        target_rotation,
        *,
        is_euler: bool,
        rotation_is_radians: bool,
        steps: int,
    ):
        """Interpolate a Cartesian SE(3) geodesic as pose waypoints."""

        rotation = self.rotation_matrix(
            target_rotation,
            is_euler=is_euler,
            is_radians=rotation_is_radians,
        )
        position = np.asarray(list(target_position), dtype=float).reshape(3)
        goal = pin.SE3(rotation, position)
        q = np.asarray(list(start_q), dtype=float)

        count = max(1, int(steps))
        with self._lock:
            start = self._forward_se3(q)
            relative = pin.log6(start.actInv(goal))
            result = []
            for index in range(1, count + 1):
                pose = start * pin.exp6(relative * (index / count))
                result.append(
                    (
                        np.asarray(pose.translation, dtype=float).copy(),
                        np.asarray(pose.rotation, dtype=float).copy(),
                    )
                )
        return result

    def inverse_kinematics(
        self,
        target_position: Iterable[float],
        target_rotation,
        *,
        current_q: Iterable[float],
        init_q: Optional[Iterable[float]],
        is_euler: bool,
        rotation_is_radians: bool,
        max_iter: int,
        eps: float,
        damping: float,
        adaptive_damping: bool,
        multi_init: bool,
        num_attempts: int,
        limits: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[np.ndarray]:
        rotation = self.rotation_matrix(
            target_rotation,
            is_euler=is_euler,
            is_radians=rotation_is_radians,
        )
        position = np.asarray(list(target_position), dtype=float).reshape(3)
        target = pin.SE3(rotation, position)
        current = np.asarray(list(current_q), dtype=float)

        with self._lock:
            if not multi_init:
                seed = (
                    current
                    if init_q is None
                    else np.asarray(list(init_q), dtype=float)
                )
                return self._ik_single(
                    target,
                    seed,
                    max_iter,
                    eps,
                    damping,
                    adaptive_damping,
                    limits,
                )

            seeds = [current, np.zeros(self.model.nq)]
            if limits is not None:
                lo, hi = limits
                seeds.append((lo + hi) / 2.0)
                rng = np.random.default_rng()
                while len(seeds) < num_attempts:
                    seeds.append(rng.uniform(lo, hi))
            else:
                rng = np.random.default_rng()
                while len(seeds) < num_attempts:
                    seeds.append(
                        rng.uniform(-np.pi / 4, np.pi / 4, self.model.nq)
                    )

            best_q = None
            best_error = float("inf")
            for seed in seeds[:num_attempts]:
                q = self._ik_single(
                    target,
                    seed,
                    max_iter,
                    eps,
                    damping,
                    adaptive_damping,
                    limits,
                )
                if q is None:
                    continue
                actual = np.asarray(
                    self._forward_se3(q).translation,
                    dtype=float,
                )
                error = float(np.linalg.norm(actual - position))
                if error < best_error:
                    best_error, best_q = error, q
                if error < eps:
                    return q
            return best_q

    def _ik_single(
        self,
        target,
        seed: np.ndarray,
        max_iter: int,
        eps: float,
        damping: float,
        adaptive_damping: bool,
        limits: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[np.ndarray]:
        q = np.asarray(seed, dtype=float).copy()
        dt = 1e-1
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            error_transform = self.data.oMf[self.eef_frame_id].actInv(target)
            error = pin.log(error_transform).vector
            error_norm = float(np.linalg.norm(error))
            if error_norm < eps:
                return q

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.eef_frame_id,
                pin.LOCAL,
            )
            jacobian = -pin.Jlog6(error_transform.inverse()).dot(jacobian)
            lam = (
                damping * (1.0 + 1.0 / (error_norm + 0.1))
                if adaptive_damping
                else damping
            )
            system = jacobian.dot(jacobian.T) + (lam**2) * np.eye(6)
            try:
                alpha = np.linalg.solve(system, error)
            except np.linalg.LinAlgError:
                return None
            velocity = -jacobian.T.dot(alpha)
            norm = float(np.linalg.norm(velocity))
            if norm > 10.0:
                velocity *= 10.0 / norm
            q = pin.integrate(self.model, q, velocity * dt)
            if limits is not None:
                lo, hi = limits
                if np.any(q < lo) or np.any(q > hi):
                    return None
        return None

    def _with_gravity(self, function):
        original = self.model.gravity.linear.copy()
        self.model.gravity.linear = self.gravity_vector
        try:
            return function()
        finally:
            self.model.gravity.linear = original

    def gravity(self, q: Iterable[float]) -> np.ndarray:
        value = np.asarray(list(q), dtype=float)
        with self._lock:
            result = np.asarray(
                self._with_gravity(
                    lambda: pin.computeGeneralizedGravity(
                        self.model,
                        self.data,
                        value,
                    )
                ),
                dtype=float,
            ).copy()
        return result

    def mass_matrix(self, q: Iterable[float]) -> np.ndarray:
        value = np.asarray(list(q), dtype=float)
        with self._lock:
            matrix = np.asarray(
                pin.crba(self.model, self.data, value),
                dtype=float,
            ).copy()
        size = len(value)
        matrix = matrix[:size, :size]
        return np.triu(matrix) + np.triu(matrix, 1).T

    def coriolis(
        self,
        q: Iterable[float],
        velocity: Iterable[float],
    ) -> np.ndarray:
        q_value = np.asarray(list(q), dtype=float)
        v_value = np.asarray(list(velocity), dtype=float)
        with self._lock:
            matrix = np.asarray(
                pin.computeCoriolisMatrix(
                    self.model,
                    self.data,
                    q_value,
                    v_value,
                ),
                dtype=float,
            ).copy()
        return matrix

    def inverse_dynamics(
        self,
        q: Iterable[float],
        velocity: Iterable[float],
        acceleration: Iterable[float],
    ) -> np.ndarray:
        q_value = np.asarray(list(q), dtype=float)
        v_value = np.asarray(list(velocity), dtype=float)
        a_value = np.asarray(list(acceleration), dtype=float)
        with self._lock:
            result = np.asarray(
                self._with_gravity(
                    lambda: pin.rnea(
                        self.model,
                        self.data,
                        q_value,
                        v_value,
                        a_value,
                    )
                ),
                dtype=float,
            ).copy()
        return result


__all__ = [
    "DynamicsModel",
    "friction_compensation",
    "resolve_urdf_path",
]
