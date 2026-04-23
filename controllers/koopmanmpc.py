"""
koopmanmpc.py  —  Koopman MPC Controller
-----------------------------------------
Implements a receding-horizon MPC using the linear Koopman model

    z_{k+1} = A z_k + B u_k
    x_k     = C z_k

identified via EDMD.  Follows the same BaseController interface as pid.py,
compatible with the commaai tinyphysics simulator.

State passed by tinyphysics:
    state.roll_lataccel  =  sin(roll) * 9.81
    state.v_ego          =  longitudinal velocity
    state.a_ego          =  longitudinal acceleration

Dictionary (must match edmd_koopman.py) — 15 observables:
    psi(x_k, x_{k-1}) = [1, v, a, r, l, l_prev, l^2, v*l,
                          r*l, dl, a*l, v^2, r^2, l^3, v*r]

Control  u = steerCommand  (right-positive, clipped to [-2, 2] by simulator)
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from controllers import BaseController

_CONTROLLER_DIR = Path(__file__).resolve().parent

# Must match V_NORM in edmd_koopman.py
V_NORM = 33.5

# Index of lateral-acceleration observable in z
# Dictionary: [1, v/V, a, r, l, l_prev, l^2, v*l/V, r*l, dl, a*l, (v/V)^2, r^2, l^3, v*r/V]
#              0  1    2  3  4  5       6    7       8    9   10   11       12   13   14
L_IDX = 4


# ---------------------------------------------------------------------------
# Lifting function  — must exactly mirror edmd_koopman.py
# ---------------------------------------------------------------------------

def lift(v, a, r, l_now, l_prev):
    dl = l_now - l_prev
    vn = v / V_NORM    # normalised velocity — must match edmd_koopman.py
    return np.array([
        1.0,           # 0
        vn,            # 1  v/V
        a,             # 2
        r,             # 3
        l_now,         # 4  <- L_IDX
        l_prev,        # 5
        l_now ** 2,    # 6
        vn * l_now,    # 7  v*l/V
        r * l_now,     # 8  roll x lataccel coupling
        dl,            # 9  lataccel rate
        a * l_now,     # 10 longitudinal-lateral coupling
        vn ** 2,       # 11 (v/V)^2
        r ** 2,        # 12 nonlinear roll
        l_now ** 3,    # 13 cubic lataccel (saturation)
        vn * r,        # 14 v*r/V
    ], dtype=float)


# ---------------------------------------------------------------------------
# Eigenvalue projection: pull unstable poles inside unit disk
# ---------------------------------------------------------------------------

def stabilise_A(A: np.ndarray, max_eig: float = 0.98) -> np.ndarray:
    """
    Project eigenvalues of A that lie outside the unit disk back to max_eig.
    Uses real Schur decomposition so the result stays real-valued.

    This is necessary when EDMD produces a marginally or actually unstable
    A matrix — without it the H-step rollout in the condensed QP diverges
    and the MPC oscillates wildly.
    """
    from scipy.linalg import schur, rsf2csf, schur as schur_real
    import numpy as np

    # Real Schur form: A = Z T Z^T, T quasi-upper-triangular
    T, Z = schur_real(A, output='real')

    # Work in complex Schur form for clean eigenvalue access
    T_c = T.astype(complex)
    eigs = np.diag(T_c)
    mags = np.abs(eigs)

    # Scale any eigenvalue outside the disk back to max_eig
    scale = np.where(mags > max_eig, max_eig / mags, 1.0)
    T_c = T_c * scale[:, None]   # scale rows (upper triangular stays upper triangular)
    # Diagonal scaling only affects diagonal for upper-triangular structure
    # but off-diagonals also need fixing — just scale diagonal directly
    T_mod = T.copy()
    for i, (s, mag) in enumerate(zip(scale, mags)):
        if mag > max_eig:
            T_mod[i, i] = T[i, i] * s

    A_stable = Z @ T_mod @ Z.T
    n_clipped = int(np.sum(mags > max_eig))
    if n_clipped > 0:
        import builtins
        builtins.print(f"[koopmanmpc] Stabilised {n_clipped} eigenvalue(s) "
                       f"(max |eig| before: {mags.max():.4f}, after: {max_eig})", flush=True)
    return A_stable


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class Controller(BaseController):
    """
    Koopman MPC lateral-acceleration controller.

    Key tuning parameters
    ---------------------
    horizon : int    Prediction steps. Keep short (3-5) if A is near-unstable.
    Q       : float  Tracking weight on lateral accel error.
    R       : float  Control effort weight (larger = smaller commands).
    Rd      : float  Rate weight (larger = smoother, stops sign-flipping).
    stabilise_eigs : bool  Project unstable A eigenvalues inside unit disk.
    max_eig : float  Target spectral radius after stabilisation (< 1).
    """

    def __init__(
        self,
        matrix_dir:     str   = None,
        horizon:        int   = 3,
        Q:              float = 2.0,
        R:              float = 0.0,
        Rd:             float = 8.0,
        u_min:          float = -2.0,
        u_max:          float =  2.0,
        stabilise_eigs: bool  = True,
        max_eig:        float = 0.98,
    ):
        mdir = Path(matrix_dir) if matrix_dir else _CONTROLLER_DIR
        A = np.load(mdir / "A.npy")
        B = np.load(mdir / "B.npy").reshape(A.shape[0], -1)
        self.C = np.load(mdir / "C.npy")

        # Stabilise A before building prediction matrices
        if stabilise_eigs:
            A = stabilise_A(A, max_eig=max_eig)

        self.A = A
        self.B = B
        self.N = A.shape[0]
        self.p = B.shape[1]
        self.H = horizon
        self.Q  = float(Q)
        self.R  = float(R)
        self.Rd = float(Rd)
        self.u_min = u_min
        self.u_max = u_max

        # Delay / rate state
        self.l_prev = 0.0
        self.u_prev = 0.0

        self._build_condensed_matrices()

    # -----------------------------------------------------------------------

    def _build_condensed_matrices(self):
        """
        Build Sx (N*H, N) and Su (N*H, p*H) such that:
            Z_vec = Sx @ z0 + Su @ U_vec
        """
        A, B = self.A, self.B
        N, p, H = self.N, self.p, self.H

        Sx = np.zeros((N * H, N))
        Su = np.zeros((N * H, p * H))

        A_pow = A.copy()
        for i in range(H):
            Sx[i*N:(i+1)*N, :] = A_pow
            for j in range(i + 1):
                Su[i*N:(i+1)*N, j*p:(j+1)*p] = np.linalg.matrix_power(A, i - j) @ B
            A_pow = A_pow @ A

        e_l = np.zeros(N);  e_l[L_IDX] = 1.0
        E_l = np.zeros((H, N * H))
        for i in range(H):
            E_l[i, i*N:(i+1)*N] = e_l

        self.El_Sx = E_l @ Sx   # (H, N)
        self.El_Su = E_l @ Su   # (H, p*H)

        # First-difference operator for rate penalty
        self._D = np.eye(H * p) - np.eye(H * p, k=-1)

    # -----------------------------------------------------------------------

    def _build_reference(self, target_lataccel, future_plan):
        ref = np.full(self.H, target_lataccel, dtype=float)
        if future_plan is not None:
            plan = np.asarray(future_plan.lataccel, dtype=float)
            n = min(len(plan), self.H)
            if n > 0:
                ref[:n] = plan[:n]
        return ref

    # -----------------------------------------------------------------------

    def _solve_qp(self, z0, l_ref):
        H, p = self.H, self.p
        El_Su, El_Sx = self.El_Su, self.El_Sx

        l0 = El_Sx @ z0
        e  = l0 - l_ref

        Q_mat  = self.Q  * np.eye(H)
        R_mat  = self.R  * np.eye(H * p)
        D      = self._D
        Rd_mat = self.Rd * (D.T @ D)

        # Encode previous u into the rate-penalty offset
        d0 = np.zeros(H * p);  d0[0] = self.u_prev

        H_qp = El_Su.T @ Q_mat @ El_Su + R_mat + Rd_mat
        f_qp = El_Su.T @ Q_mat @ e - self.Rd * D.T @ d0

        bounds = [(self.u_min, self.u_max)] * (H * p)

        result = minimize(
            lambda U: 0.5 * U @ H_qp @ U + f_qp @ U,
            np.full(H * p, self.u_prev),   # warm-start from last u
            jac=lambda U: H_qp @ U + f_qp,
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-8, 'maxiter': 200, 'disp': False},
        )

        if result.success:
            return result.x
        try:
            return np.clip(np.linalg.solve(H_qp, -f_qp), self.u_min, self.u_max)
        except np.linalg.LinAlgError:
            return np.full(H * p, self.u_prev)

    # -----------------------------------------------------------------------

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v = float(state.v_ego)
        a = float(state.a_ego)
        r = float(state.roll_lataccel)
        l = float(current_lataccel)

        z0    = lift(v, a, r, l, self.l_prev)
        l_ref = self._build_reference(target_lataccel, future_plan)
        U_opt = self._solve_qp(z0, l_ref)

        u_star = float(np.clip(U_opt[0], self.u_min, self.u_max))

        self.l_prev = l
        self.u_prev = u_star
        return u_star