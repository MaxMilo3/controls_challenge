"""
edmd_koopman.py
---------------
Constructs a Koopman linear model (A, B, C) from a collection of trajectory
CSV files using Extended Dynamic Mode Decomposition (EDMD), with:

  - Trajectory-level train/test split (no temporal leakage)
  - K-fold cross-validation over trajectories
  - Expanded nonlinear dictionary (state-only, no lifted inputs)
  - One-step and multi-step prediction error reporting

All data is preprocessed to match tinyphysics.py's get_data():
    roll_lataccel   = sin(roll) * 9.81
    steer_command   = -steerCommand

State:   x = [v, a, r, l]
           v = v_ego
           a = a_ego
           r = roll_lataccel
           l = target_lataccel

Control: u = steer_command

Dictionary  psi(x_k, x_{k-1}):
    [1,
     v,  a,  r,  l,                   # base states
     l_prev,                           # 1-step delay on l
     l^2,  v*l,                        # original nonlinear terms
     r*l,                              # roll x lataccel coupling
     l - l_prev,                       # lataccel rate (finite difference)
     a*l,                              # longitudinal-lateral coupling
     v^2,                              # quadratic speed
     r^2,                              # nonlinear roll
     l^3,                              # cubic lataccel (saturation)
     v*r ]                             # speed-dependent roll sensitivity

Usage:
    python edmd_koopman.py --data_dir /data --max_files 20000 \\
                           --out_dir ./controllers --n_folds 5
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ACC_G = 9.81

import builtins
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Data preprocessing  — mirrors tinyphysics.py get_data() exactly
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'v':  df['vEgo'].values,
        'a':  df['aEgo'].values,
        'r':  np.sin(df['roll'].values) * ACC_G,
        'l':  df['targetLateralAcceleration'].values,
        'u': -df['steerCommand'].values,
    })


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

# Normalisation constants — keeps all observables O(1) in magnitude,
# preventing large-scale terms (v*l, v^2) from dominating the regression
# and diluting B[l].  Based on typical dataset values.
V_NORM = 33.5   # typical v_ego (m/s)
L_NORM = 1.0    # lateral accel scale (m/s^2) — keep as 1 for interpretability

OBSERVABLE_NAMES = [
    "1",
    "v/V",  "a",    "r",    "l",
    "l_prev",
    "l^2",  "v*l/V",
    "r*l",
    "dl",                   # l - l_prev  (lataccel rate)
    "a*l",
    "(v/V)^2",
    "r^2",
    "l^3",
    "v*r/V",
]

N_OBS = len(OBSERVABLE_NAMES)   # 15


def lift_batch(X: np.ndarray, X_prev: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    X      : (M, 4)  [v, a, r, l]   current state
    X_prev : (M, 4)  [v, a, r, l]   previous state

    Returns
    -------
    Z : (M, 15)

    All cross/polynomial terms are normalised by V_NORM so every observable
    stays O(1) in magnitude.  This prevents v*l (~33x larger than l) from
    dominating the EDMD regression and diluting B[l].
    """
    v  = X[:, 0];  a = X[:, 1];  r = X[:, 2];  l = X[:, 3]
    lp = X_prev[:, 3]
    vn = v / V_NORM        # normalised velocity, O(1)

    return np.column_stack([
        np.ones(len(X)),   # 1
        vn, a, r, l,       # base states (v normalised)
        lp,                # l_prev
        l**2,              # l^2
        vn * l,            # v*l/V   (was 33x too large)
        r * l,             # r*l
        l - lp,            # dl  (lataccel rate)
        a * l,             # a*l
        vn**2,             # (v/V)^2 (was 1133x too large)
        r**2,              # r^2
        l**3,              # l^3
        vn * r,            # v*r/V
    ])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REQUIRED_COLS = {'vEgo', 'aEgo', 'roll', 'targetLateralAcceleration', 'steerCommand'}


def load_trajectory(filepath: str):
    """
    Load one CSV and return (Z_now, Z_next, U_k) snapshot arrays,
    or None if the file is invalid.

    Only rows where steerCommand is available are used (first ~100 per file).
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [WARN] Could not read {filepath}: {e}")
        return None

    if not REQUIRED_COLS.issubset(df.columns):
        print(f"  [WARN] Missing columns in {filepath}")
        return None

    proc = preprocess(df)
    proc = proc[proc['u'].notna()].reset_index(drop=True)   # keep steered rows only

    if len(proc) < 4:
        return None

    X = proc[['v', 'a', 'r', 'l']].values
    U = proc[['u']].values

    idx_prev = np.arange(0, len(X) - 2)
    idx_curr = np.arange(1, len(X) - 1)
    idx_next = np.arange(2, len(X)    )

    Z_now  = lift_batch(X[idx_curr], X[idx_prev])
    Z_next = lift_batch(X[idx_next], X[idx_curr])
    U_k    = U[idx_curr]

    return Z_now, Z_next, U_k


# ---------------------------------------------------------------------------
# EDMD regression
# ---------------------------------------------------------------------------

def edmd(Z_now, Z_next, U, ridge_alpha: float = 1e-4):
    """
    Solve [A B] via ridge-regularised least squares.
    Returns A, B, C, rms_residual.
    """
    N = Z_now.shape[1]
    p = U.shape[1]

    Phi = np.hstack([Z_now, U])
    PtP = Phi.T @ Phi + ridge_alpha * np.eye(N + p)
    PtZ = Phi.T @ Z_next
    Theta = np.linalg.solve(PtP, PtZ)

    AB = Theta.T
    A  = AB[:, :N]
    B  = AB[:, N:]

    # Trivial reconstruction: x=[v,a,r,l] are at z indices 1,2,3,4
    C = np.zeros((4, N))
    for i in range(4):
        C[i, i + 1] = 1.0

    rms = float(np.sqrt(np.mean((Z_next - (Z_now @ A.T + U @ B.T))**2)))
    return A, B, C, rms


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def one_step_error(A, B, Z_now, Z_next, U):
    """RMS one-step prediction error in lifted space."""
    Z_pred = Z_now @ A.T + U @ B.T
    return float(np.sqrt(np.mean((Z_next - Z_pred)**2)))


def multistep_error(A, B, Z_now, Z_next_all_steps, U_seq, n_steps=5):
    """
    Roll out the model n_steps ahead and report RMS error vs true next state.
    Z_next_all_steps: list of Z_next arrays for steps 1..n_steps
    U_seq           : list of U arrays aligned with each step
    Only uses the first M - n_steps snapshots to keep indexing clean.
    """
    M = len(Z_now)
    if M <= n_steps:
        return float('nan')

    # Predict n_steps ahead from each starting point
    Z_hat = Z_now[:-n_steps].copy()
    for k in range(n_steps):
        Z_hat = Z_hat @ A.T + U_seq[k][:-n_steps] @ B.T

    Z_true = Z_next_all_steps[n_steps - 1][:-n_steps]
    return float(np.sqrt(np.mean((Z_true - Z_hat)**2)))


def lataccel_reconstruction_error(A, B, C, Z_now, Z_next, U):
    """RMS error on the reconstructed lateral acceleration specifically."""
    Z_pred = Z_now @ A.T + U @ B.T
    l_pred = Z_pred @ C[3]          # C row 3 reconstructs l (index 4 in z -> C col 4)
    l_true = Z_next[:, 4]           # l is at index 4 in the dictionary
    return float(np.sqrt(np.mean((l_true - l_pred)**2)))


# ---------------------------------------------------------------------------
# K-fold cross-validation over trajectories
# ---------------------------------------------------------------------------

def kfold_cv(all_results: list, n_folds: int, ridge: float, seed: int = 42):
    """
    Perform k-fold CV by splitting trajectory-level snapshot groups.

    all_results : list of (Z_now, Z_next, U_k) per trajectory
    Returns dict of mean/std for one-step and lataccel errors.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_results))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    one_step_errors = []
    lataccel_errors = []

    print(f"\n--- {n_folds}-fold cross-validation ({len(all_results)} trajectories) ---")
    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        def stack(idx_list):
            Zn  = np.vstack([all_results[i][0] for i in idx_list])
            Znx = np.vstack([all_results[i][1] for i in idx_list])
            Uk  = np.vstack([all_results[i][2] for i in idx_list])
            return Zn, Znx, Uk

        Zn_tr, Znx_tr, U_tr = stack(train_idx)
        Zn_te, Znx_te, U_te = stack(test_idx)

        A, B, C, _ = edmd(Zn_tr, Znx_tr, U_tr, ridge_alpha=ridge)

        ose = one_step_error(A, B, Zn_te, Znx_te, U_te)
        lae = lataccel_reconstruction_error(A, B, C, Zn_te, Znx_te, U_te)

        one_step_errors.append(ose)
        lataccel_errors.append(lae)
        print(f"  Fold {fold_idx+1}/{n_folds}:  "
              f"one-step RMS = {ose:.5f}  |  lataccel RMS = {lae:.5f}")

    results = {
        'one_step_mean':  float(np.mean(one_step_errors)),
        'one_step_std':   float(np.std(one_step_errors)),
        'lataccel_mean':  float(np.mean(lataccel_errors)),
        'lataccel_std':   float(np.std(lataccel_errors)),
    }
    print(f"\n  One-step  RMS: {results['one_step_mean']:.5f} +/- {results['one_step_std']:.5f}")
    print(f"  Lataccel  RMS: {results['lataccel_mean']:.5f} +/- {results['lataccel_std']:.5f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDMD Koopman identification")
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--max_files",  type=int, default=20000)
    parser.add_argument("--ridge",      type=float, default=1e-4)
    parser.add_argument("--out_dir",    type=str, default="controllers")
    parser.add_argument("--test_frac",  type=float, default=0.2,
                        help="Fraction of trajectories held out as test set")
    parser.add_argument("--n_folds",    type=int, default=5,
                        help="Number of CV folds (set to 0 to skip CV)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # ---- Discover files ----------------------------------------------------
    data_dir = Path(args.data_dir)
    print(f"\n--- Path diagnostics ---")
    print(f"Looking in : {data_dir.resolve()}")
    print(f"Exists     : {data_dir.exists()}")
    if not data_dir.exists():
        print("ERROR: directory does not exist.")
        return

    entries = list(data_dir.iterdir())
    print(f"Contents   : {len(entries)} entries")
    for e in entries[:10]:
        print(f"  {e.name}{'/' if e.is_dir() else ''}")
    if len(entries) > 10:
        print(f"  ... and {len(entries) - 10} more")
    print("------------------------\n")

    files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        print("No CSVs at top level — searching recursively ...")
        files = sorted(glob.glob(str(data_dir / "**" / "*.csv"), recursive=True))
    files = files[:args.max_files]

    if not files:
        print("No CSV files found.")
        return
    print(f"Found {len(files)} files.  First: {files[0]}  Last: {files[-1]}\n")

    # ---- Load all trajectories ---------------------------------------------
    all_results = []
    for i, fpath in enumerate(files):
        result = load_trajectory(fpath)
        if result is not None:
            all_results.append(result)
        if (i + 1) % 500 == 0:
            print(f"  Loaded {i+1}/{len(files)} files  |  valid: {len(all_results)}")

    print(f"\nValid trajectories: {len(all_results)} / {len(files)}")
    if not all_results:
        print("No valid data.")
        return

    total_snaps = sum(len(r[0]) for r in all_results)
    print(f"Total snapshots   : {total_snaps}")
    print(f"Dictionary size   : {N_OBS}  ({', '.join(OBSERVABLE_NAMES)})")

    # ---- Trajectory-level train/test split ---------------------------------
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(all_results))
    rng.shuffle(idx)

    n_test  = max(1, int(len(idx) * args.test_frac))
    n_train = len(idx) - n_test
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    print(f"\nTrain trajectories: {n_train}  |  Test trajectories: {n_test}")

    def stack(index_list):
        Zn  = np.vstack([all_results[i][0] for i in index_list])
        Znx = np.vstack([all_results[i][1] for i in index_list])
        Uk  = np.vstack([all_results[i][2] for i in index_list])
        return Zn, Znx, Uk

    Zn_train, Znx_train, U_train = stack(train_idx)
    Zn_test,  Znx_test,  U_test  = stack(test_idx)

    print(f"Train snapshots   : {len(Zn_train)}")
    print(f"Test  snapshots   : {len(Zn_test)}")

    # ---- Cross-validation on training set ----------------------------------
    if args.n_folds > 1:
        train_results = [all_results[i] for i in train_idx]
        cv_stats = kfold_cv(train_results, n_folds=args.n_folds,
                            ridge=args.ridge, seed=args.seed)
    else:
        print("\n(CV skipped — n_folds < 2)")

    # ---- Final model: train on ALL training data ---------------------------
    print(f"\n--- Final model (ridge={args.ridge}) ---")
    A, B, C, train_rms = edmd(Zn_train, Znx_train, U_train, ridge_alpha=args.ridge)
    print(f"Train lifted RMS  : {train_rms:.6f}")

    test_ose = one_step_error(A, B, Zn_test, Znx_test, U_test)
    test_lae = lataccel_reconstruction_error(A, B, C, Zn_test, Znx_test, U_test)
    print(f"Test  one-step RMS: {test_ose:.6f}")
    print(f"Test  lataccel RMS: {test_lae:.6f}")

    # ---- Eigenvalue report -------------------------------------------------
    eigs = np.abs(np.linalg.eigvals(A))
    print(f"\nEigenvalue magnitudes (sorted desc): {np.sort(eigs)[::-1]}")
    n_unstable = int(np.sum(eigs > 1.0))
    if n_unstable:
        print(f"  WARNING: {n_unstable} eigenvalue(s) outside unit disk")
    else:
        print("  All eigenvalues stable (inside unit disk)")

    print(f"\nA {A.shape}:\n{np.array2string(A, precision=4, suppress_small=True)}")
    print(f"\nB {B.shape}:\n{np.array2string(B, precision=4, suppress_small=True)}")

    # ---- Save --------------------------------------------------------------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "A.npy", A)
    np.save(out / "B.npy", B)
    np.save(out / "C.npy", C)
    np.save(out / "Z_now.npy",  Zn_train)
    np.save(out / "Z_next.npy", Znx_train)
    np.save(out / "U.npy",      U_train)
    print(f"\nSaved A.npy, B.npy, C.npy  ->  {out.resolve()}")


if __name__ == "__main__":
    main()