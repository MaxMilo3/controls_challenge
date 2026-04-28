"""
test_koopman_trajectory.py
--------------------------
Tests how well the Koopman model predicts *lateral acceleration* along a
single trajectory, using the logged steer commands as the input sequence.

Two prediction modes:
  1. One-step  : z_{k+1} = A z_k + B u_k  (re-lifted from true state each step)
  2. Multi-step: open-loop rollout from a single starting z, driven by logged u_k
                 (the harder test — shows how fast errors accumulate)

Only lateral acceleration (l = targetLateralAcceleration) is plotted as the
predicted output, since that is the only signal the Koopman controller needs
to track.

Usage:
    python test_koopman_trajectory.py \
        --model_dir ./controllers \
        --csv_file  ./data/00000.csv \
        [--rollout_steps 40] [--save_fig out.png]
"""

import argparse
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── constants (must match edmd_koopman.py) ──────────────────────────────────
ACC_G  = 9.81
V_NORM = 33.5

REQUIRED_COLS = {"vEgo", "aEgo", "roll", "targetLateralAcceleration", "steerCommand"}


# ── preprocessing (mirrors tinyphysics.py get_data) ─────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "v": df["vEgo"].values,
        "a": df["aEgo"].values,
        "r": np.sin(df["roll"].values) * ACC_G,
        "l": df["targetLateralAcceleration"].values,
        "u": -df["steerCommand"].values,
    })


def lift_batch(X: np.ndarray, X_prev: np.ndarray) -> np.ndarray:
    """(M,4) x2 → (M, 15) lifted dictionary."""
    v  = X[:, 0]; a = X[:, 1]; r = X[:, 2]; l = X[:, 3]
    lp = X_prev[:, 3]
    vn = v / V_NORM
    return np.column_stack([
        np.ones(len(X)),
        vn, a, r, l,
        lp,
        l**2, vn * l,
        r * l,
        l - lp,
        a * l,
        vn**2, r**2, l**3,
        vn * r,
    ])


# Dictionary index of l (lateral accel) — matches lift_batch order:
# [1, v/V, a, r, l, l_prev, ...]
#  0   1   2  3  4     5
L_IDX = 4


# ── model loading ────────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    A = np.load(model_dir / "A.npy")
    B = np.load(model_dir / "B.npy")
    C = np.load(model_dir / "C.npy")
    print(f"Model loaded  A{A.shape}  B{B.shape}  from {model_dir.resolve()}")
    eigs = np.abs(np.linalg.eigvals(A))
    n_bad = int(np.sum(eigs > 1.0))
    print(f"Eigenvalues   max|λ|={eigs.max():.4f}  "
          f"({'all stable' if not n_bad else f'WARNING: {n_bad} outside unit disk'})")
    return A, B, C


# ── trajectory loading ───────────────────────────────────────────────────────

def load_trajectory(csv_path: str):
    """
    Returns aligned arrays of length T = len(proc) - 2:
        X_curr  (T, 4)  true state at step k
        X_next  (T, 4)  true state at step k+1   ← ground truth target
        U_k     (T, 1)  logged steer command at step k
        Z_now   (T,15)  lifted state at step k
    """
    df = pd.read_csv(csv_path)
    if not REQUIRED_COLS.issubset(df.columns):
        raise ValueError(f"Missing columns in {csv_path}")

    proc = preprocess(df)
    proc = proc[proc["u"].notna()].reset_index(drop=True)
    if len(proc) < 6:
        raise ValueError(f"Too few usable rows in {csv_path}")

    X = proc[["v", "a", "r", "l"]].values
    U = proc[["u"]].values

    ip  = np.arange(0, len(X) - 2)
    ic  = np.arange(1, len(X) - 1)
    inx = np.arange(2, len(X))

    Z_now = lift_batch(X[ic], X[ip])

    return X[ic], X[inx], U[ic], Z_now

def get_training_window(U_k, min_u=1e-3):
    """
    Returns indices [start, end) corresponding to the portion of the trajectory
    where steering inputs are active (|u| > threshold), plus one step of evolution.
    """
    u_mag = np.abs(U_k.flatten())
    active = np.where(u_mag > min_u)[0]

    if len(active) == 0:
        raise ValueError("No active steering inputs found in trajectory.")

    start = active[0]
    end   = active[-1] + 2   # include one-step evolution

    return start, min(end, len(U_k))


# ── prediction ───────────────────────────────────────────────────────────────

def one_step_lataccel(A, B, Z_now, U_k):
    """
    z_{k+1} = A z_k + B u_k at every step, re-using the true lifted state.
    Returns predicted l at k+1, shape (T,).
    """
    Z_pred = Z_now @ A.T + U_k @ B.T
    return Z_pred[:, L_IDX]


def multistep_lataccel(A, B, z0: np.ndarray, U_seq: np.ndarray):
    """
    Open-loop rollout from z0 driven entirely by logged steering U_seq.
    No re-lifting from true state — any model error compounds.

    Returns predicted l for steps 0..len(U_seq), shape (len(U_seq)+1,).
    Step 0 is the initial condition (no prediction yet).
    """
    z = z0.copy().reshape(1, -1)
    l_preds = [z[0, L_IDX]]
    for u in U_seq:
        z = z @ A.T + u.reshape(1, -1) @ B.T
        l_preds.append(z[0, L_IDX])
    return np.array(l_preds)


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_lataccel(l_true, l_onestep, l_multi, rollout_start, rollout_steps,
                  rms_one, rms_multi, csv_path, save_path=None):

    T    = len(l_true)
    time = np.arange(T)
    rs   = rollout_start
    re   = rs + len(l_multi)

    # ── brighter, clean style ────────────────────────────────────────────
    plt.style.use("default")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.32}
    )

    # ── top: lataccel traces ──────────────────────────────────────────────
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    ax.plot(time, l_true, lw=2.2, label="Ground truth")
    ax.plot(time, l_onestep, lw=1.5, linestyle="--",
            label=f"One-step pred  RMS={rms_one:.4f} m/s²", alpha=0.9)

    # highlight rollout window
    ax.axvspan(rs, re - 1, alpha=0.15)
    ax.axvline(rs, linestyle="--", linewidth=1.0)

    ax.plot(np.arange(rs, re), l_multi, lw=2.0, linestyle=":",
            label=f"{rollout_steps}-step rollout  RMS={rms_multi:.4f} m/s²")

    ax.set_ylabel("Lateral accel (m/s²)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

    # ── bottom: absolute error ────────────────────────────────────────────
    ax2.set_facecolor("white")
    ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    ax2.plot(time, np.abs(l_true - l_onestep),
             lw=1.2, label="|error| one-step")

    l_true_win = l_true[rs:re]
    ax2.plot(np.arange(rs, re), np.abs(l_true_win - l_multi),
             lw=1.5, linestyle=":", label="|error| rollout")

    ax2.set_ylabel("|Error| (m/s²)", fontsize=10)
    ax2.set_xlabel("Time step", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",     type=str, default="controllers")
    parser.add_argument("--csv_file",      type=str, default=None,
                        help="Path to a single trajectory CSV")
    parser.add_argument("--data_dir",      type=str, default=None,
                        help="Pick a random CSV from this directory if --csv_file not set")
    parser.add_argument("--rollout_steps", type=int, default=40,
                        help="Number of open-loop steps for the multi-step rollout")
    parser.add_argument("--rollout_start", type=int, default=None,
                        help="Time step to begin the rollout (default: 1/3 into traj)")
    parser.add_argument("--save_fig",      type=str, default="koopman_lataccel_test.png",
                        help="Output path; leave empty to show interactively")
    parser.add_argument("--seed",          type=int, default=0)
    args = parser.parse_args()

    # ── pick CSV ──────────────────────────────────────────────────────────
    if args.csv_file:
        csv_path = args.csv_file
    elif args.data_dir:
        random.seed(args.seed)
        files = glob.glob(str(Path(args.data_dir) / "**" / "*.csv"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No CSVs found in {args.data_dir}")
        csv_path = random.choice(files)
        print(f"Randomly selected: {csv_path}")
    else:
        raise ValueError("Provide --csv_file or --data_dir")

    # ── load ──────────────────────────────────────────────────────────────
    A, B, C = load_model(Path(args.model_dir))
    X_curr, X_next, U_k, Z_now = load_trajectory(csv_path)

    # ── restrict to training portion (active steering + evolution) ───────────
    ts, te = get_training_window(U_k)

    X_curr = X_curr[ts:te]
    X_next = X_next[ts:te]
    U_k = U_k[ts:te]
    Z_now = Z_now[ts:te]

    T = len(X_curr)
    print(f"Training window steps: {T}  (from original {ts} to {te})")

    # ── one-step ──────────────────────────────────────────────────────────
    l_true    = X_next[:, 3]            # ground truth l at k+1
    l_onestep = one_step_lataccel(A, B, Z_now, U_k)
    rms_one   = float(np.sqrt(np.mean((l_true - l_onestep) ** 2)))
    print(f"\nOne-step lataccel RMS : {rms_one:.6f} m/s²")

    # ── multi-step rollout ────────────────────────────────────────────────
    rs    = args.rollout_start if args.rollout_start is not None else max(1, T // 4)
    rs    = max(0, min(rs, T - 3))
    steps = min(args.rollout_steps, T - rs - 1)

    # Start the rollout from the true lifted state at rs, then free-run
    U_win   = U_k[rs: rs + steps]
    l_multi = multistep_lataccel(A, B, Z_now[rs], U_win)   # shape (steps+1,)

    # Ground truth window: step rs through rs+steps (aligns with l_multi)
    l_true_win = l_true[rs: rs + steps + 1]
    n = min(len(l_true_win), len(l_multi))
    rms_multi = float(np.sqrt(np.mean((l_true_win[:n] - l_multi[:n]) ** 2)))
    print(f"{steps}-step rollout RMS   : {rms_multi:.6f} m/s²")
    print(f"Rollout window        : steps {rs} – {rs + steps}")

    # ── plot ──────────────────────────────────────────────────────────────
    plot_lataccel(
        l_true        = l_true,
        l_onestep     = l_onestep,
        l_multi       = l_multi,
        rollout_start = rs,
        rollout_steps = steps,
        rms_one       = rms_one,
        rms_multi     = rms_multi,
        csv_path      = csv_path,
        save_path     = args.save_fig or None,
    )


if __name__ == "__main__":
    main()
