#!/usr/bin/env python3
"""
plot_vs_results.py
==================
Offline analysis and plotting for LQR Visual Servoing results.

Usage:
    python3 plot_vs_results.py [path/to/vs_YYYYMMDD_HHMMSS.csv]

If no path is given, the most recent CSV in ~/vs_logs/ is used.

Plots produced:
  1. Position error components (ex, ey, ez) vs time
  2. Error norm ‖e‖ vs time — shows convergence behaviour
  3. Control velocity components (vx, vy, vz) vs time
  4. Phase-timeline bar showing PRE_GRASP / GRASP / POST_GRASP / PLACE
  5. Lyapunov function V(e) = eᵀPe vs time — proves asymptotic stability
"""

import sys
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg" if sys.platform != "linux" else "Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── LQR matrices (must match lqr_visual_servoing.py) ─────────────────────────
from scipy.linalg import solve_discrete_are

DT     = 0.10
Q_DIAG = [10.0, 10.0, 15.0]
R_DIAG = [1.0,  1.0,  1.0]

A = np.eye(3)
B = DT * np.eye(3)
Q = np.diag(Q_DIAG)
R = np.diag(R_DIAG)
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

# ── Phase colours ─────────────────────────────────────────────────────────────
PHASE_COLORS = {
    "PRE_GRASP":  "#4CAF50",
    "GRASP":      "#2196F3",
    "CLOSE_GRIP": "#FF9800",
    "POST_GRASP": "#9C27B0",
    "PLACE":      "#F44336",
    "OPEN_GRIP":  "#795548",
    "WAIT":       "#9E9E9E",
    "DONE":       "#607D8B",
}


def load_csv(path: Path):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "t":      float(r["t_s"]),
                "ex":     float(r["ex_m"]),
                "ey":     float(r["ey_m"]),
                "ez":     float(r["ez_m"]),
                "e_norm": float(r["err_norm_m"]),
                "vx":     float(r["vx_ms"]),
                "vy":     float(r["vy_ms"]),
                "vz":     float(r["vz_ms"]),
                "v_norm": float(r["v_norm_ms"]),
                "phase":  r["phase"],
            })
    return rows


def lyapunov(e_vec: np.ndarray) -> float:
    """V(e) = eᵀ P e — LQR cost-to-go Lyapunov function."""
    return float(e_vec @ P @ e_vec)


def plot_results(rows: list[dict], csv_path: Path):
    if not rows:
        print("No data to plot.")
        return

    t      = np.array([r["t"]      for r in rows])
    ex     = np.array([r["ex"]     for r in rows])
    ey     = np.array([r["ey"]     for r in rows])
    ez     = np.array([r["ez"]     for r in rows])
    e_norm = np.array([r["e_norm"] for r in rows])
    vx     = np.array([r["vx"]     for r in rows])
    vy     = np.array([r["vy"]     for r in rows])
    vz     = np.array([r["vz"]     for r in rows])
    v_norm = np.array([r["v_norm"] for r in rows])
    phases = [r["phase"] for r in rows]
    V_lya  = np.array([lyapunov(np.array([ex[i], ey[i], ez[i]])) for i in range(len(t))])

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(
        f"LQR Visual Servoing — Results Analysis\n"
        f"Q = diag{Q_DIAG}   R = diag{R_DIAG}   dt = {DT} s\n"
        f"K (diag) = {np.diag(K).round(3).tolist()}",
        fontsize=10,
    )

    # ── 1. Error components ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, ex * 1000, label="eₓ", color="red")
    ax.plot(t, ey * 1000, label="eᵧ", color="green")
    ax.plot(t, ez * 1000, label="e_z", color="blue")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Position error (mm)")
    ax.set_title("Task-space error components  e = p_ee − p_des")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    _shade_phases(ax, t, phases)

    # ── 2. Error norm ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, e_norm * 1000, color="black", lw=2, label="‖e‖")
    ax.axhline(8, color="orange", ls="--", lw=1, label="PRE_GRASP threshold (8 mm)")
    ax.axhline(5, color="red",    ls="--", lw=1, label="GRASP threshold (5 mm)")
    ax.set_ylabel("‖e‖ (mm)")
    ax.set_title("Error norm — convergence behaviour")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _shade_phases(ax, t, phases)

    # ── 3. Control velocities ────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, vx * 100, label="vₓ", color="red")
    ax.plot(t, vy * 100, label="vᵧ", color="green")
    ax.plot(t, vz * 100, label="v_z", color="blue")
    ax.plot(t, v_norm * 100, label="‖v‖", color="black", lw=1.5, ls="--")
    ax.set_ylabel("Velocity (cm/s)")
    ax.set_title("LQR optimal control  u* = −K e")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    _shade_phases(ax, t, phases)

    # ── 4. Lyapunov function ─────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, V_lya, color="purple", lw=2)
    ax.set_ylabel("V(e) = eᵀPe")
    ax.set_title("Lyapunov function — monotone decrease confirms closed-loop stability")
    ax.grid(True, alpha=0.3)
    _shade_phases(ax, t, phases)

    # ── 5. Phase timeline ────────────────────────────────────────────────────
    ax = axes[4]
    prev_phase = phases[0]
    seg_start  = t[0]
    for i in range(1, len(t)):
        if phases[i] != prev_phase or i == len(t) - 1:
            color = PHASE_COLORS.get(prev_phase, "#aaaaaa")
            ax.axvspan(seg_start, t[i], alpha=0.6, color=color, label=prev_phase)
            prev_phase = phases[i]
            seg_start  = t[i]
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase")
    ax.set_title("Servoing phase timeline")
    ax.set_yticks([])
    # Legend deduplicated
    handles = [mpatches.Patch(color=c, label=p, alpha=0.7)
               for p, c in PHASE_COLORS.items() if p in set(phases)]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=3)

    plt.tight_layout()

    # Save alongside CSV
    png_path = csv_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {png_path}")

    # Print summary statistics
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  Duration          : {t[-1]:.2f} s")
    print(f"  Final ‖e‖         : {e_norm[-1]*1000:.2f} mm")
    print(f"  Min ‖e‖           : {e_norm.min()*1000:.2f} mm")
    print(f"  Mean control ‖v‖  : {v_norm.mean()*100:.2f} cm/s")
    print(f"  Max control ‖v‖   : {v_norm.max()*100:.2f} cm/s")
    lqr_cost = sum((np.array([r["ex"],r["ey"],r["ez"]]) @ Q @ np.array([r["ex"],r["ey"],r["ez"]])
                    + np.array([r["vx"],r["vy"],r["vz"]]) @ R @ np.array([r["vx"],r["vy"],r["vz"]]))
                   for r in rows)
    print(f"  Cumulative LQR cost J = {lqr_cost:.4f}")
    print(f"  LQR gain K (diag) : {np.diag(K).round(4).tolist()}")
    print(f"  Closed-loop eigs  : {np.linalg.eigvals(np.eye(3) - DT * K).round(4).tolist()}")
    print("─────────────────────────────────────────────────────────────\n")

    plt.show()


def _shade_phases(ax, t: np.ndarray, phases: list[str]):
    """Shade background by phase on any axis."""
    prev = phases[0]
    s    = t[0]
    for i in range(1, len(t)):
        if phases[i] != prev or i == len(t) - 1:
            c = PHASE_COLORS.get(prev, "#aaaaaa")
            ax.axvspan(s, t[i], alpha=0.08, color=c)
            prev = phases[i]
            s    = t[i]


def find_latest_csv() -> Path:
    log_dir = Path.home() / "vs_logs"
    csvs = sorted(log_dir.glob("vs_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No vs_*.csv files found in {log_dir}")
    return csvs[-1]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = find_latest_csv()
        print(f"Using latest log: {csv_path}")

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} samples from {csv_path.name}")
    plot_results(rows, csv_path)
