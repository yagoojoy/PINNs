"""
evaluate.py — Post-training evaluation and visualization

Loads the PDE residual log produced by train.py and the final trained model,
then generates three plots saved to results/:

    pde_residual.png — PDE residual measured before each Phase 2 training step
                       (the actual values that triggered BOOST decisions).
    mse_log.png      — Data MSE comparison: LF baseline vs. PINN prediction.
    imp_rate.png     — PINN improvement rate over the LF baseline (%).

Prerequisites (run train.py first):
    model_phase1.pth, pde_log.npy, pde_threshold.npy
    (model, generator, get_lf_solution_torch, get_pde_loss, device, L_val,
     GRID, T_scale must be defined in the same session or imported.)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# ── Load Phase 2 log from train.py ────────────────────────────────────────────
# pde_log contains per-step (time, pde_error_before_training, is_boost).
# Using pre-training residuals ensures the plot reflects the actual trigger
# condition rather than the post-training (already improved) model state.
pde_log       = np.load("pde_log.npy")
pde_threshold = float(np.load("pde_threshold.npy")[0])

times_phase2 = pde_log['time']
pde_errors   = pde_log['pde_error']
is_boost_arr = pde_log['is_boost']
boost_times  = times_phase2[is_boost_arr]
boost_errors = pde_errors[is_boost_arr]

print(f"Loaded pde_log: {len(pde_log)} steps | threshold: {pde_threshold:.2e}")
print(f"BOOST points ({len(boost_times)}): {boost_times.tolist()}")


def evaluate_and_report():
    """Compute data MSE over t = 0–30s and generate all three result plots."""

    print("\nEvaluating data MSE (t = 0 ~ 30s)...")
    t_eval = np.arange(0, 30.05, 0.5)
    results = []

    model.eval()
    with torch.no_grad():
        # Build fixed spatial grid once; reuse across all time steps
        x = np.linspace(0, L_val, GRID)
        y = np.linspace(0, L_val, GRID)
        X, Y = np.meshgrid(x, y)
        x_in = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
        y_in = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(device)

        for t_val in t_eval:
            t_in = torch.ones(x_in.shape[0], 1, device=device) * t_val / T_scale

            _, _, u_h, _, _ = generator.get_data_at_t(GRID, GRID, t_val)
            u_hf = u_h.flatten()

            u_lf = get_lf_solution_torch(x_in, y_in, t_in, L_val, T_scale)[0].cpu().numpy().flatten()
            u_pred = u_lf + model(x_in, y_in, t_in)[:, 0].cpu().numpy().flatten()

            mse_lf   = np.mean((u_lf   - u_hf) ** 2)
            mse_pinn = np.mean((u_pred  - u_hf) ** 2)
            imp = (mse_lf - mse_pinn) / mse_lf * 100 if mse_lf > 1e-15 else 0.0

            results.append({
                'Time(s)':        round(t_val, 2),
                'Input_MSE_LF':   mse_lf,
                'Model_MSE_PINN': mse_pinn,
                'Improvement_%':  imp,
            })
            if t_val % 5.0 < 0.05:
                print(f"  t={t_val:4.1f}s | LF: {mse_lf:.2e} | PINN: {mse_pinn:.2e} | Imp: {imp:.1f}%")

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "evaluation_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f">> Saved: {csv_path}")

    times_eval    = df['Time(s)'].values
    mse_lf_vals   = df['Input_MSE_LF'].values
    mse_pinn_vals = df['Model_MSE_PINN'].values
    imp_vals      = df['Improvement_%'].values

    # ── Plot 1: PDE Residual (primary diagnostic plot) ────────────────────────
    # Shows the NS residual as measured immediately before each training step —
    # i.e., the exact values used to decide BOOST vs. KEEP.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times_phase2, pde_errors, 'b-', linewidth=2,
            label='PDE residual before training')
    ax.scatter(boost_times, boost_errors,
               color='gold', marker='*', s=200, edgecolor='black',
               label='HF injection (BOOST)', zorder=5)
    ax.axhline(y=pde_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'BOOST threshold ({pde_threshold:.2e})')
    ax.axvline(x=10, color='k', linestyle=':', linewidth=2,
               label='Extrapolation start (t = 10s)')
    ax.axvspan(10, 30, color='orange', alpha=0.05)
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('PDE Residual (log scale)', fontsize=12)
    ax.set_title('PDE Residual vs Time — BOOST Trigger Values', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "pde_residual.png"), dpi=300)
    plt.show()
    print(">> Saved: results/pde_residual.png")

    # ── Plot 2: Data MSE comparison ───────────────────────────────────────────
    # BOOST markers use nearest evaluation time to the logged BOOST time.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times_eval, mse_lf_vals,   'r--', alpha=0.5, label='LF baseline error')
    ax.plot(times_eval, mse_pinn_vals, 'b-',  linewidth=2, label='PINN prediction error')
    for bt in boost_times:
        idx = np.abs(times_eval - bt).argmin()
        ax.scatter(times_eval[idx], mse_pinn_vals[idx],
                   color='gold', marker='*', s=150, edgecolor='black', zorder=5)
    ax.scatter([], [], color='gold', marker='*', s=150,
               edgecolor='black', label='HF injection (BOOST)')
    ax.axvline(x=10, color='k', linestyle=':', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Data MSE: LF Baseline vs. PINN Prediction', fontsize=13)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "mse_log.png"), dpi=300)
    plt.show()
    print(">> Saved: results/mse_log.png")

    # ── Plot 3: Improvement rate ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='Target (95%)')
    ax.plot(times_eval, imp_vals, 'g-', linewidth=2, label='Improvement rate')
    for bt in boost_times:
        idx = np.abs(times_eval - bt).argmin()
        ax.scatter(times_eval[idx], imp_vals[idx],
                   color='gold', marker='*', s=200, edgecolor='black', zorder=5)
    ax.scatter([], [], color='gold', marker='*', s=200,
               edgecolor='black', label='HF injection (BOOST)')
    ax.axvline(x=10, color='k', linestyle=':', linewidth=2)
    ax.axvspan(10, 30, color='orange', alpha=0.05)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('PINN Improvement Rate over LF Baseline', fontsize=13)
    ax.set_ylim(-20, 110)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "imp_rate.png"), dpi=300)
    plt.show()
    print(">> Saved: results/imp_rate.png")


evaluate_and_report()
