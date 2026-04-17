import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os

# Manually record time steps where BOOST (HF injection) occurred during Phase 2
manual_boost_times = [14.5, 15.0, 15.5, 16.0, 17.0, 18.5, 20.0, 21.5, 23.0, 25.5, 29.0]

# Directory for saving results
save_dir = "results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def evaluate_and_report(boost_points=[]):
    print(f"Starting detailed evaluation (0s ~ 30s). Manual BOOST points: {boost_points}")

    # Evaluation time steps: 0.1s interval
    t_eval = np.arange(0, 30.05, 0.1)
    results_list = []

    model.eval()

    # Spatial grid for evaluation
    x = np.linspace(0, L_val, GRID); y = np.linspace(0, L_val, GRID)
    X, Y = np.meshgrid(x, y)
    x_in = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_in = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        for t_val in t_eval:
            t_in = torch.tensor(np.full(x_in.shape, t_val), dtype=torch.float32).to(device) / T_scale

            # Ground truth (HF) and LF base at this time step
            _, _, u_h, v_h, p_h = generator.get_data_at_t(GRID, GRID, t_val)
            u_hf_flat = u_h.flatten()
            u_lf_tensor, _, _ = get_lf_solution_torch(x_in, y_in, t_in, L_val, T_scale)
            u_lf_flat = u_lf_tensor.cpu().numpy().flatten()

            # Model prediction: LF + correction
            correction = model(x_in, y_in, t_in)
            u_pred_flat = u_lf_flat + correction[:, 0].cpu().numpy().flatten()

            # MSE and improvement rate
            mse_lf = np.mean((u_lf_flat - u_hf_flat)**2)
            mse_pinn = np.mean((u_pred_flat - u_hf_flat)**2)
            imp = (mse_lf - mse_pinn) / mse_lf * 100 if mse_lf > 1e-15 else 0.0

            # Check if this time step is a BOOST point
            is_boost = any(np.isclose(t_val, bp, atol=0.01) for bp in boost_points)

            results_list.append({
                'Time(s)': round(t_val, 2),
                'Input_MSE_LF': mse_lf,
                'Model_MSE_PINN': mse_pinn,
                'Improvement_Percentage': imp,
                'Is_BOOST': is_boost
            })

            if t_val % 5.0 < 0.05:
                print(f"  > Evaluating t={t_val:.1f}s...")

    # Save numerical results to CSV
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(save_dir, "evaluation_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n>> Numerical data saved to: {csv_path}")

    times = df['Time(s)'].values
    mse_lf_list = df['Input_MSE_LF'].values
    mse_pinn_list = df['Model_MSE_PINN'].values
    imp_list = df['Improvement_Percentage'].values

    # Graph 1: MSE Comparison (log scale)
    plt.figure(figsize=(10, 6))
    plt.plot(times, mse_lf_list, 'r--', label='Input LF Error', alpha=0.5)
    plt.plot(times, mse_pinn_list, 'b-', label='Model Prediction Error', linewidth=2)

    if len(boost_points) > 0:
        boost_mses = [mse_pinn_list[np.abs(times - bt).argmin()] for bt in boost_points]
        plt.scatter(boost_points, boost_mses, color='gold', marker='*', s=150,
                    edgecolor='black', label='HF Injection (BOOST)', zorder=5)

    plt.axvline(x=10, color='k', linestyle=':', linewidth=2)
    plt.yscale('log'); plt.xlabel('Time (s)'); plt.ylabel('MSE (Log Scale)')
    plt.legend(loc='upper right'); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mse_log.png"), dpi=300)
    plt.show()

    # Graph 2: Improvement Rate (%)
    plt.figure(figsize=(10, 6))
    plt.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='Target (95%)')
    plt.plot(times, imp_list, 'g-', label='Improvement Rate', linewidth=2)

    if len(boost_points) > 0:
        boost_imps = [imp_list[np.abs(times - bt).argmin()] for bt in boost_points]
        plt.scatter(boost_points, boost_imps, color='gold', marker='*', s=200,
                    edgecolor='black', label='HF Injection (BOOST)', zorder=5)

    plt.axvline(x=10, color='k', linestyle=':', linewidth=2)
    plt.axvspan(10, 30, color='orange', alpha=0.05)
    plt.xlabel('Time (s)'); plt.ylabel('Improvement Percentage (%)')
    plt.ylim(-20, 110); plt.legend(loc='lower right'); plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "imp_rate.png"), dpi=300)
    plt.show()


evaluate_and_report(manual_boost_times)
