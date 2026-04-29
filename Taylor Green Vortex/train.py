import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# ── Hyperparameters ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
torch.manual_seed(42)

L_val      = 4.0    # Domain size [0, L] x [0, L]
nu_val     = 0.001  # Kinematic viscosity
GRID       = 100    # Spatial grid resolution for data generation
T_max_train = 10.0  # Phase 1 training horizon
T_final    = 30.0   # Phase 2 extrapolation horizon
T_scale    = 30.0   # Time normalization factor (t_norm = t / T_scale)
# ──────────────────────────────────────────────────────────────────────────────


# ── Low-Fidelity Solution (differentiable, PyTorch) ───────────────────────────
def get_lf_solution_torch(x_norm, y_norm, t_norm, L, T_s):
    """
    Returns the LF field: TGV analytical solution + unphysical high-frequency noise.
    Inputs are physical coords (x, y in [0, L]) and normalized time t_norm = t / T_s.
    Implemented in PyTorch to support autograd through the composite field u = u_LF + delta_u.
    """
    t_real = t_norm * T_s
    decay  = torch.exp(-2 * (np.pi)**2 * nu_val * t_real)

    # TGV exact solution
    u_tgv = -torch.cos(np.pi * x_norm) * torch.sin(np.pi * y_norm) * decay
    v_tgv =  torch.sin(np.pi * x_norm) * torch.cos(np.pi * y_norm) * decay
    p_tgv = -0.25 * (torch.cos(2*np.pi*x_norm) + torch.cos(2*np.pi*y_norm)) * decay**2

    # Unphysical noise at spatial frequency 8π/L (the target for correction)
    noise_scale = 0.1 * decay
    arg_x = 8 * np.pi * x_norm / L
    arg_y = 8 * np.pi * y_norm / L
    u_n = noise_scale * torch.sin(arg_x) * torch.sin(arg_y)
    v_n = noise_scale * torch.cos(arg_x) * torch.cos(arg_y)
    p_n = decay**2 * 0.025 * torch.sin(arg_x / 2.0)

    return u_tgv + u_n, v_tgv + v_n, p_tgv + p_n


# ── High-Fidelity Ground Truth Generator (NumPy) ──────────────────────────────
class FluidGenerator:
    """Generates HF ground truth (pure TGV analytical solution) on a spatial grid."""

    def __init__(self, L=4.0, nu=0.001):
        self.L, self.nu = L, nu

    def get_decay(self, t):
        return np.exp(-2 * (np.pi)**2 * self.nu * t)

    def get_tgv_solution(self, X, Y, t):
        d = self.get_decay(t)
        u = -np.cos(np.pi * X) * np.sin(np.pi * Y) * d
        v =  np.sin(np.pi * X) * np.cos(np.pi * Y) * d
        p = -0.25 * (np.cos(2*np.pi*X) + np.cos(2*np.pi*Y)) * d**2
        return u, v, p

    def get_data_at_t(self, nx, ny, t_val):
        x = np.linspace(0, self.L, nx)
        y = np.linspace(0, self.L, ny)
        X, Y = np.meshgrid(x, y)
        u, v, p = self.get_tgv_solution(X, Y, t_val)
        return X, Y, u, v, p

generator = FluidGenerator(L=L_val, nu=nu_val)


# ── Model ──────────────────────────────────────────────────────────────────────
class FourierEmbedding(nn.Module):
    """Random Fourier Feature embedding to improve spectral representation."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(3, 128) * scale)

    def forward(self, coords):
        x_proj = 2 * np.pi * coords @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (N, 256)


class ResidualPINN(nn.Module):
    """
    5-layer MLP that predicts the correction field (delta_u, delta_v, delta_p).
    Input:  Fourier embedding (256-dim) concatenated with raw coords (x, y, t) → 259-dim.
    Output: correction vector of size 3.
    """

    def __init__(self):
        super().__init__()
        self.embedding = FourierEmbedding(scale=1.0)
        self.net = nn.Sequential(
            nn.Linear(259, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x, y, t):
        coords = torch.cat([x, y, t], dim=1)
        return self.net(torch.cat([self.embedding(coords), coords], dim=1))


def get_pde_loss(model, x, y, t):
    """
    Computes Navier-Stokes residuals on the composite field u = u_LF + delta_u.
    Returns (continuity_loss, momentum_loss) as scalar tensors.
    Requires x, y, t to have requires_grad=True — do NOT call inside torch.no_grad().
    """
    delta = model(x, y, t)
    u_lf, v_lf, p_lf = get_lf_solution_torch(x, y, t, L_val, T_scale)

    u = u_lf + delta[:, 0:1]
    v = v_lf + delta[:, 1:2]
    p = p_lf + delta[:, 2:3]

    def grad(out, inp):
        return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]

    # Temporal derivatives: divide by T_scale to account for normalized time input
    u_t = grad(u, t) / T_scale
    v_t = grad(v, t) / T_scale

    u_x = grad(u, x); u_y = grad(u, y)
    v_x = grad(v, x); v_y = grad(v, y)
    p_x = grad(p, x); p_y = grad(p, y)
    u_xx = grad(u_x, x); u_yy = grad(u_y, y)
    v_xx = grad(v_x, x); v_yy = grad(v_y, y)

    continuity = u_x + v_y
    mom_u = u_t + u*u_x + v*u_y + p_x - nu_val*(u_xx + u_yy)
    mom_v = v_t + u*v_x + v*v_y + p_y - nu_val*(v_xx + v_yy)

    return torch.mean(continuity**2), torch.mean(mom_u**2 + mom_v**2)

print("Setup complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Supervised + Physics Training  (t = 0 ~ 10s)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Phase 1 training data (t = 0 ~ 10s)...")

train_times = np.linspace(0, T_max_train, 11)
x_list, y_list, t_list = [], [], []
u_hf_list, v_hf_list, p_hf_list = [], [], []

for t_val in train_times:
    X, Y, u_h, v_h, p_h = generator.get_data_at_t(GRID, GRID, t_val)
    x_list.append(X.flatten());      y_list.append(Y.flatten())
    t_list.append(np.full(X.size, t_val))
    u_hf_list.append(u_h.flatten()); v_hf_list.append(v_h.flatten()); p_hf_list.append(p_h.flatten())

def to_tensor(arr):
    return torch.tensor(np.concatenate(arr), dtype=torch.float32).view(-1, 1).to(device)

x_train    = to_tensor(x_list)
y_train    = to_tensor(y_list)
t_train    = to_tensor(t_list) / T_scale  # normalized
u_hf_train = to_tensor(u_hf_list)
v_hf_train = to_tensor(v_hf_list)
p_hf_train = to_tensor(p_hf_list)

model = ResidualPINN().to(device)

print("\n=======================================================")
print("   [Phase 1] Training  (t = 0 ~ 10s)")
print("=======================================================")
phase1_start = time.time()


# ── Step 1: Denoising warm-up (data loss only) ────────────────────────────────
# Train purely on HF data first so the network learns the basic correction
# before physics constraints are introduced.
print("\n[Step 1] Denoising warm-up...")
s1_start = time.time()
opt_warmup = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2001):
    opt_warmup.zero_grad()
    u_lf, v_lf, p_lf = get_lf_solution_torch(x_train, y_train, t_train, L_val, T_scale)
    corr = model(x_train, y_train, t_train)
    loss = torch.mean(
        (u_lf + corr[:,0:1] - u_hf_train)**2 +
        (v_lf + corr[:,1:2] - v_hf_train)**2 +
        (p_lf + corr[:,2:3] - p_hf_train)**2
    )
    loss.backward()
    opt_warmup.step()
    if epoch % 500 == 0:
        print(f"  Ep {epoch:4d} | MSE: {loss.item():.2e}")
    if loss < 1e-4:
        print(f"  Ep {epoch:4d} | MSE: {loss.item():.2e}  (early stop)")
        break

print(f">> Step 1 done.  ({time.time()-s1_start:.1f}s)")


# ── Step 2: Physics-informed training with Adam ────────────────────────────────
# Joint optimization of data loss and NS residual loss.
# Collocation points are randomly sampled each epoch.
print("\n[Step 2] Physics training (Adam)...")
s2_start = time.time()
opt_adam = optim.Adam(model.parameters(), lr=5e-4)
x_r = y_r = t_r = None  # retained for L-BFGS reuse in Step 3

for epoch in range(5001):
    opt_adam.zero_grad()

    u_lf, v_lf, p_lf = get_lf_solution_torch(x_train, y_train, t_train, L_val, T_scale)
    corr = model(x_train, y_train, t_train)
    loss_data = torch.mean(
        (u_lf + corr[:,0:1] - u_hf_train)**2 +
        (v_lf + corr[:,1:2] - v_hf_train)**2 +
        (p_lf + corr[:,2:3] - p_hf_train)**2
    )

    idx = torch.randperm(x_train.size(0))[:5000]
    x_r = x_train[idx].clone().detach().requires_grad_(True)
    y_r = y_train[idx].clone().detach().requires_grad_(True)
    t_r = t_train[idx].clone().detach().requires_grad_(True)
    l_mass, l_mom = get_pde_loss(model, x_r, y_r, t_r)

    loss = loss_data * 100.0 + (l_mass + l_mom) * 1.0
    loss.backward()
    opt_adam.step()

    if epoch % 500 == 0:
        print(f"  Ep {epoch:4d} | Data: {loss_data.item():.2e} | PDE: {(l_mass+l_mom).item():.2e}"
              f" | {time.time()-s2_start:.0f}s")

print(f">> Step 2 done.  ({time.time()-s2_start:.1f}s)")


# ── Step 3: L-BFGS fine-tuning ────────────────────────────────────────────────
# Second-order optimizer for high-precision convergence using the last
# collocation sample from Step 2.
print("\n[Step 3] L-BFGS fine-tuning...")
s3_start = time.time()
lbfgs = optim.LBFGS(model.parameters(), max_iter=2000, line_search_fn="strong_wolfe", history_size=50)

def closure():
    lbfgs.zero_grad()
    u_lf, v_lf, p_lf = get_lf_solution_torch(x_train, y_train, t_train, L_val, T_scale)
    corr = model(x_train, y_train, t_train)
    loss_data = torch.mean(
        (u_lf + corr[:,0:1] - u_hf_train)**2 +
        (v_lf + corr[:,1:2] - v_hf_train)**2 +
        (p_lf + corr[:,2:3] - p_hf_train)**2
    )
    l_mass, l_mom = get_pde_loss(model, x_r, y_r, t_r)
    loss = loss_data * 100.0 + (l_mass + l_mom) * 1.0
    loss.backward()
    return loss

lbfgs.step(closure)
print(f">> Step 3 done.  ({time.time()-s3_start:.1f}s)")

torch.save(model.state_dict(), "model_phase1.pth")
print("=======================================================")
print(f"   [Phase 1] Complete.  Total: {(time.time()-phase1_start)/60:.1f} min")
print("=======================================================")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Active Extrapolation  (t = 10 ~ 30s)
# ══════════════════════════════════════════════════════════════════════════════
#
# At each time step the model is evaluated before any training:
#   - BOOST: PDE residual exceeds threshold → inject HF data and retrain.
#   - KEEP : PDE residual is acceptable    → self-refine with PDE loss only.
#
# A replay buffer prevents catastrophic forgetting of Phase 1 knowledge.
# ──────────────────────────────────────────────────────────────────────────────
print("\n=======================================================")
print("   [Phase 2] Active Extrapolation  (t = 10 ~ 30s)")
print("=======================================================")

# Phase 2 hyperparameters
sensitivity  = 24.0   # threshold = baseline_error * sensitivity
dt           = 0.5    # time step (s)
keep_epochs  = 200    # self-refining epochs for KEEP steps

model.load_state_dict(torch.load("model_phase1.pth"))
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Adaptive threshold: measure PDE residual at t=10s (end of training domain)
# and scale by sensitivity. Requires autograd — must be outside torch.no_grad().
current_t = 10.0
N_test = 2000
x_b = (torch.rand(N_test, 1) * L_val).to(device).requires_grad_(True)
y_b = (torch.rand(N_test, 1) * L_val).to(device).requires_grad_(True)
t_b = (torch.ones(N_test, 1) * current_t / T_scale).to(device).requires_grad_(True)
base_mass, base_mom = get_pde_loss(model, x_b, y_b, t_b)
baseline_error = (base_mass + base_mom).item()
pde_threshold  = baseline_error * sensitivity

print(f">> Baseline PDE residual at t=10s : {baseline_error:.2e}")
print(f">> BOOST threshold (x{sensitivity:.0f})      : {pde_threshold:.2e}")
print("-------------------------------------------------------")

# Replay buffer initialized with Phase 1 training data
rb_x, rb_y, rb_t = x_train.detach(), y_train.detach(), t_train.detach()
rb_u, rb_v, rb_p = u_hf_train.detach(), v_hf_train.detach(), p_hf_train.detach()

# Per-step log: (time, pde_error_before_training, is_boost)
# Saved to pde_log.npy for post-hoc analysis in evaluate.py.
boost_points = []
pde_log = []

phase2_start = time.time()

while current_t < T_final:
    target_t = current_t + dt
    print(f"\n[t = {target_t:5.1f}s]", end="  ")

    # Evaluate PDE residual BEFORE any training at this step (the actual trigger value)
    x_test = (torch.rand(N_test, 1) * L_val).to(device).requires_grad_(True)
    y_test = (torch.rand(N_test, 1) * L_val).to(device).requires_grad_(True)
    t_test = (torch.ones(N_test, 1) * target_t / T_scale).to(device).requires_grad_(True)
    l_mass, l_mom = get_pde_loss(model, x_test, y_test, t_test)
    current_pde_error = (l_mass + l_mom).item()
    print(f"PDE residual: {current_pde_error:.2e}  (ratio: {current_pde_error/pde_threshold:.2f})", end="  |  ")

    t0 = time.time()
    model.train()

    if current_pde_error > pde_threshold:
        # ── BOOST ─────────────────────────────────────────────────────────────
        # Fetch HF data at target_t, append to replay buffer, and retrain with
        # combined data loss + forward-looking PDE loss over [current_t, target_t].
        boost_points.append(target_t)
        pde_log.append((target_t, current_pde_error, True))
        print("BOOST")

        X_new, Y_new, u_h, v_h, p_h = generator.get_data_at_t(GRID, GRID, target_t)
        x_new_t = to_tensor([X_new.flatten()])
        y_new_t = to_tensor([Y_new.flatten()])
        t_new_t = to_tensor([np.full(X_new.size, target_t)]) / T_scale
        u_new_t = to_tensor([u_h.flatten()])
        v_new_t = to_tensor([v_h.flatten()])
        p_new_t = to_tensor([p_h.flatten()])

        rb_x = torch.cat([rb_x, x_new_t]); rb_y = torch.cat([rb_y, y_new_t]); rb_t = torch.cat([rb_t, t_new_t])
        rb_u = torch.cat([rb_u, u_new_t]); rb_v = torch.cat([rb_v, v_new_t]); rb_p = torch.cat([rb_p, p_new_t])

        for _ in range(300):
            optimizer.zero_grad()
            u_lf, v_lf, p_lf = get_lf_solution_torch(x_new_t, y_new_t, t_new_t, L_val, T_scale)
            corr = model(x_new_t, y_new_t, t_new_t)
            loss_data = torch.mean(
                (u_lf + corr[:,0:1] - u_new_t)**2 +
                (v_lf + corr[:,1:2] - v_new_t)**2 +
                (p_lf + corr[:,2:3] - p_new_t)**2
            )
            # Forward-looking PDE loss over the interval [current_t, target_t]
            t_fwd = ((torch.rand(2000, 1, device=device) * dt + current_t) / T_scale).requires_grad_(True)
            x_p   = (torch.rand(2000, 1, device=device) * L_val).requires_grad_(True)
            y_p   = (torch.rand(2000, 1, device=device) * L_val).requires_grad_(True)
            lm, lmo = get_pde_loss(model, x_p, y_p, t_fwd)
            (loss_data * 100.0 + (lm + lmo) * 10.0).backward()
            optimizer.step()

        # L-BFGS fine-tuning on the newly injected data
        lbfgs = optim.LBFGS(model.parameters(), max_iter=50, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            u_lf, v_lf, p_lf = get_lf_solution_torch(x_new_t, y_new_t, t_new_t, L_val, T_scale)
            corr = model(x_new_t, y_new_t, t_new_t)
            loss = torch.mean(
                (u_lf + corr[:,0:1] - u_new_t)**2 +
                (v_lf + corr[:,1:2] - v_new_t)**2
            )
            loss.backward()
            return loss
        lbfgs.step(closure)

    else:
        # ── KEEP ──────────────────────────────────────────────────────────────
        # Self-refine using only PDE loss at the current time step,
        # regularized by replay buffer to prevent catastrophic forgetting.
        pde_log.append((target_t, current_pde_error, False))
        print("KEEP")

        for _ in range(keep_epochs):
            optimizer.zero_grad()
            lm, lmo = get_pde_loss(model, x_test, y_test, t_test)

            idx = torch.randperm(rb_x.size(0))[:1000]
            u_lf, _, _ = get_lf_solution_torch(rb_x[idx], rb_y[idx], rb_t[idx], L_val, T_scale)
            corr_old = model(rb_x[idx], rb_y[idx], rb_t[idx])
            loss_reg = torch.mean((u_lf + corr_old[:,0:1] - rb_u[idx])**2)

            ((lm + lmo) * 10.0 + loss_reg * 100.0).backward()
            optimizer.step()

    print(f"        training time: {time.time()-t0:.1f}s")
    current_t = target_t

print(f"\n[Phase 2] Complete.  Total: {(time.time()-phase2_start)/60:.1f} min")
print(f"BOOST points ({len(boost_points)}): {boost_points}")


# ── Save logs for evaluate.py ─────────────────────────────────────────────────
pde_log_arr = np.array(
    pde_log,
    dtype=[('time', float), ('pde_error', float), ('is_boost', bool)]
)
np.save("pde_log.npy",       pde_log_arr)
np.save("pde_threshold.npy", np.array([pde_threshold]))
print(f">> Saved pde_log.npy ({len(pde_log_arr)} steps), pde_threshold.npy")
