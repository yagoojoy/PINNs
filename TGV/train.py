import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# ==========================================
# 0. Environment & Parameter Setup
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
torch.manual_seed(42)

# Physical and training parameters
L_val, nu_val, GRID = 4.0, 0.001, 100
T_max_train = 10.0
T_final = 30.0
T_scale = 30.0  # Time normalization scale

# ==========================================
# 1. Differentiable LF Solution (Core)
# ==========================================
# Torch-based reimplementation of the LF generator to enable autograd
def get_lf_solution_torch(x_norm, y_norm, t_norm, L, T_s):
    # Inputs: physical coordinates x, y in [0, L] and normalized time t_norm = t / T_s
    # Returns: LF approximation of u, v, p (TGV solution + unphysical noise)

    # Recover real time from normalized time
    t_real = t_norm * T_s

    # Exponential decay factor
    decay = torch.exp(-2 * (np.pi)**2 * 0.001 * t_real)
    noise_scale = 0.1 * decay

    # TGV analytical solution (high-fidelity base)
    u_tgv = -1.0 * torch.cos(np.pi * x_norm) * torch.sin(np.pi * y_norm) * decay
    v_tgv =  1.0 * torch.sin(np.pi * x_norm) * torch.cos(np.pi * y_norm) * decay
    p_tgv = -0.25 * (torch.cos(2*np.pi*x_norm) + torch.cos(2*np.pi*y_norm)) * (decay**2)

    # Unphysical noise added to simulate LF error (spatial frequency: 8pi/L)
    arg_x = 8 * np.pi * x_norm / L
    arg_y = 8 * np.pi * y_norm / L

    u_n = noise_scale * torch.sin(arg_x) * torch.sin(arg_y)
    v_n = noise_scale * torch.cos(arg_x) * torch.cos(arg_y)
    p_n = (decay**2) * 0.1 * 0.25 * torch.sin(arg_x / 2.0)

    return u_tgv + u_n, v_tgv + v_n, p_tgv + p_n


# Ground truth (HF) data generator using NumPy
class FluidGenerator:
    def __init__(self, L=4.0, nu=0.001):
        self.L, self.nu = L, nu

    def get_decay(self, t):
        k = 1
        return np.exp(-2 * (k * np.pi)**2 * self.nu * t)

    def get_tgv_solution(self, X, Y, t):
        decay = self.get_decay(t)
        u = -1.0 * np.cos(np.pi * X) * np.sin(np.pi * Y) * decay
        v =  1.0 * np.sin(np.pi * X) * np.cos(np.pi * Y) * decay
        p = -0.25 * (np.cos(2*np.pi * X) + np.cos(2*np.pi*Y)) * (decay**2)
        return u, v, p

    def get_data_at_t(self, nx, ny, t_val):
        x, y = np.linspace(0, self.L, nx), np.linspace(0, self.L, ny)
        X, Y = np.meshgrid(x, y)
        u_hf, v_hf, p_hf = self.get_tgv_solution(X, Y, t_val)
        return X, Y, u_hf, v_hf, p_hf

generator = FluidGenerator(L=L_val, nu=nu_val)


# ==========================================
# 2. Model Definition (Input: x, y, t -> Output: Residual Correction)
# ==========================================
class FourierEmbedding(nn.Module):
    def __init__(self, scale=10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(3, 128) * scale)  # Random Fourier features

    def forward(self, coords):
        x_proj = 2 * np.pi * coords @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # Output dim: 256


class ResidualPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = FourierEmbedding(scale=1.0)
        # Input: Fourier embedding (256) + raw coordinates (3) = 259
        self.net = nn.Sequential(
            nn.Linear(256+3, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 3)  # Output: corrections for u, v, p
        )

    def forward(self, x, y, t):
        coords = torch.cat([x, y, t], dim=1)
        emb = self.embedding(coords)
        inp = torch.cat([emb, coords], dim=1)
        return self.net(inp)


# PDE residual loss: enforces Navier-Stokes equations on the predicted flow field
def get_pde_loss(model, x, y, t):
    # Residual correction predicted by the model
    correction = model(x, y, t)
    u_c, v_c, p_c = correction[:,0:1], correction[:,1:2], correction[:,2:3]

    # LF base solution (differentiable)
    u_lf, v_lf, p_lf = get_lf_solution_torch(x, y, t, L_val, T_scale)

    # Final predicted flow field: LF + correction
    u = u_lf + u_c
    v = v_lf + v_c
    p = p_lf + p_c

    # Automatic differentiation helper
    def grad(out, inp):
        return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]

    # First and second-order derivatives (chain rule applied for normalized time)
    u_t = grad(u, t) / T_scale
    v_t = grad(v, t) / T_scale
    u_x = grad(u, x); u_y = grad(u, y)
    v_x = grad(v, x); v_y = grad(v, y)
    p_x = grad(p, x); p_y = grad(p, y)
    u_xx = grad(u_x, x); u_yy = grad(u_y, y)
    v_xx = grad(v_x, x); v_yy = grad(v_y, y)

    # Navier-Stokes residuals: continuity and momentum equations
    f_mass = u_x + v_y
    f_u = u_t + (u*u_x + v*u_y) + p_x - nu_val*(u_xx + u_yy)
    f_v = v_t + (u*v_x + v*v_y) + p_y - nu_val*(v_xx + v_yy)

    return torch.mean(f_mass**2), torch.mean(f_u**2 + f_v**2)

print("Setup Complete: Functions & Model Defined.")


# ==========================================
# [Phase 1] Data Generation & 3-Step Training (0 ~ 10s)
# ==========================================
print("Generating Training Data (0 ~ 10s)...")

train_times = np.linspace(0, T_max_train, 11)
x_list, y_list, t_list = [], [], []
u_hf_list, v_hf_list, p_hf_list = [], [], []

for t_val in train_times:
    X, Y, u_h, v_h, p_h = generator.get_data_at_t(GRID, GRID, t_val)
    x_list.append(X.flatten())
    y_list.append(Y.flatten())
    t_list.append(np.full(X.size, t_val))
    u_hf_list.append(u_h.flatten())
    v_hf_list.append(v_h.flatten())
    p_hf_list.append(p_h.flatten())

def to_tensor(arr):
    return torch.tensor(np.concatenate(arr), dtype=torch.float32).view(-1, 1).to(device)

x_train = to_tensor(x_list)
y_train = to_tensor(y_list)
t_train = to_tensor(t_list) / T_scale  # Normalize time to [0, 1]
u_hf_train = to_tensor(u_hf_list)
v_hf_train = to_tensor(v_hf_list)
p_hf_train = to_tensor(p_hf_list)

print(f"Data Generation Complete. Shape: {x_train.shape}")

model = ResidualPINN().to(device)

print("\n=======================================================")
print("   [Phase 1] Initial Training (0 ~ 10s) Start")
print("=======================================================")
phase1_start_time = time.time()

# ----------------------------------------------------------------
# Step 1: Denoising Warm-up (Data Loss Only)
# ----------------------------------------------------------------
print("\n[Step 1] Denoising Warm-up...")
s1_start = time.time()
optimizer_warmup = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2001):
    optimizer_warmup.zero_grad()

    u_lf, v_lf, p_lf = get_lf_solution_torch(x_train, y_train, t_train, L_val, T_scale)
    correction = model(x_train, y_train, t_train)
    u_pred = u_lf + correction[:,0:1]
    v_pred = v_lf + correction[:,1:2]
    p_pred = p_lf + correction[:,2:3]

    loss = torch.mean((u_pred - u_hf_train)**2 + (v_pred - v_hf_train)**2 + (p_pred - p_hf_train)**2)
    loss.backward()
    optimizer_warmup.step()

    if epoch % 500 == 0:
        print(f"  > Ep {epoch} | MSE: {loss.item():.7f}")
    if loss < 1e-4:
        print(f"  > Ep {epoch} | MSE: {loss.item():.7f}")
        break

print(f">> Step 1 Done. Time: {time.time() - s1_start:.2f}s")

# ----------------------------------------------------------------
# Step 2: Physics Training with Adam (Data + PDE Loss)
# ----------------------------------------------------------------
print("\n[Step 2] Physics Training (Adam)...")
s2_start = time.time()
optimizer_adam = optim.Adam(model.parameters(), lr=5e-4)

x_r, y_r, t_r = None, None, None  # Will store last PDE sample for L-BFGS reuse

for epoch in range(5001):
    optimizer_adam.zero_grad()

    # Data loss
    u_lf, v_lf, p_lf = get_lf_solution_torch(x_train, y_train, t_train, L_val, T_scale)
    correction = model(x_train, y_train, t_train)
    u_pred = u_lf + correction[:,0:1]
    v_pred = v_lf + correction[:,1:2]
    p_pred = p_lf + correction[:,2:3]
    loss_data = torch.mean((u_pred - u_hf_train)**2 + (v_pred - v_hf_train)**2 + (p_pred - p_hf_train)**2)

    # PDE loss on random collocation points
    idx = torch.randperm(x_train.size(0))[:5000]
    x_r = x_train[idx].clone().detach().requires_grad_(True)
    y_r = y_train[idx].clone().detach().requires_grad_(True)
    t_r = t_train[idx].clone().detach().requires_grad_(True)

    l_mass, l_mom = get_pde_loss(model, x_r, y_r, t_r)
    loss_pde = l_mass + l_mom

    loss = loss_data * 100.0 + loss_pde * 1.0
    loss.backward()
    optimizer_adam.step()

    if epoch % 500 == 0:
        elapsed = time.time() - s2_start
        print(f"  > Ep {epoch} | Data: {loss_data.item():.6f} | PDE: {loss_pde.item():.6f} | Elapsed: {elapsed:.1f}s")

print(f">> Step 2 Done. Time: {time.time() - s2_start:.2f}s")

# ----------------------------------------------------------------
# Step 3: L-BFGS Fine-tuning
# ----------------------------------------------------------------
print("\n[Step 3] L-BFGS Fine-tuning...")
s3_start = time.time()

lbfgs = optim.LBFGS(model.parameters(), max_iter=2000, line_search_fn="strong_wolfe", history_size=50)

def closure():
    lbfgs.zero_grad()
    u_lf, v_lf, p_lf = get_lf_solution_torch(x_train, y_train, t_train, L_val, T_scale)
    correction = model(x_train, y_train, t_train)
    u_pred = u_lf + correction[:,0:1]
    v_pred = v_lf + correction[:,1:2]
    p_pred = p_lf + correction[:,2:3]
    loss_data = torch.mean((u_pred - u_hf_train)**2 + (v_pred - v_hf_train)**2 + (p_pred - p_hf_train)**2)

    # Reuse last PDE sample from Step 2
    l_mass, l_mom = get_pde_loss(model, x_r, y_r, t_r)

    loss = loss_data * 100.0 + (l_mass + l_mom) * 1.0
    loss.backward()
    return loss

lbfgs.step(closure)
print(f">> Step 3 Done. Time: {time.time() - s3_start:.2f}s")

# Save Phase 1 model checkpoint
torch.save(model.state_dict(), "model_phase1.pth")
total_time = time.time() - phase1_start_time
print("=======================================================")
print(f"   [Phase 1] Complete.")
print(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
print("=======================================================")


# ==========================================
# [Phase 2] Active Extrapolation (10 ~ 30s)
# ==========================================
print("\n=======================================================")
print("   [Phase 2] Active Extrapolation (10 ~ 30s) Start")
print("=======================================================")

# ----------------------------------------------------------------
# Tuning parameters
# ----------------------------------------------------------------
sensitivity = 24.0   # PDE threshold multiplier
dt = 0.5             # Time step for extrapolation
T_final = 30.0
keep_epochs = 200    # Self-refining epochs when PDE error is below threshold
# ----------------------------------------------------------------

try:
    model.load_state_dict(torch.load("model_phase1.pth"))
except FileNotFoundError:
    print("Error: model_phase1.pth not found. Run Phase 1 first.")
    assert False, "Model file not found."

optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Compute baseline PDE error at t=10s to set adaptive threshold
current_t = 10.0
print("Calculating Baseline PDE Error at t=10.0s...")
N_test = 2000
with torch.no_grad():
    x_test_base = torch.rand(N_test, 1).to(device) * L_val
    y_test_base = torch.rand(N_test, 1).to(device) * L_val
    t_test_base = (torch.ones(N_test, 1).to(device) * current_t) / T_scale
    x_test_base.requires_grad_(True)
    y_test_base.requires_grad_(True)
    t_test_base.requires_grad_(True)

base_mass, base_mom = get_pde_loss(model, x_test_base, y_test_base, t_test_base)
baseline_error = (base_mass + base_mom).item()
pde_threshold = baseline_error * sensitivity

print(f">> Baseline PDE Error: {baseline_error:.7f}")
print(f">> Adaptive Threshold: {pde_threshold:.7f} (Sensitivity: x{sensitivity})")
print("-------------------------------------------------------")

# Initialize replay buffer with Phase 1 training data
rb_x, rb_y, rb_t = x_train.detach(), y_train.detach(), t_train.detach()
rb_u, rb_v, rb_p = u_hf_train.detach(), v_hf_train.detach(), p_hf_train.detach()

phase2_start_time = time.time()

while current_t < T_final:
    target_t = current_t + dt
    print(f"\n[Time {target_t:4.1f}s]", end=" ")

    # Quality check: evaluate PDE residual at target time
    x_test = torch.rand(N_test, 1).to(device) * L_val
    y_test = torch.rand(N_test, 1).to(device) * L_val
    t_test = (torch.ones(N_test, 1).to(device) * target_t) / T_scale
    x_test.requires_grad_(True)
    y_test.requires_grad_(True)
    t_test.requires_grad_(True)

    l_mass, l_mom = get_pde_loss(model, x_test, y_test, t_test)
    current_pde_error = (l_mass + l_mom).item()
    ratio = current_pde_error / pde_threshold
    print(f"PDE Err: {current_pde_error:.6f} (x{ratio:.2f})", end=" | ")

    train_start_time = time.time()
    model.train()

    if current_pde_error > pde_threshold:
        # BOOST: PDE error exceeds threshold -> inject HF data and retrain
        print("Status: [BOOST] (Surgical Precision Mode)...")

        # Fetch HF ground truth at target time
        X_new, Y_new, u_h, v_h, p_h = generator.get_data_at_t(GRID, GRID, target_t)
        x_new_t = to_tensor([X_new.flatten()])
        y_new_t = to_tensor([Y_new.flatten()])
        t_new_t = to_tensor([np.full(X_new.size, target_t)]) / T_scale
        u_new_t = to_tensor([u_h.flatten()])
        v_new_t = to_tensor([v_h.flatten()])
        p_new_t = to_tensor([p_h.flatten()])

        # Append new data to replay buffer
        rb_x = torch.cat([rb_x, x_new_t], 0)
        rb_y = torch.cat([rb_y, y_new_t], 0)
        rb_t = torch.cat([rb_t, t_new_t], 0)
        rb_u = torch.cat([rb_u, u_new_t], 0)
        rb_v = torch.cat([rb_v, v_new_t], 0)
        rb_p = torch.cat([rb_p, p_new_t], 0)

        # Step B: Adam optimization with data loss + forward-looking PDE loss
        for i in range(300):
            optimizer.zero_grad()

            c_new = model(x_new_t, y_new_t, t_new_t)
            lfn_u, lfn_v, lfn_p = get_lf_solution_torch(x_new_t, y_new_t, t_new_t, L_val, T_scale)
            loss_new = torch.mean((lfn_u + c_new[:,0:1] - u_new_t)**2 +
                                  (lfn_v + c_new[:,1:2] - v_new_t)**2 +
                                  (lfn_p + c_new[:,2:3] - p_new_t)**2)

            # Forward-looking PDE loss: sample collocation points in [current_t, target_t]
            t_future = (torch.rand(2000, 1, device=device) * dt + current_t) / T_scale
            x_p = (torch.rand(2000, 1, device=device) * L_val).requires_grad_(True)
            y_p = (torch.rand(2000, 1, device=device) * L_val).requires_grad_(True)
            t_future = t_future.requires_grad_(True)
            lm, lmo = get_pde_loss(model, x_p, y_p, t_future)

            total_loss = loss_new * 100.0 + (lm + lmo) * 10.0
            total_loss.backward()
            optimizer.step()

        # Step C: L-BFGS fine-tuning on current time step
        print("   -> L-BFGS Fine-tuning...", end="")
        lbfgs = optim.LBFGS(model.parameters(), max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            c = model(x_new_t, y_new_t, t_new_t)
            lfn_u, lfn_v, lfn_p = get_lf_solution_torch(x_new_t, y_new_t, t_new_t, L_val, T_scale)
            loss = torch.mean((lfn_u + c[:,0:1] - u_new_t)**2 + (lfn_v + c[:,1:2] - v_new_t)**2)
            loss.backward()
            return loss

        lbfgs.step(closure)
        print(" Done.")

    else:
        # KEEP: PDE error within threshold -> self-refine with PDE + replay buffer
        print("Status: [KEEP] (Self-Refining)...")
        for i in range(keep_epochs):
            optimizer.zero_grad()
            lm, lmo = get_pde_loss(model, x_test, y_test, t_test)

            # Replay buffer regularization to prevent catastrophic forgetting
            idx = torch.randperm(rb_x.size(0))[:1000]
            corr_old = model(rb_x[idx], rb_y[idx], rb_t[idx])
            lf_u, _, _ = get_lf_solution_torch(rb_x[idx], rb_y[idx], rb_t[idx], L_val, T_scale)
            loss_reg = torch.mean((lf_u + corr_old[:,0:1] - rb_u[idx])**2)

            loss = (lm + lmo) * 10.0 + loss_reg * 100.0
            loss.backward()
            optimizer.step()

    print(f"   -> Train: {time.time() - train_start_time:.2f}s")
    current_t = target_t

total_phase2_time = time.time() - phase2_start_time
print(f"\n[Phase 2] Complete. Total Time: {total_phase2_time:.1f}s")
