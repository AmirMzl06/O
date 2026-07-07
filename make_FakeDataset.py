import sys
import os
import shutil
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIG & PATCH CEBRA
# ============================================================
REPO_DIR       = "CEBRA"
adv_epsilon    = 0.5
MAX_ITER       = 1000
OUTPUT_DIM     = 80
BATCH_SIZE     = 512

# ── Patch CEBRA ─────────────────────────────────────────────
if not os.path.exists(REPO_DIR):
    subprocess.run([
        "git", "clone",
        "https://github.com/AdaptiveMotorControlLab/CEBRA.git",
    ], check=True)

if os.path.exists("base.py") and os.path.exists("cebra.py"):
    shutil.copy("base.py", os.path.join(REPO_DIR, "cebra/solver/base.py"))
    shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py"))
    shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/cebra.py"))

base_path = os.path.join(REPO_DIR, "cebra/solver/base.py")
if os.path.exists(base_path):
    with open(base_path, "r") as f:
        content = f.read()
    if "AuxiliaryVariableSolver" not in content:
        with open(base_path, "a") as f:
            f.write("\nclass AuxiliaryVariableSolver(Solver):\n    pass\n")
            f.write("\nclass DiscreteAuxiliaryVariableSolver(Solver):\n    pass\n")
    print("Patch applied.")

sys.path.insert(0, REPO_DIR)
if "cebra" in sys.modules:
    del sys.modules["cebra"]

import cebra
import cebra.attribution
from cebra import CEBRA

print("CEBRA:", cebra.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 2. DATA GENERATION
# ============================================================
def generate_synthetic_dataset(
    T=5000,
    n_latent=80,
    n_neurons=120,
    seed=42,
):
    rng = np.random.default_rng(seed)
    n_noise_neurons = n_neurons - n_latent
    assert n_noise_neurons >= 0, "n_neurons must be >= n_latent"

    # 1) 2D position
    steps = rng.normal(loc=0.0, scale=0.03, size=(T, 2))
    position = np.cumsum(steps, axis=0)
    position = position - position.min(axis=0, keepdims=True)
    position = position / (position.max(axis=0, keepdims=True) + 1e-9)

    # 2) Latent variables
    centers = rng.uniform(0.1, 0.9, size=(n_latent, 2))
    widths  = rng.uniform(0.05, 0.12, size=(n_latent,))
    latent = np.zeros((T, n_latent), dtype=np.float32)

    for i in range(n_latent):
        d2 = np.sum((position - centers[i]) ** 2, axis=1)
        latent[:, i] = np.exp(-d2 / (2 * widths[i] ** 2))

    latent += 0.02 * rng.normal(size=latent.shape).astype(np.float32)
    latent = np.clip(latent, 0.0, None)

    # 3) Neural generation
    neural = np.zeros((T, n_neurons), dtype=np.float32)
    neural[:, :n_latent] = latent + 0.05 * rng.normal(size=(T, n_latent))
    neural[:, n_latent:] = 0.5 * rng.normal(size=(T, n_noise_neurons))

    neural = (neural - neural.mean(axis=0, keepdims=True)) / (
        neural.std(axis=0, keepdims=True) + 1e-6
    )

    W = np.zeros((n_latent, n_neurons), dtype=np.float32)
    for i in range(n_latent):
        W[i, i] = 1.0

    return neural, position, latent, W, centers, widths

print("\nGenerating Synthetic Data...")
neural, position, latent, W, centers, widths = generate_synthetic_dataset()
print("Neural shape   :", neural.shape)
print("Position shape :", position.shape)


# ============================================================
# 3. ATTRIBUTION HELPERS
# ============================================================
def get_torch_model(model):
    torch_model = model.solver_.model
    torch_model.split_outputs = False
    torch_model.to(device)
    torch_model.eval()
    return torch_model


# ============================================================
# 4. MAIN PIPELINE (TRAINING & ATTRIBUTION)
# ============================================================
results = {}

ground_truth_attribution = (W.sum(axis=0) > 0).astype(float)
print(f"Derived Ground Truth Attribution Mask Shape: {ground_truth_attribution.shape}")

for training_mode, adv in [("clean", False), ("adversarial", True)]:
    print("\n" + "=" * 60)
    print(f"Training Model: {training_mode.upper()}")
    print("=" * 60)

    torch.manual_seed(0)
    np.random.seed(0)

    model = CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset36-model",
        time_offsets=4,
        max_iterations=MAX_ITER,
        output_dimension=OUTPUT_DIM,
        verbose=True,
        training_mode=training_mode,
        adv_alpha=adv_epsilon / 5,
        adv_epsilon=adv_epsilon,
        adv_steps=10,
        attack_norm="l2",
        jacobian_weight=0,
        adv_aggregate=True,
    )

    model.fit(neural, position)
    
    print(f"\nComputing Attribution for {training_mode}...")
    neural_t = torch.from_numpy(neural).float().to(device)
    neural_t.requires_grad_(True)
    
    torch_model = get_torch_model(model)
    
    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=torch_model,
        input_data=neural_t,
        output_dimension=torch_model.num_output,
        num_samples=2000,
    )
    
    attr = method.compute_attribution_map(batch_size=256)
    
    print("\n--- Diagnostic Prints ---")
    print("Keys in attr dictionary:", list(attr.keys()))
    if "jf" in attr:
        print("jf shape:", attr["jf"].shape)
    if "jf-inv-svd" in attr:
        print("jf-inv-svd shape:", attr["jf-inv-svd"].shape)
    if "jf-conv-abs-inv" in attr:
        print("jf-conv-abs-inv shape:", attr["jf-conv-abs-inv"].shape)
    print("-------------------------\n")

    print(f"--- AUC Scores ({training_mode}) ---")
    
    if "jf" in attr:
        auc_jf = method.compute_attribution_score(attr["jf"], ground_truth_attribution)
        print(f"auc_jf,             {auc_jf:.2f}")
        
    if "jf-inv-svd" in attr:
        auc_jfinv = method.compute_attribution_score(attr["jf-inv-svd"], ground_truth_attribution)
        print(f"auc_jfinv,          {auc_jfinv:.2f}")
        
    if "jf-conv-abs-inv" in attr:
        auc_jfconv = method.compute_attribution_score(attr["jf-conv-abs-inv"], ground_truth_attribution)
        print(f"auc_jfconvabsinv,   {auc_jfconv:.2f}")
    
    mode_key = "adv" if adv else "clean"
    results[mode_key] = attr
    
    del method
    del model
    torch.cuda.empty_cache()


# ============================================================
# 5. VISUALIZATION
# ============================================================
print("\nPlotting Results...")

clean_map = np.abs(results["clean"]["jf-inv-svd"]).mean(axis=0, keepdims=True)
adv_map   = np.abs(results["adv"]["jf-inv-svd"]).mean(axis=0, keepdims=True)

fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1, 1]})

# 1. Ground Truth W
im0 = axs[0].matshow(W, aspect="auto", cmap="Greys")
axs[0].set_title("Ground Truth Mapping (W)", pad=10)
axs[0].set_ylabel("Latents (80)")
axs[0].set_xlabel("Neurons (120)")
fig.colorbar(im0, ax=axs[0])

# 2. Clean JF-inv
im1 = axs[1].matshow(clean_map, aspect="auto", cmap="Reds")
axs[1].set_title("Clean Model: JF-inv Attribution Map", pad=10)
axs[1].set_ylabel("Importance")
axs[1].set_yticks([])
fig.colorbar(im1, ax=axs[1])

# 3. Adversarial JF-inv
im2 = axs[2].matshow(adv_map, aspect="auto", cmap="Blues")
axs[2].set_title("Adversarial Model: JF-inv Attribution Map", pad=10)
axs[2].set_ylabel("Importance")
axs[2].set_xlabel("Neurons (120)")
axs[2].set_yticks([])
fig.colorbar(im2, ax=axs[2])

for ax in axs:
    ax.axvline(x=79.5, color="red", linestyle="--", linewidth=1.5, label="Signal/Noise Boundary")

axs[0].legend(loc="upper right")
plt.tight_layout()
plt.show()

print("Finished.")
