import sys
import os
import shutil
import subprocess
import random
import json

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
REPO_DIR = "CEBRA"
DATASET_DIR = "dataset"
RESULT_DIR = "results_smoothgrad_fake_neurons"

RATS = [
    "achilles",
    # "buddy",
    # "cicero",
    # "gatsby",
]

# CEBRA settings
ADV_EPSILON = 0.1
MAX_ITER = 1500
OUTPUT_DIM = 48
BATCH_SIZE = 2048

# Fake neurons
N_FAKE = 10
FAKE_RNG_SEED = 0

# SmoothGrad settings
N_SMOOTHGRAD_SAMPLES = 50
SMOOTHGRAD_NOISE_SCALE = 0.90  # multiplied by per-neuron std from the augmented train set
SMOOTHGRAD_CLIP_MIN = 0.0

os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# PATCH CEBRA
# ============================================================
if not os.path.exists(REPO_DIR):
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/AdaptiveMotorControlLab/CEBRA.git",
        ],
        check=True,
    )

# These two files are from your local patched version.
# Keep them if you already have your patched CEBRA wrappers.
shutil.copy(
    "base.py",
    os.path.join(REPO_DIR, "cebra/solver/base.py"),
)
shutil.copy(
    "cebra.py",
    os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py"),
)
shutil.copy(
    "cebra.py",
    os.path.join(REPO_DIR, "cebra/cebra.py"),
)

base_path = os.path.join(REPO_DIR, "cebra/solver/base.py")
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
from cebra import CEBRA

print("CEBRA:", cebra.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# HELPERS
# ============================================================
def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_local_rat_dataset(name, dataset_dir=DATASET_DIR):
    path = os.path.join(dataset_dir, f"{name}.jl")
    data = joblib.load(path)

    spikes = data["spikes"].astype(np.float32)
    position = data["position"].astype(np.float32)

    if position.ndim == 1:
        position = position[:, None]

    print(f"{name}: spikes={spikes.shape}  position={position.shape}")
    return spikes, position


def insert_fake_at_positions(x, positions, rng, mu, sigma):
    """Insert fake neurons at the chosen positions.

    x: (T, N_real)
    positions: fake neuron indices in the augmented space of size N_real + N_FAKE
    mu/sigma: scalar Gaussian parameters sampled from the real dataset
    """
    n_total = x.shape[1] + len(positions)
    is_fake = np.zeros(n_total, dtype=bool)
    is_fake[positions] = True

    fake_values = rng.normal(
        loc=mu,
        scale=sigma,
        size=(x.shape[0], len(positions)),
    ).astype(np.float32)

    combined = np.zeros((x.shape[0], n_total), dtype=np.float32)
    combined[:, is_fake] = fake_values
    combined[:, ~is_fake] = x
    return combined


def add_fake_neurons(train_data, test_data, n_fake=N_FAKE, seed=FAKE_RNG_SEED):
    """Add Gaussian fake neurons to both train and test with same positions."""
    if n_fake == 0:
        return train_data, test_data, np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    n_real = train_data.shape[1]

    positions = np.sort(
        rng.choice(
            n_real + n_fake,
            size=n_fake,
            replace=False,
        )
    )

    mu = float(train_data.mean())
    sigma = float(train_data.std() + 1e-6)

    train_data_aug = insert_fake_at_positions(train_data, positions, rng, mu, sigma)
    test_data_aug = insert_fake_at_positions(test_data, positions, rng, mu, sigma)
    return train_data_aug, test_data_aug, positions


def get_torch_model(model):
    torch_model = model.solver_.model
    torch_model.split_outputs = False
    torch_model.to(device)
    torch_model.eval()
    for p in torch_model.parameters():
        p.requires_grad_(False)
    return torch_model


def smoothgrad_feature_importance(
    torch_model,
    input_np,
    feature_scale,
    n_samples=N_SMOOTHGRAD_SAMPLES,
    noise_scale=SMOOTHGRAD_NOISE_SCALE,
    clip_min=SMOOTHGRAD_CLIP_MIN,
):
    """
    input_np: [T, N]
    model input: [1, N, T]
    """
    x = torch.tensor(input_np.T[None, :, :], dtype=torch.float32, device=device)

    scale = torch.as_tensor(feature_scale, dtype=torch.float32, device=device)
    if scale.ndim == 0:
        scale = scale.view(1, 1, 1)
    elif scale.ndim == 1:
        scale = scale.view(1, -1, 1)
    elif scale.ndim == 2:
        scale = scale.view(1, scale.shape[1], 1)
    else:
        raise ValueError(f"Unsupported feature_scale shape: {tuple(scale.shape)}")

    total_raw = torch.zeros_like(x)
    total_std = torch.zeros_like(x)

    for _ in range(n_samples):
        noise = torch.randn_like(x) * (noise_scale * scale)
        noisy_x = x + noise

        if clip_min is not None:
            noisy_x = torch.clamp(noisy_x, min=clip_min)

        noisy_x = noisy_x.detach().requires_grad_(True)

        latent = torch_model(noisy_x)
        score = (latent ** 2).sum()

        grad = torch.autograd.grad(score, noisy_x, retain_graph=False, create_graph=False)[0]

        total_raw += grad.abs().detach()
        total_std += (grad * scale).abs().detach()   # standardized importance

    avg_raw = total_raw / float(n_samples)
    avg_std = total_std / float(n_samples)

    raw_feature_importance = avg_raw.squeeze(0).mean(dim=1).cpu().numpy()
    std_feature_importance = avg_std.squeeze(0).mean(dim=1).cpu().numpy()

    return avg_raw.cpu().numpy(), raw_feature_importance, std_feature_importance

def importance_stats(feature_importance, fake_positions):
    n_features = len(feature_importance)
    all_idx = np.arange(n_features)
    fake_positions = np.asarray(fake_positions, dtype=int)
    real_positions = np.setdiff1d(all_idx, fake_positions, assume_unique=False)

    fake_vals = feature_importance[fake_positions] if len(fake_positions) > 0 else np.array([])
    real_vals = feature_importance[real_positions] if len(real_positions) > 0 else np.array([])

    fake_mean = float(fake_vals.mean()) if len(fake_vals) > 0 else np.nan
    real_mean = float(real_vals.mean()) if len(real_vals) > 0 else np.nan
    fake_sum = float(fake_vals.sum()) if len(fake_vals) > 0 else 0.0
    total_sum = float(feature_importance.sum()) + 1e-12

    return {
        "fake_mean": fake_mean,
        "real_mean": real_mean,
        "fake_to_real_ratio": float(fake_mean / (real_mean + 1e-12)) if np.isfinite(fake_mean) and np.isfinite(real_mean) else np.nan,
        "fake_share_of_total": float(fake_sum / total_sum),
        "n_fake": int(len(fake_positions)),
        "n_real": int(len(real_positions)),
    }


def save_feature_importance_plot(rat, mode, feature_importance, fake_positions, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(14, 4.5))
    plt.plot(feature_importance, linewidth=1.2)
    for f in fake_positions:
        plt.axvline(f, color="red", linestyle="--", linewidth=0.8)
    plt.title(f"{rat} — {mode} — SmoothGrad feature importance")
    plt.xlabel("Neuron index (including fakes)")
    plt.ylabel("Mean |gradient|")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"smoothgrad_importance_{mode}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot -> {out_path}")

    if len(fake_positions) > 0:
        fake_vals = feature_importance[fake_positions]
        real_positions = np.setdiff1d(np.arange(len(feature_importance)), fake_positions)
        real_vals = feature_importance[real_positions]

        plt.figure(figsize=(6, 4.5))
        plt.bar([0, 1], [real_vals.mean(), fake_vals.mean()], tick_label=["real", "fake"])
        plt.title(f"{rat} — {mode} — mean attribution")
        plt.ylabel("Mean |gradient|")
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"smoothgrad_real_vs_fake_{mode}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot -> {out_path}")


def print_summary(all_results):
    print()
    print("=" * 96)
    print("SMOOTHGRAD FAKE-NEURON SUMMARY")
    print("=" * 96)
    for rat, rat_res in all_results.items():
        print(f"\n{rat}")
        for mode in ["clean", "adv"]:
            if mode not in rat_res:
                continue
            s = rat_res[mode]["stats"]
            print(
                f"  {mode:4s} | "
                f"fake_mean={s['fake_mean']:.6f} | "
                f"real_mean={s['real_mean']:.6f} | "
                f"fake/real={s['fake_to_real_ratio']:.4f} | "
                f"fake_share={s['fake_share_of_total']:.4f}"
            )
    print("=" * 96)
    print()


def save_global_compare_plot(rat, clean_stats, adv_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    labels = ["fake_mean", "real_mean", "fake/real", "fake_share"]

    clean_vals = [
        clean_stats["fake_mean"],
        clean_stats["real_mean"],
        clean_stats["fake_to_real_ratio"],
        clean_stats["fake_share_of_total"],
    ]
    adv_vals = [
        adv_stats["fake_mean"],
        adv_stats["real_mean"],
        adv_stats["fake_to_real_ratio"],
        adv_stats["fake_share_of_total"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width / 2, clean_vals, width, label="clean")
    plt.bar(x + width / 2, adv_vals, width, label="adv")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title(f"{rat} — Clean vs Adv SmoothGrad summary")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, "clean_vs_adv_summary.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot -> {out_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================
all_results = {}

for training_mode in ["clean", "adversarial"]:
    print("=" * 80)
    print(training_mode.upper())
    print("=" * 80)

    for rat_name in RATS:
        print(f"\nTraining {rat_name} ...")

        spikes, position = load_local_rat_dataset(rat_name)
        split = int(0.8 * len(spikes))
        train_data = spikes[:split]
        test_data = spikes[split:]
        train_label = position[:split, :2]
        test_label = position[split:, :2]

        # Add fake neurons to both train and test, same positions.
        train_data_aug, test_data_aug, fake_positions = add_fake_neurons(
            train_data,
            test_data,
            n_fake=N_FAKE,
            seed=FAKE_RNG_SEED,
        )

        if len(fake_positions) > 0:
            print("Fake neuron positions:", fake_positions)
        else:
            print("No fake neurons.")

        # Per-neuron scale for SmoothGrad noise, computed from the augmented train set.
        train_feature_scale = train_data_aug.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
        print(
            "Dataset Gaussian scale (for fake neurons + SmoothGrad): "
            f"mean={float(train_data_aug.mean()):.6f}, std_mean={float(train_feature_scale.mean()):.6f}"
        )

        setup_seed(0)
        model = CEBRA(
            batch_size=BATCH_SIZE,
            temperature=0.4,
            model_architecture="offset36-model-more-dropout",
            time_offsets=4,
            max_iterations=MAX_ITER,
            output_dimension=OUTPUT_DIM,
            verbose=True,
            training_mode=training_mode,
            adv_alpha=ADV_EPSILON / 5,
            adv_epsilon=ADV_EPSILON,
            adv_steps=10,
            attack_norm="l2",
            jacobian_weight=0,
            adv_aggregate=False,
        )
        model.fit(train_data_aug, train_label)

        torch_model = get_torch_model(model)

        # SmoothGrad on the TEST set with fake neurons included.
        save_dir = os.path.join(RESULT_DIR, rat_name, training_mode)
        os.makedirs(save_dir, exist_ok=True)

        _, raw_importance, std_importance = smoothgrad_feature_importance(
            torch_model=torch_model,
            input_np=test_data_aug,
            feature_scale=train_feature_scale,
            n_samples=N_SMOOTHGRAD_SAMPLES,
            noise_scale=SMOOTHGRAD_NOISE_SCALE,
            clip_min=SMOOTHGRAD_CLIP_MIN,
        )


        raw_stats = importance_stats(raw_importance, fake_positions)
        std_stats = importance_stats(std_importance, fake_positions)
        
        
        print("\nRAW SmoothGrad")
        print(
            f"fake_mean={raw_stats['fake_mean']:.6f} | "
            f"real_mean={raw_stats['real_mean']:.6f} | "
            f"fake/real={raw_stats['fake_to_real_ratio']:.4f} | "
            f"fake_share={raw_stats['fake_share_of_total']:.4f}"
        )
        
        print("STANDARDIZED SmoothGrad")
        print(
            f"fake_mean={std_stats['fake_mean']:.6f} | "
            f"real_mean={std_stats['real_mean']:.6f} | "
            f"fake/real={std_stats['fake_to_real_ratio']:.4f} | "
            f"fake_share={std_stats['fake_share_of_total']:.4f}"
        )

        del model
        del torch_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
