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
RESULT_DIR = "results_latent_noise"

RATS = [
    # "achilles",
    "buddy",
    # "cicero",
    # "gatsby",
]

ADV_EPSILON = 0.1
MAX_ITER = 1500
OUTPUT_DIM = 48
BATCH_SIZE = 2048
N_FAKE = 0

# Noise settings
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.50]
N_NOISE_REPEATS = 5
NOISE_CLIP_MIN = 0.0  # suitable for spike counts / firing rates

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

# These two files are from your local patched version
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


def add_fake_neurons(train_data, test_data, n_fake=0):
    """Disabled by default. Kept here so the pipeline can be extended later."""
    if n_fake == 0:
        return train_data, test_data, np.array([], dtype=int)

    rng = np.random.default_rng(0)
    n_real = train_data.shape[1]
    positions = np.sort(
        rng.choice(
            n_real + n_fake,
            size=n_fake,
            replace=False,
        )
    )

    def insert_fake_at_positions(x, positions, rng, mu, sigma):
        n_total = x.shape[1] + len(positions)
        is_fake = np.zeros(n_total, dtype=bool)
        is_fake[positions] = True

        fake_values = rng.normal(
            loc=mu,
            scale=sigma,
            size=(x.shape[0], len(positions)),
        )

        combined = np.zeros((x.shape[0], n_total), dtype=np.float32)
        combined[:, is_fake] = fake_values
        combined[:, ~is_fake] = x
        return combined

    mu = 0.0
    sigma = 1.0
    train_data = insert_fake_at_positions(train_data, positions, rng, mu, sigma)
    test_data = insert_fake_at_positions(test_data, positions, rng, mu, sigma)
    return train_data, test_data, positions


def get_latent(model, data_np):
    z = model.transform(data_np)
    return np.asarray(z, dtype=np.float32)


def add_gaussian_noise(x, noise_level, rng, ref_std=None, clip_min=None):
    """
    If ref_std is provided, noise_level multiplies the per-feature std.
    Otherwise noise_level is used as an absolute sigma.
    """
    if ref_std is None:
        sigma = noise_level
    else:
        sigma = noise_level * ref_std

    noisy = x + rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)

    if clip_min is not None:
        noisy = np.clip(noisy, clip_min, None)

    return noisy.astype(np.float32)


def latent_shift_metrics(z_clean, z_noisy):
    """Compare latent representations sample-by-sample."""
    diff = z_noisy - z_clean

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    l2_per_sample = np.linalg.norm(diff, axis=1)
    mean_l2 = float(np.mean(l2_per_sample))

    clean_norm = np.linalg.norm(z_clean, axis=1)
    rel_l2 = float(mean_l2 / (np.mean(clean_norm) + 1e-8))

    num = np.sum(z_clean * z_noisy, axis=1)
    den = np.linalg.norm(z_clean, axis=1) * np.linalg.norm(z_noisy, axis=1) + 1e-8
    cosine = float(np.mean(num / den))

    return {
        "latent_mse": mse,
        "latent_rmse": rmse,
        "latent_mean_l2": mean_l2,
        "latent_relative_l2": rel_l2,
        "latent_cosine": cosine,
    }


def save_noise_curve_plot(rat, mode, results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    noise_levels = [r["noise_level"] for r in results]
    mean_l2 = [r["latent_mean_l2_mean"] for r in results]
    std_l2 = [r["latent_mean_l2_std"] for r in results]
    mse = [r["latent_mse_mean"] for r in results]
    std_mse = [r["latent_mse_std"] for r in results]
    cosine = [r["latent_cosine_mean"] for r in results]
    std_cosine = [r["latent_cosine_std"] for r in results]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    axs[0].errorbar(noise_levels, mean_l2, yerr=std_l2, marker="o", capsize=3)
    axs[0].set_title("Mean latent L2 shift")
    axs[0].set_xlabel("Noise level")
    axs[0].set_ylabel("Shift")

    axs[1].errorbar(noise_levels, mse, yerr=std_mse, marker="o", capsize=3)
    axs[1].set_title("Latent MSE")
    axs[1].set_xlabel("Noise level")
    axs[1].set_ylabel("MSE")

    axs[2].errorbar(noise_levels, cosine, yerr=std_cosine, marker="o", capsize=3)
    axs[2].set_title("Latent cosine similarity")
    axs[2].set_xlabel("Noise level")
    axs[2].set_ylabel("Cosine")

    plt.suptitle(f"{rat} — {mode} — latent stability under noise", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"latent_noise_curve_{mode}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot -> {out_path}")


def print_summary_table(all_results):
    print()
    print("=" * 88)
    print("LATENT STABILITY SUMMARY")
    print("=" * 88)
    for rat, rat_res in all_results.items():
        print(f"\n{rat}")
        for mode in ["clean", "adv"]:
            if mode not in rat_res:
                continue
            print(f"  {mode}")
            for row in rat_res[mode]:
                print(
                    f"    noise={row['noise_level']:.2f} | "
                    f"L2={row['latent_mean_l2_mean']:.6f} ± {row['latent_mean_l2_std']:.6f} | "
                    f"MSE={row['latent_mse_mean']:.6f} ± {row['latent_mse_std']:.6f} | "
                    f"Cos={row['latent_cosine_mean']:.4f} ± {row['latent_cosine_std']:.4f}"
                )
    print("=" * 88)
    print()


# ============================================================
# MAIN PIPELINE
# ============================================================
all_results = {}

for training_mode, adv in [
    ("clean", False),
    ("adversarial", True),
]:
    print("=" * 80)
    print(training_mode.upper())
    print("=" * 80)

    for name in RATS:
        print(f"\nTraining {name} ...")

        spikes, position = load_local_rat_dataset(name)
        split = int(0.8 * len(spikes))
        train_data = spikes[:split]
        test_data = spikes[split:]

        train_label = position[:split, :2]
        test_label = position[split:, :2]

        train_data, test_data, fake_positions = add_fake_neurons(
            train_data,
            test_data,
            n_fake=N_FAKE,
        )
        if len(fake_positions) > 0:
            print("Fake neurons:", fake_positions)
        else:
            print("No fake neurons.")

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
            jacobian_weight=0.01,
            adv_aggregate=True,
        )
        model.fit(train_data, train_label)

        # Clean latent on test set
        z_clean = get_latent(model, test_data)

        # Reference scale for noise: per-neuron std from train set
        train_std = train_data.std(axis=0, keepdims=True).astype(np.float32) + 1e-6

        rat_results = []
        save_dir = os.path.join(RESULT_DIR, name, training_mode)
        os.makedirs(save_dir, exist_ok=True)

        for noise_level in NOISE_LEVELS:
            metrics_list = []

            for rep in range(N_NOISE_REPEATS):
                rng = np.random.default_rng(1000 + rep)

                noisy_test = add_gaussian_noise(
                    test_data,
                    noise_level=noise_level,
                    rng=rng,
                    ref_std=train_std,
                    clip_min=NOISE_CLIP_MIN,
                )

                z_noisy = get_latent(model, noisy_test)
                metrics = latent_shift_metrics(z_clean, z_noisy)
                metrics_list.append(metrics)

            row = {
                "noise_level": noise_level,
                "latent_mse_mean": float(np.mean([m["latent_mse"] for m in metrics_list])),
                "latent_mse_std": float(np.std([m["latent_mse"] for m in metrics_list])),
                "latent_rmse_mean": float(np.mean([m["latent_rmse"] for m in metrics_list])),
                "latent_rmse_std": float(np.std([m["latent_rmse"] for m in metrics_list])),
                "latent_mean_l2_mean": float(np.mean([m["latent_mean_l2"] for m in metrics_list])),
                "latent_mean_l2_std": float(np.std([m["latent_mean_l2"] for m in metrics_list])),
                "latent_relative_l2_mean": float(np.mean([m["latent_relative_l2"] for m in metrics_list])),
                "latent_relative_l2_std": float(np.std([m["latent_relative_l2"] for m in metrics_list])),
                "latent_cosine_mean": float(np.mean([m["latent_cosine"] for m in metrics_list])),
                "latent_cosine_std": float(np.std([m["latent_cosine"] for m in metrics_list])),
            }
            rat_results.append(row)

            print(
                f"{name} | {training_mode} | noise={noise_level:.2f} "
                f"=> L2={row['latent_mean_l2_mean']:.6f} "
                f"MSE={row['latent_mse_mean']:.6f} "
                f"Cos={row['latent_cosine_mean']:.4f}"
            )

        save_noise_curve_plot(
            rat=name,
            mode=training_mode,
            results=rat_results,
            save_dir=save_dir,
        )

        if name not in all_results:
            all_results[name] = {}
        all_results[name]["clean" if training_mode == "clean" else "adv"] = rat_results

        summary_path = os.path.join(save_dir, f"latent_noise_summary_{training_mode}.json")
        with open(summary_path, "w") as f:
            json.dump(rat_results, f, indent=2)
        print(f"Saved summary -> {summary_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print_summary_table(all_results)

# ============================================================
# GLOBAL COMPARISON PLOT
# ============================================================
for name in RATS:
    if name not in all_results:
        continue
    if "clean" not in all_results[name] or "adv" not in all_results[name]:
        continue

    save_dir = os.path.join(RESULT_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    clean_res = all_results[name]["clean"]
    adv_res = all_results[name]["adv"]

    noise_levels = [r["noise_level"] for r in clean_res]

    clean_l2 = [r["latent_mean_l2_mean"] for r in clean_res]
    adv_l2 = [r["latent_mean_l2_mean"] for r in adv_res]

    clean_cos = [r["latent_cosine_mean"] for r in clean_res]
    adv_cos = [r["latent_cosine_mean"] for r in adv_res]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    axs[0].plot(noise_levels, clean_l2, marker="o", label="clean")
    axs[0].plot(noise_levels, adv_l2, marker="o", label="adv")
    axs[0].set_title(f"{name} — latent L2 shift")
    axs[0].set_xlabel("Noise level")
    axs[0].set_ylabel("Shift")
    axs[0].legend()

    axs[1].plot(noise_levels, clean_cos, marker="o", label="clean")
    axs[1].plot(noise_levels, adv_cos, marker="o", label="adv")
    axs[1].set_title(f"{name} — latent cosine similarity")
    axs[1].set_xlabel("Noise level")
    axs[1].set_ylabel("Cosine")
    axs[1].legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "clean_vs_adv_latent_noise_compare.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved compare plot -> {out_path}")

print("Finished.")




################### neuron -> latent -> decoder -> neurom
# import sys
# import os
# import shutil
# import subprocess
# import random

# import joblib
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from scipy.stats import zscore

# # ============================================================
# # CONFIG
# # ============================================================
# REPO_DIR = "CEBRA"
# DATASET_DIR = "dataset"
# RESULT_DIR = "results"

# rats = [
#     # "achilles",
#     # "buddy",
#     # "cicero",
#     "gatsby"
# ]

# adv_epsilon = 0.5
# N_FAKE = 0

# MAX_ITER = 1500
# OUTPUT_DIM = 48
# BATCH_SIZE = 2048

# RECON_EPOCHS = 3000
# RECON_LR = 1e-2

# os.makedirs(RESULT_DIR, exist_ok=True)

# scores = {
#     "clean": {},
#     "adv": {}
# }

# fake_neuron_indices = {}

# # ============================================================
# # PATCH CEBRA
# # ============================================================
# if not os.path.exists(REPO_DIR):
#     subprocess.run([
#         "git",
#         "clone",
#         "https://github.com/AdaptiveMotorControlLab/CEBRA.git"
#     ], check=True)

# shutil.copy(
#     "base.py",
#     os.path.join(REPO_DIR, "cebra/solver/base.py")
# )

# shutil.copy(
#     "cebra.py",
#     os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py")
# )

# shutil.copy(
#     "cebra.py",
#     os.path.join(REPO_DIR, "cebra/cebra.py")
# )

# base_path = os.path.join(REPO_DIR, "cebra/solver/base.py")
# with open(base_path, "r") as f:
#     content = f.read()

# if "AuxiliaryVariableSolver" not in content:
#     with open(base_path, "a") as f:
#         f.write("\nclass AuxiliaryVariableSolver(Solver):\n    pass\n")
#         f.write("\nclass DiscreteAuxiliaryVariableSolver(Solver):\n    pass\n")

# print("Patch applied.")

# sys.path.insert(0, REPO_DIR)

# if "cebra" in sys.modules:
#     del sys.modules["cebra"]

# import cebra
# import cebra.attribution
# from cebra import CEBRA

# print("CEBRA:", cebra.__version__)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # ============================================================
# # HELPERS
# # ============================================================
# def setup_seed(seed=42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def load_local_rat_dataset(name, dataset_dir=DATASET_DIR):
#     path = os.path.join(dataset_dir, f"{name}.jl")
#     data = joblib.load(path)

#     spikes = data["spikes"].astype(np.float32)
#     position = data["position"].astype(np.float32)

#     if position.ndim == 1:
#         position = position[:, None]

#     print(f"{name}: spikes={spikes.shape}  position={position.shape}")
#     return spikes, position


# def insert_fake_at_positions(x, positions, rng, mu, sigma):
#     n_total = x.shape[1] + len(positions)
#     is_fake = np.zeros(n_total, dtype=bool)
#     is_fake[positions] = True

#     fake_values = rng.normal(
#         loc=mu,
#         scale=sigma,
#         size=(x.shape[0], len(positions)),
#     )

#     combined = np.zeros((x.shape[0], n_total), dtype=np.float32)
#     combined[:, is_fake] = fake_values
#     combined[:, ~is_fake] = x
#     return combined


# def add_fake_neurons(train_data, test_data, key, n_fake=N_FAKE):
#     if n_fake == 0:
#         fake_neuron_indices[key] = np.array([], dtype=int)
#         return train_data, test_data, np.array([], dtype=int)

#     rng = np.random.default_rng(0)
#     n_real = train_data.shape[1]

#     positions = np.sort(
#         rng.choice(
#             n_real + n_fake,
#             size=n_fake,
#             replace=False,
#         )
#     )

#     fake_neuron_indices[key] = positions

#     mu = 0.0
#     sigma = 1.0

#     train_data = insert_fake_at_positions(
#         train_data, positions, rng, mu, sigma
#     )
#     test_data = insert_fake_at_positions(
#         test_data, positions, rng, mu, sigma
#     )

#     return train_data, test_data, positions


# def standardize_train_test(train_x, test_x, eps=1e-6):
#     mu = train_x.mean(axis=0, keepdims=True)
#     sd = train_x.std(axis=0, keepdims=True) + eps
#     train_x_z = (train_x - mu) / sd
#     test_x_z = (test_x - mu) / sd
#     return train_x_z, test_x_z, mu, sd


# def get_torch_model(model):
#     torch_model = model.solver_.model
#     torch_model.split_outputs = False
#     torch_model.to(device)
#     torch_model.eval()
#     return torch_model


# def compute_attribution(model, neural, batch_size=256, num_samples=2000):
#     neural = torch.from_numpy(neural).float().to(device)
#     neural.requires_grad_(True)

#     torch_model = get_torch_model(model)

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=torch_model,
#         input_data=neural,
#         output_dimension=torch_model.num_output,
#         num_samples=num_samples,
#     )

#     attribution = method.compute_attribution_map(batch_size=batch_size)

#     jf = np.abs(attribution["jf"]).mean(axis=0)
#     jfinv = np.abs(attribution["jf-inv-svd"]).mean(axis=0)

#     del method
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     return {
#         "jf": jf,
#         "jfinv": jfinv,
#     }


# def binary_maps(jf, jfinv):
#     jf_bin = (zscore(jf, axis=None) > 0).astype(np.int32)
#     jfinv_bin = (zscore(jfinv, axis=None) > 0).astype(np.int32)
#     return jf_bin, jfinv_bin


# class LinearReconstructionDecoder(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)


# def get_latent(model, data_np):
#     z = model.transform(data_np)
#     return torch.tensor(z, dtype=torch.float32, device=device)


# def train_linear_decoder(train_latent, train_target, input_dim, output_dim,
#                          epochs=RECON_EPOCHS, batch_size=BATCH_SIZE, lr=RECON_LR):
#     decoder = LinearReconstructionDecoder(
#         input_dim=input_dim,
#         output_dim=output_dim
#     ).to(device)

#     opt = torch.optim.Adam(decoder.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()

#     ds = torch.utils.data.TensorDataset(train_latent, train_target)
#     dl = torch.utils.data.DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=False
#     )

#     decoder.train()
#     for ep in range(epochs):
#         total_loss = 0.0
#         for z_b, y_b in dl:
#             opt.zero_grad()
#             pred = decoder(z_b)
#             loss = loss_fn(pred, y_b)
#             loss.backward()
#             opt.step()
#             total_loss += loss.item() * z_b.size(0)

#         if (ep + 1) % 50 == 0 or ep == 0:
#             print(f"  decoder epoch {ep+1:03d} | loss={total_loss / len(ds):.6f}")

#     return decoder


# @torch.no_grad()
# def evaluate_reconstruction(decoder, test_latent, test_target_raw, mu, sd):
#     decoder.eval()
#     pred_z = decoder(test_latent)

#     pred_raw = pred_z.cpu().numpy() * sd + mu
#     true_raw = test_target_raw.cpu().numpy()

#     mse = float(np.mean((pred_raw - true_raw) ** 2))
#     rmse = float(np.sqrt(mse))
#     mae = float(np.mean(np.abs(pred_raw - true_raw)))

#     ss_res = np.sum((pred_raw - true_raw) ** 2)
#     ss_tot = np.sum((true_raw - true_raw.mean(axis=0, keepdims=True)) ** 2)
#     r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

#     corrs = []
#     for i in range(true_raw.shape[1]):
#         a = true_raw[:, i]
#         b = pred_raw[:, i]
#         if np.std(a) < 1e-12 or np.std(b) < 1e-12:
#             continue
#         corrs.append(np.corrcoef(a, b)[0, 1])

#     mean_corr = float(np.nanmean(corrs)) if len(corrs) > 0 else np.nan

#     return {
#         "mse": mse,
#         "rmse": rmse,
#         "mae": mae,
#         "r2": r2,
#         "mean_corr": mean_corr,
#         "pred_raw": pred_raw,
#         "true_raw": true_raw,
#     }


# def save_recon_preview(rat, mode, pred_raw, true_raw, save_dir, n_show=4, t_show=500):
#     os.makedirs(save_dir, exist_ok=True)

#     T = min(t_show, true_raw.shape[0])
#     chans = min(n_show, true_raw.shape[1])

#     fig, axs = plt.subplots(chans, 1, figsize=(14, 2.2 * chans), sharex=True)
#     if chans == 1:
#         axs = [axs]

#     for i in range(chans):
#         axs[i].plot(true_raw[:T, i], label="true")
#         axs[i].plot(pred_raw[:T, i], label="recon", alpha=0.8)
#         axs[i].set_ylabel(f"ch {i}")
#         axs[i].legend(loc="upper right", fontsize=8)

#     axs[-1].set_xlabel("time bin")
#     plt.suptitle(f"{rat} — {mode} — reconstruction preview", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"recon_preview_{mode}.png"), dpi=300)
#     plt.close()


# def save_comparison_plots(rat, results, fake_positions):
#     save_dir = os.path.join(RESULT_DIR, rat)
#     os.makedirs(save_dir, exist_ok=True)

#     fig, axs = plt.subplots(1, 2, figsize=(14, 5))

#     im0 = axs[0].imshow(results["clean"]["jfinv"], aspect="auto")
#     axs[0].set_title(f"{rat} — clean — JF-inv")
#     axs[0].set_xlabel("Input neurons")
#     axs[0].set_ylabel("Latent dims")
#     plt.colorbar(im0, ax=axs[0])

#     im1 = axs[1].imshow(results["adv"]["jfinv"], aspect="auto")
#     axs[1].set_title(f"{rat} — adversarial — JF-inv")
#     axs[1].set_xlabel("Input neurons")
#     axs[1].set_ylabel("Latent dims")
#     plt.colorbar(im1, ax=axs[1])

#     for fpos in fake_positions:
#         axs[0].axvline(fpos, color="red", linestyle="--", linewidth=0.8)
#         axs[1].axvline(fpos, color="red", linestyle="--", linewidth=0.8)

#     plt.suptitle(f"{rat} — JF-inv attribution (raw)", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "jfinv_raw.png"), dpi=300)
#     plt.close()

#     clean_bin = (zscore(results["clean"]["jfinv"], axis=None) > 0).astype(int)
#     adv_bin = (zscore(results["adv"]["jfinv"], axis=None) > 0).astype(int)

#     fig, axs = plt.subplots(1, 2, figsize=(14, 5))

#     im0 = axs[0].imshow(clean_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     axs[0].set_title(f"{rat} — clean — JF-inv (binary)")
#     axs[0].set_xlabel("Input neurons")
#     axs[0].set_ylabel("Latent dims")
#     plt.colorbar(im0, ax=axs[0])

#     im1 = axs[1].imshow(adv_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     axs[1].set_title(f"{rat} — adversarial — JF-inv (binary)")
#     axs[1].set_xlabel("Input neurons")
#     axs[1].set_ylabel("Latent dims")
#     plt.colorbar(im1, ax=axs[1])

#     for fpos in fake_positions:
#         axs[0].axvline(fpos, color="red", linestyle="--", linewidth=0.8)
#         axs[1].axvline(fpos, color="red", linestyle="--", linewidth=0.8)

#     plt.suptitle(f"{rat} — JF-inv attribution (binary)", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "jfinv_binary.png"), dpi=300)
#     plt.close()

#     if len(fake_positions) > 0:
#         fake_clean = results["clean"]["jfinv"][:, fake_positions]
#         fake_adv = results["adv"]["jfinv"][:, fake_positions]

#         fake_clean_bin = (zscore(fake_clean, axis=None) > 0).astype(int)
#         fake_adv_bin = (zscore(fake_adv, axis=None) > 0).astype(int)

#         fig, axs = plt.subplots(1, 2, figsize=(10, 4))

#         im0 = axs[0].imshow(fake_clean_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#         axs[0].set_title(f"{rat} — clean — fake neurons")
#         axs[0].set_xlabel(f"positions: {list(fake_positions)}")
#         axs[0].set_ylabel("Latent dims")
#         plt.colorbar(im0, ax=axs[0])

#         im1 = axs[1].imshow(fake_adv_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#         axs[1].set_title(f"{rat} — adversarial — fake neurons")
#         axs[1].set_xlabel(f"positions: {list(fake_positions)}")
#         axs[1].set_ylabel("Latent dims")
#         plt.colorbar(im1, ax=axs[1])

#         plt.suptitle(f"{rat} — fake neurons binary", fontsize=12)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, "fake_binary.png"), dpi=300)
#         plt.close()

#     print(f"[{rat}] Figures saved -> {save_dir}/")


# def print_metrics(rat, scores):
#     print()
#     print("=" * 72)
#     print(f"Session: {rat}")
#     print("=" * 72)
#     for mode in ["clean", "adv"]:
#         if mode in scores:
#             m = scores[mode]
#             print(f"\n{mode.upper()}")
#             print(f"  MSE       : {m['mse']:.6f}")
#             print(f"  RMSE      : {m['rmse']:.6f}")
#             print(f"  MAE       : {m['mae']:.6f}")
#             print(f"  R²        : {m['r2']:.4f}")
#             print(f"  Corr(mean): {m['mean_corr']:.4f}")
#     print("=" * 72)
#     print()


# # ============================================================
# # MAIN PIPELINE
# # ============================================================
# attribution_store = {}
# fake_positions_store = {}

# for training_mode, adv in [
#     ("clean", False),
#     ("adversarial", True),
# ]:
#     print("=" * 80)
#     print(training_mode)
#     print("=" * 80)

#     for name in rats:
#         print(f"\nTraining {name} ...")

#         spikes, position = load_local_rat_dataset(name)
#         split = int(0.8 * len(spikes))

#         train_data = spikes[:split]
#         test_data = spikes[split:]

#         train_label = position[:split, :2]
#         test_label = position[split:, :2]

#         key = f"{name}_adv" if adv else name
#         train_data, test_data, fake_positions = add_fake_neurons(
#             train_data, test_data, key
#         )
#         fake_positions_store[name] = fake_positions
#         print("Fake neurons:", fake_positions)

#         setup_seed(0)
#         model = CEBRA(
#             batch_size=BATCH_SIZE,
#             temperature=0.4,
#             model_architecture="offset36-model-more-dropout",
#             time_offsets=4,
#             max_iterations=MAX_ITER,
#             output_dimension=OUTPUT_DIM,
#             verbose=True,
#             training_mode=training_mode,
#             adv_alpha=adv_epsilon / 5,
#             adv_epsilon=adv_epsilon,
#             adv_steps=10,
#             attack_norm="l2",
#             jacobian_weight=0.5,
#             adv_aggregate=False,
#         )
#         model.fit(train_data, train_label)

#         # ------------------------------------------------------------
#         # Reconstruction probe:
#         # latent -> original neural activity
#         # ------------------------------------------------------------
#         train_x_z, test_x_z, mu, sd = standardize_train_test(train_data, test_data)

#         train_latent = get_latent(model, train_data)
#         test_latent = get_latent(model, test_data)

#         train_target = torch.tensor(train_x_z, dtype=torch.float32, device=device)
#         test_target_raw = torch.tensor(test_data, dtype=torch.float32, device=device)

#         decoder = train_linear_decoder(
#             train_latent=train_latent,
#             train_target=train_target,
#             input_dim=train_latent.shape[1],
#             output_dim=train_target.shape[1],
#             epochs=RECON_EPOCHS,
#             batch_size=BATCH_SIZE,
#             lr=RECON_LR,
#         )

#         recon_metrics = evaluate_reconstruction(
#             decoder=decoder,
#             test_latent=test_latent,
#             test_target_raw=test_target_raw,
#             mu=mu,
#             sd=sd,
#         )

#         mode = "adv" if adv else "clean"
#         scores[mode][name] = recon_metrics

#         print(
#             f"Reconstruction -> "
#             f"MSE={recon_metrics['mse']:.6f} | "
#             f"RMSE={recon_metrics['rmse']:.6f} | "
#             f"MAE={recon_metrics['mae']:.6f} | "
#             f"R²={recon_metrics['r2']:.4f} | "
#             f"Corr(mean)={recon_metrics['mean_corr']:.4f}"
#         )

#         save_recon_preview(
#             rat=name,
#             mode=mode,
#             pred_raw=recon_metrics["pred_raw"],
#             true_raw=recon_metrics["true_raw"],
#             save_dir=os.path.join(RESULT_DIR, name),
#             n_show=4,
#             t_show=500,
#         )

#         # ------------------------------------------------------------
#         # Attribution
#         # ------------------------------------------------------------
#         result = compute_attribution(model, train_data)

#         if name not in attribution_store:
#             attribution_store[name] = {}
#         attribution_store[name][mode] = result

#         if len(fake_positions) > 0:
#             fake_score = result["jfinv"][:, fake_positions]
#             print("\nFake neuron importance:")
#             print(fake_score.mean(axis=0))
#         else:
#             print("\nNo fake neurons to inspect.")

#         del model
#         del decoder
#         del train_latent
#         del test_latent
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# print("\nSaving comparison plots...")
# for name in rats:
#     if "clean" in attribution_store.get(name, {}) and "adv" in attribution_store.get(name, {}):
#         save_comparison_plots(
#             rat=name,
#             results=attribution_store[name],
#             fake_positions=fake_positions_store.get(name, np.array([], dtype=int)),
#         )
#     else:
#         print(f"[{name}] skip (missing clean or adv result)")

# print()
# print("=" * 60)
# print("Reconstruction Results")
# print("=" * 60)
# for mode in scores:
#     print(f"\n{mode}")
#     for rat in rats:
#         if rat in scores[mode]:
#             m = scores[mode][rat]
#             print(
#                 f"  {rat:10s}: "
#                 f"MSE={m['mse']:.6f}, "
#                 f"RMSE={m['rmse']:.6f}, "
#                 f"MAE={m['mae']:.6f}, "
#                 f"R²={m['r2']:.4f}, "
#                 f"Corr(mean)={m['mean_corr']:.4f}"
#             )

# print()
# print("Finished.")



# import sys
# import os
# import shutil
# import subprocess
# import random
# from pathlib import Path

# import h5py
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d

# # ============================================================
# # CONFIG
# # ============================================================
# DATASET_DIR = "monkey_dataset"

# sessions = [
#     Path(DATASET_DIR) / "Jango_20150730_001.mat",
#     # Path(DATASET_DIR) / "Jango_20150731_001.mat",
#     # Path(DATASET_DIR) / "Jango_20150801_001.mat",
#     # Path(DATASET_DIR) / "Jango_20150805_001.mat",
#     # Path(DATASET_DIR) / "Jango_20150806_001.mat",
#     # Path(DATASET_DIR) / "Jango_20150807_001.mat",
# ]

# RESULT_DIR   = "results_monkey_recon"
# REPO_DIR     = "CEBRA"

# N_FAKE       = 0
# adv_epsilon  = 0.5
# MAX_ITER     = 3000
# OUTPUT_DIM   = 48
# BATCH_SIZE   = 512
# N_ELEC       = 96
# BIN_SIZE_MS  = 50
# SMOOTH_SD_MS = 100

# RECON_EPOCHS = 6000
# RECON_LR     = 1e-2
# RECON_HIDDEN = 0   # linear decoder only

# # ============================================================
# # SETUP
# # ============================================================
# os.makedirs(RESULT_DIR, exist_ok=True)

# # ── Patch CEBRA ─────────────────────────────────────────────
# if not os.path.exists(REPO_DIR):
#     subprocess.run([
#         "git", "clone",
#         "https://github.com/AdaptiveMotorControlLab/CEBRA.git",
#     ], check=True)

# shutil.copy("base.py", os.path.join(REPO_DIR, "cebra/solver/base.py"))
# shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py"))
# shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/cebra.py"))

# base_path = os.path.join(REPO_DIR, "cebra/solver/base.py")
# with open(base_path, "r") as f:
#     content = f.read()
# if "AuxiliaryVariableSolver" not in content:
#     with open(base_path, "a") as f:
#         f.write("\nclass AuxiliaryVariableSolver(Solver):\n    pass\n")
#         f.write("\nclass DiscreteAuxiliaryVariableSolver(Solver):\n    pass\n")

# print("Patch applied.")

# sys.path.insert(0, REPO_DIR)
# if "cebra" in sys.modules:
#     del sys.modules["cebra"]

# import cebra
# from cebra import CEBRA

# print("CEBRA:", cebra.__version__)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)


# # ============================================================
# # UTILITIES
# # ============================================================
# def setup_seed(seed=42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def read_h5_scalar(obj):
#     arr = obj[:]
#     return float(arr.flat[0])


# def read_h5_1d(obj):
#     return obj[:].flatten()


# def load_monkey_session(mat_path,
#                         bin_size_ms=BIN_SIZE_MS,
#                         smooth_sd_ms=SMOOTH_SD_MS,
#                         n_elec=N_ELEC):
#     with h5py.File(mat_path, "r") as f:
#         xds = f["xds"]

#         raw_bin_ms = read_h5_scalar(xds["bin_width"]) * 1000
#         print(f"  raw bin_width = {raw_bin_ms:.2f} ms")

#         sc_raw = xds["spike_counts"][:]
#         if sc_raw.shape[0] < sc_raw.shape[1]:
#             sc_raw = sc_raw.T
#         T_raw, N_raw = sc_raw.shape
#         print(f"  spike_counts raw: ({T_raw}, {N_raw})")

#         cp_raw = xds["curs_p"][:]
#         if cp_raw.shape[0] < cp_raw.shape[1]:
#             cp_raw = cp_raw.T  # -> (T, 2)

#         trial_start = read_h5_1d(xds["trial_start_time"])  # seconds
#         trial_end   = read_h5_1d(xds["trial_end_time"])    # seconds

#         trial_results = [chr(int(v)) for v in xds["trial_result"][:].ravel()]

#         assert bin_size_ms % raw_bin_ms < 1e-6, \
#             f"bin_size_ms={bin_size_ms} must be a multiple of raw_bin_ms={raw_bin_ms}"
#         ratio = int(round(bin_size_ms / raw_bin_ms))

#         T_trim = (T_raw // ratio) * ratio
#         sc_trimmed = sc_raw[:T_trim, :]
#         cp_trimmed = cp_raw[:T_trim, :]

#         sc_binned = sc_trimmed.reshape(T_trim // ratio, ratio, N_raw).sum(axis=1)
#         cp_binned = cp_trimmed.reshape(T_trim // ratio, ratio, 2).mean(axis=1)
#         T_binned = sc_binned.shape[0]
#         print(f"  after {bin_size_ms}ms binning: ({T_binned}, {N_raw})")

#         smooth_sd_bins = smooth_sd_ms / bin_size_ms
#         sc_smooth = gaussian_filter1d(
#             sc_binned.astype(np.float32), sigma=smooth_sd_bins, axis=0
#         )
#         print(f"  Gaussian smoothing: SD={smooth_sd_ms}ms = {smooth_sd_bins:.1f} bins")

#         if N_raw < n_elec:
#             pad = np.zeros((T_binned, n_elec - N_raw), dtype=np.float32)
#             sc_smooth = np.concatenate([sc_smooth, pad], axis=1)
#             print(f"  zero-padded: {N_raw} -> {n_elec} channels")

#         spikes_list = []
#         curs_list = []
#         n_ok = 0

#         for t_start, t_end, t_res in zip(trial_start, trial_end, trial_results):
#             if t_res != "R":
#                 continue
#             if np.isnan(t_start) or np.isnan(t_end):
#                 continue

#             bin_start = int(round(t_start * 1000 / bin_size_ms))
#             bin_end   = int(round(t_end   * 1000 / bin_size_ms))

#             bin_start = max(0, bin_start)
#             bin_end   = min(T_binned, bin_end)
#             if bin_end <= bin_start:
#                 continue

#             spikes_list.append(sc_smooth[bin_start:bin_end, :].astype(np.float32))
#             curs_list.append(cp_binned[bin_start:bin_end, :].astype(np.float32))
#             n_ok += 1

#         print(f"  extracted {n_ok} successful trials (R)")
#         if n_ok == 0:
#             raise RuntimeError("No successful trial found!")

#         return spikes_list, curs_list


# def concat_trials(spikes_list, curs_list):
#     spikes = np.concatenate(spikes_list, axis=0)
#     curs = np.concatenate(curs_list, axis=0)
#     print(f"\nConcatenated: spikes={spikes.shape}  cursor={curs.shape}")
#     return spikes, curs


# def insert_fake_at_positions(x, positions, rng, mu=0.0, sigma=1.0):
#     n_total = x.shape[1] + len(positions)
#     is_fake = np.zeros(n_total, dtype=bool)
#     is_fake[positions] = True
#     fake_values = rng.normal(loc=mu, scale=sigma, size=(x.shape[0], len(positions)))
#     combined = np.zeros((x.shape[0], n_total), dtype=np.float32)
#     combined[:, is_fake] = fake_values
#     combined[:, ~is_fake] = x
#     return combined


# def add_fake_neurons(train_data, test_data, key, fake_store, n_fake=N_FAKE):
#     if n_fake == 0:
#         fake_positions = np.array([], dtype=int)
#         fake_store[key] = fake_positions
#         return train_data, test_data, fake_positions

#     rng = np.random.default_rng(0)
#     n_real = train_data.shape[1]
#     positions = np.sort(rng.choice(n_real + n_fake, size=n_fake, replace=False))
#     fake_store[key] = positions

#     mu = train_data.mean()
#     sigma = train_data.std() + 1e-6

#     train_data = insert_fake_at_positions(train_data, positions, rng, mu=mu, sigma=sigma)
#     test_data  = insert_fake_at_positions(test_data,  positions, rng, mu=mu, sigma=sigma)
#     return train_data, test_data, positions


# def standardize_train_test(train_x, test_x, eps=1e-6):
#     mu = train_x.mean(axis=0, keepdims=True)
#     sd = train_x.std(axis=0, keepdims=True) + eps
#     train_x_z = (train_x - mu) / sd
#     test_x_z = (test_x - mu) / sd
#     return train_x_z, test_x_z, mu, sd


# def get_latent(model, data_np):
#     z = model.transform(data_np)
#     return torch.tensor(z, dtype=torch.float32, device=device)


# class LinearReconstructionDecoder(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)


# def train_linear_decoder(train_latent, train_target, input_dim, output_dim,
#                          epochs=RECON_EPOCHS, batch_size=BATCH_SIZE, lr=RECON_LR):
#     decoder = LinearReconstructionDecoder(input_dim=input_dim, output_dim=output_dim).to(device)
#     opt = torch.optim.Adam(decoder.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()

#     ds = torch.utils.data.TensorDataset(train_latent, train_target)
#     dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

#     decoder.train()
#     for ep in range(epochs):
#         total_loss = 0.0
#         for z_b, y_b in dl:
#             opt.zero_grad()
#             pred = decoder(z_b)
#             loss = loss_fn(pred, y_b)
#             loss.backward()
#             opt.step()
#             total_loss += loss.item() * z_b.size(0)

#         if (ep + 1) % 50 == 0 or ep == 0:
#             print(f"  decoder epoch {ep+1:03d} | loss={total_loss / len(ds):.6f}")

#     return decoder


# @torch.no_grad()
# def evaluate_reconstruction(decoder, test_latent, test_target_raw, mu, sd):
#     decoder.eval()
#     pred_z = decoder(test_latent)

#     pred_raw = pred_z.cpu().numpy() * sd + mu
#     true_raw = test_target_raw.cpu().numpy()

#     mse = float(np.mean((pred_raw - true_raw) ** 2))
#     rmse = float(np.sqrt(mse))
#     mae = float(np.mean(np.abs(pred_raw - true_raw)))

#     ss_res = np.sum((pred_raw - true_raw) ** 2)
#     ss_tot = np.sum((true_raw - true_raw.mean(axis=0, keepdims=True)) ** 2)
#     r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

#     corrs = []
#     for i in range(true_raw.shape[1]):
#         a = true_raw[:, i]
#         b = pred_raw[:, i]
#         if np.std(a) < 1e-12 or np.std(b) < 1e-12:
#             continue
#         corrs.append(np.corrcoef(a, b)[0, 1])
#     mean_corr = float(np.nanmean(corrs)) if len(corrs) > 0 else np.nan

#     return {
#         "mse": mse,
#         "rmse": rmse,
#         "mae": mae,
#         "r2": r2,
#         "mean_corr": mean_corr,
#         "pred_raw": pred_raw,
#         "true_raw": true_raw,
#     }


# def save_recon_plot(session_name, mode, pred_raw, true_raw, save_dir, n_show=4, t_show=500):
#     os.makedirs(save_dir, exist_ok=True)

#     T = min(t_show, true_raw.shape[0])
#     chans = min(n_show, true_raw.shape[1])

#     fig, axs = plt.subplots(chans, 1, figsize=(14, 2.2 * chans), sharex=True)
#     if chans == 1:
#         axs = [axs]

#     for i in range(chans):
#         axs[i].plot(true_raw[:T, i], label="true")
#         axs[i].plot(pred_raw[:T, i], label="recon", alpha=0.8)
#         axs[i].set_ylabel(f"ch {i}")
#         axs[i].legend(loc="upper right", fontsize=8)

#     axs[-1].set_xlabel("time bin")
#     plt.suptitle(f"{session_name} — {mode} — reconstruction preview", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"recon_preview_{mode}.png"), dpi=300)
#     plt.close()


# def print_metrics(session_name, scores):
#     print()
#     print("=" * 72)
#     print(f"Session: {session_name}")
#     print("=" * 72)
#     for mode in ["clean", "adv"]:
#         if mode in scores:
#             m = scores[mode]
#             print(f"\n{mode.upper()}")
#             print(f"  MSE       : {m['mse']:.6f}")
#             print(f"  RMSE      : {m['rmse']:.6f}")
#             print(f"  MAE       : {m['mae']:.6f}")
#             print(f"  R²        : {m['r2']:.4f}")
#             print(f"  Corr(mean): {m['mean_corr']:.4f}")
#     print("=" * 72)
#     print()


# # ============================================================
# # MAIN PIPELINE
# # ============================================================
# for SESSION_FILE in sessions:
#     session_name = Path(SESSION_FILE).stem
#     print(f"\nLoading session: {session_name}")

#     spikes_list, curs_list = load_monkey_session(SESSION_FILE)
#     spikes, cursor = concat_trials(spikes_list, curs_list)

#     split = int(0.8 * len(spikes))
#     train_data = spikes[:split]
#     test_data = spikes[split:]
#     train_label = cursor[:split, :2]
#     test_label = cursor[split:, :2]

#     print(f"Train: {train_data.shape}  Test: {test_data.shape}")
#     print(f"Label: {train_label.shape}")

#     scores = {"clean": {}, "adv": {}}
#     fake_store = {}

#     for training_mode, adv in [("clean", False), ("adversarial", True)]:
#         print("\n" + "=" * 80)
#         print(f"Training mode: {training_mode}")
#         print("=" * 80)

#         key = f"{session_name}_adv" if adv else session_name

#         tr_data, te_data, fake_positions = add_fake_neurons(
#             train_data.copy(), test_data.copy(), key, fake_store
#         )

#         if len(fake_positions) > 0:
#             print(f"Fake neurons ({N_FAKE}) at: {fake_positions}")
#         else:
#             print("No fake neurons.")

#         setup_seed(0)

#         model = CEBRA(
#             batch_size=BATCH_SIZE,
#             temperature=0.4,
#             model_architecture="offset36-model-more-dropout",
#             time_offsets=4,
#             max_iterations=MAX_ITER,
#             output_dimension=OUTPUT_DIM,
#             verbose=True,
#             training_mode=training_mode,
#             adv_alpha=adv_epsilon / 5,
#             adv_epsilon=adv_epsilon,
#             adv_steps=10,
#             attack_norm="l2",
#             jacobian_weight=0,
#             adv_aggregate=True,
#         )

#         model.fit(tr_data, train_label)

#         # ------------------------------------------------------------
#         # Reconstruction probe: latent -> original neural activity
#         # ------------------------------------------------------------
#         tr_x_z, te_x_z, mu, sd = standardize_train_test(tr_data, te_data)

#         train_latent = get_latent(model, tr_data)
#         test_latent = get_latent(model, te_data)

#         train_target = torch.tensor(tr_x_z, dtype=torch.float32, device=device)
#         test_target_raw = torch.tensor(te_data, dtype=torch.float32, device=device)

#         decoder = train_linear_decoder(
#             train_latent=train_latent,
#             train_target=train_target,
#             input_dim=train_latent.shape[1],
#             output_dim=train_target.shape[1],
#             epochs=RECON_EPOCHS,
#             batch_size=BATCH_SIZE,
#             lr=RECON_LR,
#         )

#         recon_metrics = evaluate_reconstruction(
#             decoder=decoder,
#             test_latent=test_latent,
#             test_target_raw=test_target_raw,
#             mu=mu,
#             sd=sd,
#         )

#         mode = "adv" if adv else "clean"
#         scores[mode] = recon_metrics

#         print(
#             f"Reconstruction -> "
#             f"MSE={recon_metrics['mse']:.6f} | "
#             f"RMSE={recon_metrics['rmse']:.6f} | "
#             f"MAE={recon_metrics['mae']:.6f} | "
#             f"R²={recon_metrics['r2']:.4f} | "
#             f"Corr(mean)={recon_metrics['mean_corr']:.4f}"
#         )

#         save_dir = os.path.join(RESULT_DIR, session_name)
#         save_recon_plot(
#             session_name=session_name,
#             mode=mode,
#             pred_raw=recon_metrics["pred_raw"],
#             true_raw=recon_metrics["true_raw"],
#             save_dir=save_dir,
#             n_show=4,
#             t_show=500,
#         )

#         del model
#         del decoder
#         torch.cuda.empty_cache()

#     print_metrics(session_name, scores)

# print("Finished.")
