import sys
import os
import shutil
import subprocess
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import zscore


SESSION_FILE   = "monkey_dataset/Jango_20150730_001.mat"
RESULT_DIR     = "results_monkey"
REPO_DIR       = "CEBRA"

MAX_TIMESTEPS  = 50_000
N_FAKE         = 0

adv_epsilon    = 0.5
MAX_ITER       = 1000
OUTPUT_DIM     = 48
BATCH_SIZE     = 512  
# ============================================================

os.makedirs(RESULT_DIR, exist_ok=True)

# ── Patch CEBRA ─────────────────────────────────────────────
if not os.path.exists(REPO_DIR):
    subprocess.run([
        "git", "clone",
        "https://github.com/AdaptiveMotorControlLab/CEBRA.git",
    ], check=True)

shutil.copy("base.py",
            os.path.join(REPO_DIR, "cebra/solver/base.py"))
shutil.copy("cebra.py",
            os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py"))
shutil.copy("cebra.py",
            os.path.join(REPO_DIR, "cebra/cebra.py"))

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
import cebra.attribution
from cebra import CEBRA
from decoder import TwoLayerMLP

print("CEBRA:", cebra.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ── Helpers ─────────────────────────────────────────────────
def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_monkey_session(mat_path, max_timesteps=None):
    with h5py.File(mat_path, "r") as f:
        xds = f["xds"]

        spike_counts = xds["spike_counts"][:] 
        curs_p       = xds["curs_p"][:]       

        if spike_counts.shape[0] < spike_counts.shape[1]:
            spike_counts = spike_counts.T
        if curs_p.shape[0] < curs_p.shape[1]:
            curs_p = curs_p.T

        spikes   = spike_counts.astype(np.float32)  # (T, N)
        position = curs_p.astype(np.float32)          # (T, 2)

        bin_width = float(xds["bin_width"][0, 0]) 

    print(f"Loaded: {Path(mat_path).name}")
    print(f"  bin_width = {bin_width*1000:.1f} ms")
    print(f"  raw spikes={spikes.shape}  position={position.shape}")
    print(f"  duration  = {spikes.shape[0] * bin_width:.1f} s  "
          f"({spikes.shape[0]} timesteps)")

    if max_timesteps is not None and spikes.shape[0] > max_timesteps:
        spikes   = spikes[:max_timesteps]
        position = position[:max_timesteps]
        print(f"  → subsampled to {max_timesteps} timesteps "
              f"({max_timesteps * bin_width:.1f} s)")

    print(f"  final: spikes={spikes.shape}  position={position.shape}")
    return spikes, position


def insert_fake_at_positions(x, positions, rng, mu=0.0, sigma=1.0):
    n_total = x.shape[1] + len(positions)
    is_fake = np.zeros(n_total, dtype=bool)
    is_fake[positions] = True
    fake_values = rng.normal(loc=mu, scale=sigma,
                             size=(x.shape[0], len(positions)))
    combined = np.zeros((x.shape[0], n_total), dtype=np.float32)
    combined[:, is_fake]  = fake_values
    combined[:, ~is_fake] = x
    return combined


def add_fake_neurons(train_data, test_data, key, fake_store, n_fake=N_FAKE):
    if n_fake == 0:
        fake_positions = np.array([], dtype=int)
        fake_store[key] = fake_positions
        return train_data, test_data, fake_positions

    rng = np.random.default_rng(0)
    n_real = train_data.shape[1]
    positions = np.sort(
        rng.choice(n_real + n_fake, size=n_fake, replace=False))
    fake_store[key] = positions

    train_data = insert_fake_at_positions(train_data, positions, rng)
    test_data  = insert_fake_at_positions(test_data,  positions, rng)
    return train_data, test_data, positions


def get_torch_model(model):
    torch_model = model.solver_.model
    torch_model.split_outputs = False
    torch_model.to(device)
    torch_model.eval()
    return torch_model


def compute_decoder_score(model, train_data, test_data,
                           train_label, test_label):
    train_latent = torch.tensor(
        model.transform(train_data), dtype=torch.float32).to(device)
    test_latent  = torch.tensor(
        model.transform(test_data),  dtype=torch.float32).to(device)
    train_label_t = torch.tensor(train_label, dtype=torch.float32).to(device)
    test_label_t  = torch.tensor(test_label,  dtype=torch.float32).to(device)

    decoder = TwoLayerMLP(input_dim=OUTPUT_DIM, output_dim=2)
    setup_seed(0)
    decoder.fit(train_latent, train_label_t)
    with torch.no_grad():
        r2 = decoder.score(test_latent, test_label_t, device)
    return r2


def compute_attribution(model, neural_np, batch_size=256, num_samples=2000):
    neural = torch.from_numpy(neural_np).float().to(device)
    neural.requires_grad_(True)
    torch_model = get_torch_model(model)
    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=torch_model,
        input_data=neural,
        output_dimension=torch_model.num_output,
        num_samples=num_samples,
    )
    attr   = method.compute_attribution_map(batch_size=batch_size)
    jf     = np.abs(attr["jf"]).mean(axis=0)
    jfinv  = np.abs(attr["jf-inv-svd"]).mean(axis=0)
    del method
    torch.cuda.empty_cache()
    return {"jf": jf, "jfinv": jfinv}


def save_comparison_plots(session_name, results, fake_positions):
    save_dir = os.path.join(RESULT_DIR, session_name)
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axs[0].imshow(results["clean"]["jfinv"], aspect="auto")
    axs[0].set_title(f"{session_name} — clean — JF-inv")
    axs[0].set_xlabel("Input neurons (M1 channels)")
    axs[0].set_ylabel("Latent dims")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(results["adv"]["jfinv"], aspect="auto")
    axs[1].set_title(f"{session_name} — adversarial — JF-inv")
    axs[1].set_xlabel("Input neurons (M1 channels)")
    axs[1].set_ylabel("Latent dims")
    plt.colorbar(im1, ax=axs[1])

    for fpos in fake_positions:
        axs[0].axvline(fpos, color="red", linestyle="--", linewidth=0.8)
        axs[1].axvline(fpos, color="red", linestyle="--", linewidth=0.8)

    plt.suptitle(f"{session_name} — JF-inv attribution (raw)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "jfinv_raw.png"), dpi=300)
    plt.close()

    clean_bin = (zscore(results["clean"]["jfinv"], axis=None) > 0).astype(int)
    adv_bin   = (zscore(results["adv"]["jfinv"],   axis=None) > 0).astype(int)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axs[0].imshow(clean_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axs[0].set_title(f"{session_name} — clean — JF-inv (binary, z>0)")
    axs[0].set_xlabel("Input neurons (M1 channels)")
    axs[0].set_ylabel("Latent dims")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(adv_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axs[1].set_title(f"{session_name} — adversarial — JF-inv (binary, z>0)")
    axs[1].set_xlabel("Input neurons (M1 channels)")
    axs[1].set_ylabel("Latent dims")
    plt.colorbar(im1, ax=axs[1])

    for fpos in fake_positions:
        axs[0].axvline(fpos, color="red", linestyle="--", linewidth=0.8)
        axs[1].axvline(fpos, color="red", linestyle="--", linewidth=0.8)

    plt.suptitle(f"{session_name} — JF-inv attribution (binary, z-score>0)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "jfinv_binary.png"), dpi=300)
    plt.close()

    if len(fake_positions) > 0:
        fake_clean = results["clean"]["jfinv"][:, fake_positions]
        fake_adv   = results["adv"]["jfinv"][:,   fake_positions]
        fake_clean_bin = (zscore(fake_clean, axis=None) > 0).astype(int)
        fake_adv_bin   = (zscore(fake_adv,   axis=None) > 0).astype(int)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axs[0].imshow(fake_clean_bin, aspect="auto",
                             cmap="Greys", vmin=0, vmax=1)
        axs[0].set_title(f"{session_name} — clean — fake neurons")
        axs[0].set_xlabel(f"positions: {list(fake_positions)}")
        axs[0].set_ylabel("Latent dims")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(fake_adv_bin, aspect="auto",
                             cmap="Greys", vmin=0, vmax=1)
        axs[1].set_title(f"{session_name} — adversarial — fake neurons")
        axs[1].set_xlabel(f"positions: {list(fake_positions)}")
        axs[1].set_ylabel("Latent dims")
        plt.colorbar(im1, ax=axs[1])

        plt.suptitle(
            f"{session_name} — fake neurons (binary, z-score>0)\n"
            f"positions: {list(fake_positions)}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "fake_binary.png"), dpi=300)
        plt.close()

    print(f"[{session_name}] Figures saved -> {save_dir}/")
    print(f"  jfinv_raw.png | jfinv_binary.png"
          + (" | fake_binary.png" if len(fake_positions) > 0 else ""))


# ============================================================
# MAIN PIPELINE
# ============================================================
session_name = Path(SESSION_FILE).stem   # "Jango_20150730_001"

spikes, position = load_monkey_session(SESSION_FILE,
                                        max_timesteps=MAX_TIMESTEPS)

split       = int(0.8 * len(spikes))
train_data  = spikes[:split]
test_data   = spikes[split:]
train_label = position[:split, :2]
test_label  = position[split:, :2]

print(f"\nTrain: {train_data.shape}  Test: {test_data.shape}")
print(f"Label: {train_label.shape} (cursor x, y)")

scores            = {"clean": {}, "adv": {}}
attribution_store = {}
fake_store        = {}

for training_mode, adv in [("clean", False), ("adversarial", True)]:

    print("\n" + "=" * 80)
    print(f"Training mode: {training_mode}")
    print("=" * 80)

    key = f"{session_name}_adv" if adv else session_name

    tr_data, te_data, fake_positions = add_fake_neurons(
        train_data.copy(), test_data.copy(), key, fake_store)

    if len(fake_positions) > 0:
        print(f"Fake neurons ({N_FAKE}) at: {fake_positions}")
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
        adv_alpha=adv_epsilon / 5,
        adv_epsilon=adv_epsilon,
        adv_steps=10,
        attack_norm="l2",
        jacobian_weight=0.01,
        adv_aggregate=True,
    )

    model.fit(tr_data, train_label)

    # Decoder R²
    r2 = compute_decoder_score(
        model, tr_data, te_data, train_label, test_label)
    mode = "adv" if adv else "clean"
    scores[mode][session_name] = r2
    print(f"R² = {r2:.4f}")

    # Attribution
    result = compute_attribution(model, tr_data)
    attribution_store[mode] = result

    if len(fake_positions) > 0:
        fake_score = result["jfinv"][:, fake_positions]
        print("Fake neuron importance (mean per neuron):")
        print(fake_score.mean(axis=0))

    del model
    torch.cuda.empty_cache()

# ── Comparison plots ────────────────────────────────────────
if "clean" in attribution_store and "adv" in attribution_store:
    save_comparison_plots(
        session_name=session_name,
        results=attribution_store,
        fake_positions=fake_store.get(session_name, np.array([])),
    )

# ── Final summary ───────────────────────────────────────────
print()
print("=" * 60)
print("Decoder Results")
print("=" * 60)
for mode in scores:
    print(f"\n{mode}")
    for sess, r2 in scores[mode].items():
        print(f"  {sess}: {r2:.4f}")

print()
print("Finished.")

# import sys
# import os
# import shutil
# import subprocess
# import random
# from pathlib import Path

# import h5py
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from scipy.stats import zscore

# SESSION_FILE  = "monkey_dataset/Jango_20150805_001.mat"
# DATASET_DIR   = "monkey_dataset"
# RESULT_DIR    = "results_monkey"
# REPO_DIR      = "CEBRA"
# N_FAKE        = 0
# adv_epsilon   = 0.5
# MAX_ITER      = 2000
# OUTPUT_DIM    = 48
# BATCH_SIZE    = 512

# os.makedirs(RESULT_DIR, exist_ok=True)

# # ── Patch CEBRA ─────────────────────────────────────────────
# if not os.path.exists(REPO_DIR):
#     subprocess.run([
#         "git", "clone",
#         "https://github.com/AdaptiveMotorControlLab/CEBRA.git",
#     ], check=True)

# shutil.copy("base.py",
#             os.path.join(REPO_DIR, "cebra/solver/base.py"))
# shutil.copy("cebra.py",
#             os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py"))
# shutil.copy("cebra.py",
#             os.path.join(REPO_DIR, "cebra/cebra.py"))

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
# from decoder import TwoLayerMLP

# print("CEBRA:", cebra.__version__)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # ── Helpers ─────────────────────────────────────────────────
# def setup_seed(seed=42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def load_monkey_session(mat_path):
#     with h5py.File(mat_path, "r") as f:
#         xds = f["xds"]
#         spikes   = xds["spike_counts"][:].T.astype(np.float32)
#         position = xds["curs_p"][:].T.astype(np.float32) 

#     print(f"Loaded: {Path(mat_path).name}")
#     print(f"  spikes={spikes.shape}  position={position.shape}")
#     return spikes, position


# def insert_fake_at_positions(x, positions, rng, mu=0.0, sigma=1.0):
#     n_total = x.shape[1] + len(positions)
#     is_fake = np.zeros(n_total, dtype=bool)
#     is_fake[positions] = True
#     fake_values = rng.normal(loc=mu, scale=sigma,
#                              size=(x.shape[0], len(positions)))
#     combined = np.zeros((x.shape[0], n_total), dtype=np.float32)
#     combined[:, is_fake]  = fake_values
#     combined[:, ~is_fake] = x
#     return combined


# def add_fake_neurons(train_data, test_data, key,
#                      fake_store, n_fake=N_FAKE):
#     if n_fake == 0:
#         fake_positions = np.array([], dtype=int)
#         fake_store[key] = fake_positions
#         return train_data, test_data, fake_positions

#     rng = np.random.default_rng(0)
#     n_real = train_data.shape[1]
#     positions = np.sort(
#         rng.choice(n_real + n_fake, size=n_fake, replace=False))
#     fake_store[key] = positions

#     train_data = insert_fake_at_positions(train_data, positions, rng)
#     test_data  = insert_fake_at_positions(test_data,  positions, rng)
#     return train_data, test_data, positions


# def get_torch_model(model):
#     torch_model = model.solver_.model
#     torch_model.split_outputs = False
#     torch_model.to(device)
#     torch_model.eval()
#     return torch_model


# def compute_decoder_score(model, train_data, test_data,
#                            train_label, test_label):
#     train_latent = torch.tensor(
#         model.transform(train_data), dtype=torch.float32).to(device)
#     test_latent  = torch.tensor(
#         model.transform(test_data),  dtype=torch.float32).to(device)
#     train_label_t = torch.tensor(train_label, dtype=torch.float32).to(device)
#     test_label_t  = torch.tensor(test_label,  dtype=torch.float32).to(device)

#     decoder = TwoLayerMLP(input_dim=OUTPUT_DIM, output_dim=2)
#     setup_seed(0)
#     decoder.fit(train_latent, train_label_t)
#     with torch.no_grad():
#         r2 = decoder.score(test_latent, test_label_t, device)
#     return r2


# def compute_attribution(model, neural_np,
#                          batch_size=256, num_samples=2000):
#     neural = torch.from_numpy(neural_np).float().to(device)
#     neural.requires_grad_(True)
#     torch_model = get_torch_model(model)
#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=torch_model,
#         input_data=neural,
#         output_dimension=torch_model.num_output,
#         num_samples=num_samples,
#     )
#     attr = method.compute_attribution_map(batch_size=batch_size)
#     jf     = np.abs(attr["jf"]).mean(axis=0)
#     jfinv  = np.abs(attr["jf-inv-svd"]).mean(axis=0)
#     del method
#     torch.cuda.empty_cache()
#     return {"jf": jf, "jfinv": jfinv}


# def save_comparison_plots(session_name, results, fake_positions):
#     save_dir = os.path.join(RESULT_DIR, session_name)
#     os.makedirs(save_dir, exist_ok=True)

#     fig, axs = plt.subplots(1, 2, figsize=(14, 5))
#     im0 = axs[0].imshow(results["clean"]["jfinv"], aspect="auto")
#     axs[0].set_title(f"{session_name} — clean — JF-inv")
#     axs[0].set_xlabel("Input neurons")
#     axs[0].set_ylabel("Latent dims")
#     plt.colorbar(im0, ax=axs[0])

#     im1 = axs[1].imshow(results["adv"]["jfinv"], aspect="auto")
#     axs[1].set_title(f"{session_name} — adversarial — JF-inv")
#     axs[1].set_xlabel("Input neurons")
#     axs[1].set_ylabel("Latent dims")
#     plt.colorbar(im1, ax=axs[1])

#     for fpos in fake_positions:
#         axs[0].axvline(fpos, color="red", linestyle="--", linewidth=0.8)
#         axs[1].axvline(fpos, color="red", linestyle="--", linewidth=0.8)

#     plt.suptitle(f"{session_name} — JF-inv (raw)", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "jfinv_raw.png"), dpi=300)
#     plt.close()

#     clean_bin = (zscore(results["clean"]["jfinv"], axis=None) > 0).astype(int)
#     adv_bin   = (zscore(results["adv"]["jfinv"],   axis=None) > 0).astype(int)

#     fig, axs = plt.subplots(1, 2, figsize=(14, 5))
#     im0 = axs[0].imshow(clean_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     axs[0].set_title(f"{session_name} — clean — JF-inv (binary, z>0)")
#     axs[0].set_xlabel("Input neurons")
#     axs[0].set_ylabel("Latent dims")
#     plt.colorbar(im0, ax=axs[0])

#     im1 = axs[1].imshow(adv_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     axs[1].set_title(f"{session_name} — adversarial — JF-inv (binary, z>0)")
#     axs[1].set_xlabel("Input neurons")
#     axs[1].set_ylabel("Latent dims")
#     plt.colorbar(im1, ax=axs[1])

#     for fpos in fake_positions:
#         axs[0].axvline(fpos, color="red", linestyle="--", linewidth=0.8)
#         axs[1].axvline(fpos, color="red", linestyle="--", linewidth=0.8)

#     plt.suptitle(f"{session_name} — JF-inv (binary, z-score>0)", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "jfinv_binary.png"), dpi=300)
#     plt.close()

#     if len(fake_positions) > 0:
#         fake_clean = results["clean"]["jfinv"][:, fake_positions]
#         fake_adv   = results["adv"]["jfinv"][:,   fake_positions]
#         fake_clean_bin = (zscore(fake_clean, axis=None) > 0).astype(int)
#         fake_adv_bin   = (zscore(fake_adv,   axis=None) > 0).astype(int)

#         fig, axs = plt.subplots(1, 2, figsize=(10, 4))
#         im0 = axs[0].imshow(fake_clean_bin, aspect="auto",
#                              cmap="Greys", vmin=0, vmax=1)
#         axs[0].set_title(f"{session_name} — clean — fake neurons")
#         axs[0].set_xlabel(f"positions: {list(fake_positions)}")
#         axs[0].set_ylabel("Latent dims")
#         plt.colorbar(im0, ax=axs[0])

#         im1 = axs[1].imshow(fake_adv_bin, aspect="auto",
#                              cmap="Greys", vmin=0, vmax=1)
#         axs[1].set_title(f"{session_name} — adversarial — fake neurons")
#         axs[1].set_xlabel(f"positions: {list(fake_positions)}")
#         axs[1].set_ylabel("Latent dims")
#         plt.colorbar(im1, ax=axs[1])

#         plt.suptitle(
#             f"{session_name} — fake neurons (binary, z-score>0)\n"
#             f"positions: {list(fake_positions)}", fontsize=12)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, "fake_binary.png"), dpi=300)
#         plt.close()

#     print(f"[{session_name}] Figures saved -> {save_dir}/")


# # ============================================================
# # MAIN PIPELINE
# # ============================================================
# session_name = Path(SESSION_FILE).stem 

# spikes, position = load_monkey_session(SESSION_FILE)

# MAX_TIMESTEPS = 50_000 
# if len(spikes) > MAX_TIMESTEPS:
#     print(f"Subsampling from {len(spikes)} to {MAX_TIMESTEPS} timesteps...")
#     spikes   = spikes[:MAX_TIMESTEPS]
#     position = position[:MAX_TIMESTEPS]

# split        = int(0.8 * len(spikes))
# train_data   = spikes[:split]
# test_data    = spikes[split:]
# train_label  = position[:split,  :2]
# test_label   = position[split:,  :2]

# print(f"\nTrain: {train_data.shape}  Test: {test_data.shape}")

# scores             = {"clean": {}, "adv": {}}
# attribution_store  = {}
# fake_store         = {}

# for training_mode, adv in [("clean", False), ("adversarial", True)]:

#     print("\n" + "=" * 80)
#     print(f"Training mode: {training_mode}")
#     print("=" * 80)

#     key = f"{session_name}_adv" if adv else session_name

#     tr_data, te_data, fake_positions = add_fake_neurons(
#         train_data.copy(), test_data.copy(), key, fake_store)

#     if len(fake_positions) > 0:
#         print(f"Fake neurons ({N_FAKE}) at positions: {fake_positions}")
#     else:
#         print("No fake neurons.")

#     setup_seed(0)

#     model = CEBRA(
#         batch_size=2048 , #BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=5000 , #MAX_ITER,
#         output_dimension=48 , #OUTPUT_DIM,
#         verbose=True,
#         training_mode=training_mode,
#         adv_alpha=0.03 ,#adv_epsilon / 5,
#         adv_epsilon=0.2 , #adv_epsilon,
#         adv_steps=10,
#         attack_norm="l2",
#         jacobian_weight=0,
#         adv_aggregate=False,
#     )

#     model.fit(tr_data, train_label)

#     # Decoder R²
#     r2 = compute_decoder_score(
#         model, tr_data, te_data, train_label, test_label)
#     mode = "adv" if adv else "clean"
#     scores[mode][session_name] = r2
#     print(f"R² = {r2:.4f}")

#     # Attribution
#     result = compute_attribution(model, tr_data)
#     attribution_store[mode] = result

#     # Fake neuron importance
#     if len(fake_positions) > 0:
#         fake_score = result["jfinv"][:, fake_positions]
#         print("Fake neuron importance (mean per neuron):")
#         print(fake_score.mean(axis=0))

#     del model
#     torch.cuda.empty_cache()

# # ── Comparison plots ────────────────────────────────────────
# if "clean" in attribution_store and "adv" in attribution_store:
#     save_comparison_plots(
#         session_name=session_name,
#         results=attribution_store,
#         fake_positions=fake_store.get(session_name, np.array([])),
#     )

# # ── Final summary ───────────────────────────────────────────
# print()
# print("=" * 60)
# print("Decoder Results")
# print("=" * 60)
# for mode in scores:
#     print(f"\n{mode}")
#     for sess, r2 in scores[mode].items():
#         print(f"  {sess}: {r2:.4f}")

# print()
# print("Finished.")
