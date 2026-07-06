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
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIG
# ============================================================
# SESSION_FILE   = "monkey_dataset/Jango_20150730_001.mat"
DATASET_DIR = "monkey_dataset"

sessions = [
    # Path(DATASET_DIR) / "Jango_20150730_001.mat",
    Path(DATASET_DIR) / "Jango_20150731_001.mat",
    # Path(DATASET_DIR) / "Jango_20150801_001.mat",
    # Path(DATASET_DIR) / "Jango_20150805_001.mat",
    # Path(DATASET_DIR) / "Jango_20150806_001.mat",
    # Path(DATASET_DIR) / "Jango_20150807_001.mat",
]

RESULT_DIR     = "results_monkey"
REPO_DIR       = "CEBRA"
N_FAKE         = 10
adv_epsilon    = 0.5
MAX_ITER       = 1500
OUTPUT_DIM     = 48
BATCH_SIZE     = 512
N_ELEC         = 96
BIN_SIZE_MS    = 50
SMOOTH_SD_MS   = 100
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


# ── Data loading helpers ─────────────────────────────────────
def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_h5_scalar(obj):
    arr = obj[:]
    return float(arr.flat[0])


def read_h5_1d(obj):
    return obj[:].flatten()


def load_monkey_session(mat_path,
                        bin_size_ms=BIN_SIZE_MS,
                        smooth_sd_ms=SMOOTH_SD_MS,
                        n_elec=N_ELEC,
                        trial_result_filter=b'R'):
    with h5py.File(mat_path, "r") as f:
        xds = f["xds"]

        raw_bin_ms = read_h5_scalar(xds["bin_width"]) * 1000 
        print(f"  raw bin_width = {raw_bin_ms:.2f} ms")

        sc_raw = xds["spike_counts"][:]
        if sc_raw.shape[0] < sc_raw.shape[1]:
            sc_raw = sc_raw.T        
        T_raw, N_raw = sc_raw.shape
        print(f"  spike_counts raw: ({T_raw}, {N_raw})")

        cp_raw = xds["curs_p"][:]
        if cp_raw.shape[0] < cp_raw.shape[1]:
            cp_raw = cp_raw.T          # → (T, 2)

        trial_start = read_h5_1d(xds["trial_start_time"])   # seconds
        trial_end   = read_h5_1d(xds["trial_end_time"])     # seconds

        trial_results = [
            chr(int(v))
            for v in xds["trial_result"][:].ravel()
        ]

        # trial_result_raw = xds["trial_result"]
        # trial_results = []
        # for ref in trial_result_raw.flat:
        #     if isinstance(ref, h5py.Reference):
        #         char_data = f[ref][:]
        #         result_char = chr(int(char_data.flat[0]))
        #         trial_results.append(result_char.encode())
        #     else:
        #         trial_results.append(str(ref).encode())

        assert bin_size_ms % raw_bin_ms < 1e-6, \
            f"bin_size_ms={bin_size_ms} must mazrab raw_bin_ms={raw_bin_ms}"
        ratio = int(round(bin_size_ms / raw_bin_ms))

        T_trim = (T_raw // ratio) * ratio
        sc_trimmed = sc_raw[:T_trim, :]
        cp_trimmed = cp_raw[:T_trim, :]

        sc_binned = sc_trimmed.reshape(T_trim // ratio, ratio, N_raw).sum(axis=1)
        cp_binned = cp_trimmed.reshape(T_trim // ratio, ratio, 2).mean(axis=1)
        T_binned = sc_binned.shape[0]
        print(f"  after {bin_size_ms}ms binning: ({T_binned}, {N_raw})")

        # ── Gaussian smoothing ─────────────────────────────────
        smooth_sd_bins = smooth_sd_ms / bin_size_ms   # = 100/50 = 2 bins
        sc_smooth = gaussian_filter1d(
            sc_binned.astype(np.float32), sigma=smooth_sd_bins, axis=0)
        print(f"  Gaussian smoothing: SD={smooth_sd_ms}ms = {smooth_sd_bins:.1f} bins")

        if N_raw < n_elec:
            pad = np.zeros((T_binned, n_elec - N_raw), dtype=np.float32)
            sc_smooth = np.concatenate([sc_smooth, pad], axis=1)
            print(f"  zero-padded: {N_raw} → {n_elec} channels")

        spikes_list = []
        curs_list   = []
        n_ok = 0

        for i, (t_start, t_end, t_res) in enumerate(
                zip(trial_start, trial_end, trial_results)):

            if t_res != "R" :#trial_result_filter:
                continue
            if np.isnan(t_start) or np.isnan(t_end):
                continue

            bin_start = int(round(t_start * 1000 / bin_size_ms))
            bin_end   = int(round(t_end   * 1000 / bin_size_ms))

            bin_start = max(0, bin_start)
            bin_end   = min(T_binned, bin_end)
            if bin_end <= bin_start:
                continue

            spikes_list.append(
                sc_smooth[bin_start:bin_end, :].astype(np.float32))
            curs_list.append(
                cp_binned[bin_start:bin_end, :].astype(np.float32))
            n_ok += 1

        print(f"  extracted {n_ok} successful trials (R)")
        if n_ok == 0:
            raise RuntimeError("no found success trial!")

        return spikes_list, curs_list


def concat_trials(spikes_list, curs_list):
    spikes = np.concatenate(spikes_list, axis=0)   # (T_total, N)
    curs   = np.concatenate(curs_list,   axis=0)   # (T_total, 2)
    print(f"\nConcatenated: spikes={spikes.shape}  cursor={curs.shape}")
    return spikes, curs


# ── CEBRA helpers ────────────────────────────────────────────
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
    attr  = method.compute_attribution_map(batch_size=batch_size)
    jf    = np.abs(attr["jf"]).mean(axis=0)
    jfinv = np.abs(attr["jf-inv-svd"]).mean(axis=0)
    del method
    torch.cuda.empty_cache()
    return {"jf": jf, "jfinv": jfinv}


def save_comparison_plots(session_name, results, fake_positions):
    save_dir = os.path.join(RESULT_DIR, session_name)
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axs[0].imshow(results["clean"]["jfinv"], aspect="auto")
    axs[0].set_title(f"{session_name} — clean — JF-inv")
    axs[0].set_xlabel("M1 channels")
    axs[0].set_ylabel("Latent dims")
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(results["adv"]["jfinv"], aspect="auto")
    axs[1].set_title(f"{session_name} — adversarial — JF-inv")
    axs[1].set_xlabel("M1 channels")
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
    axs[0].set_title(f"{session_name} — clean — JF-inv binary")
    axs[0].set_xlabel("M1 channels")
    axs[0].set_ylabel("Latent dims")
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(adv_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axs[1].set_title(f"{session_name} — adversarial — JF-inv binary")
    axs[1].set_xlabel("M1 channels")
    axs[1].set_ylabel("Latent dims")
    plt.colorbar(im1, ax=axs[1])
    for fpos in fake_positions:
        axs[0].axvline(fpos, color="red", linestyle="--", linewidth=0.8)
        axs[1].axvline(fpos, color="red", linestyle="--", linewidth=0.8)
    plt.suptitle(f"{session_name} — JF-inv (binary, z>0)", fontsize=13)
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
        axs[0].set_title(f"clean — fake neurons")
        axs[0].set_xlabel(f"positions: {list(fake_positions)}")
        axs[0].set_ylabel("Latent dims")
        plt.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(fake_adv_bin, aspect="auto",
                             cmap="Greys", vmin=0, vmax=1)
        axs[1].set_title(f"adversarial — fake neurons")
        axs[1].set_xlabel(f"positions: {list(fake_positions)}")
        axs[1].set_ylabel("Latent dims")
        plt.colorbar(im1, ax=axs[1])
        plt.suptitle(f"{session_name} — fake neurons binary", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "fake_binary.png"), dpi=300)
        plt.close()

    print(f"[{session_name}] Saved -> {save_dir}/")


# ============================================================
# MAIN PIPELINE
# ============================================================
for SESSION_FILE in sessions:
    session_name = Path(SESSION_FILE).stem
    
    print(f"\nLoading session: {session_name}")
    spikes_list, curs_list = load_monkey_session(SESSION_FILE)
    
    spikes, cursor = concat_trials(spikes_list, curs_list)
    
    split       = int(0.8 * len(spikes))
    train_data  = spikes[:split]
    test_data   = spikes[split:]
    train_label = cursor[:split,  :2]
    test_label  = cursor[split:,  :2]
    
    print(f"Train: {train_data.shape}  Test: {test_data.shape}")
    print(f"Label: {train_label.shape}")
    
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
            jacobian_weight=0,
            adv_aggregate=False,
        )
    
        model.fit(tr_data, train_label)
    
        r2 = compute_decoder_score(
            model, tr_data, te_data, train_label, test_label)
        mode = "adv" if adv else "clean"
        scores[mode][session_name] = r2
        print(f"R² = {r2:.4f}")
    
        result = compute_attribution(model, tr_data)
        attribution_store[mode] = result
    
        if len(fake_positions) > 0:
            fake_score = result["jfinv"][:, fake_positions]
            print("Fake neuron importance:")
            print(fake_score.mean(axis=0))
    
        del model
        torch.cuda.empty_cache()
    
    # plots
    if "clean" in attribution_store and "adv" in attribution_store:
        save_comparison_plots(
            session_name=session_name,
            results=attribution_store,
            fake_positions=fake_store.get(session_name, np.array([])),
        )
    
    print()
    print("=" * 60)
    print("Decoder Results")
    print(f"session = {session_name}")
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
