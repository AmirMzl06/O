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
DATASET_DIR = "monkey_dataset"

sessions = [
    Path(DATASET_DIR) / "Jango_20150805_001.mat",
    Path(DATASET_DIR) / "Jango_20150807_001.mat",
]

RESULT_DIR     = "results_monkey"
REPO_DIR       = "CEBRA"
N_FAKE         = 0
adv_epsilon    = 0.5
MAX_ITER       = 1500
OUTPUT_DIM     = 48
BATCH_SIZE     = 512
N_ELEC         = 96
BIN_SIZE_MS    = 50
SMOOTH_SD_MS   = 100

N_SILENCE      = 5
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


def read_h5_scalar(obj):
    return float(obj[:].flat[0])


def read_h5_1d(obj):
    return obj[:].flatten()


def load_monkey_session(mat_path,
                        bin_size_ms=BIN_SIZE_MS,
                        smooth_sd_ms=SMOOTH_SD_MS,
                        n_elec=N_ELEC):
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
            cp_raw = cp_raw.T

        trial_start   = read_h5_1d(xds["trial_start_time"])
        trial_end     = read_h5_1d(xds["trial_end_time"])
        trial_results = [chr(int(v)) for v in xds["trial_result"][:].ravel()]

        ratio  = int(round(bin_size_ms / raw_bin_ms))
        T_trim = (T_raw // ratio) * ratio

        sc_binned = sc_raw[:T_trim].reshape(T_trim // ratio, ratio, N_raw).sum(1)
        cp_binned = cp_raw[:T_trim].reshape(T_trim // ratio, ratio, 2).mean(1)
        T_binned  = sc_binned.shape[0]
        print(f"  after {bin_size_ms}ms binning: ({T_binned}, {N_raw})")

        smooth_sd_bins = smooth_sd_ms / bin_size_ms
        sc_smooth = gaussian_filter1d(
            sc_binned.astype(np.float32), sigma=smooth_sd_bins, axis=0)

        if N_raw < n_elec:
            pad = np.zeros((T_binned, n_elec - N_raw), dtype=np.float32)
            sc_smooth = np.concatenate([sc_smooth, pad], axis=1)
            print(f"  zero-padded: {N_raw} → {n_elec} channels")

        spikes_list, curs_list, n_ok = [], [], 0
        for t_start, t_end, t_res in zip(trial_start, trial_end, trial_results):
            if t_res != "R" or np.isnan(t_start) or np.isnan(t_end):
                continue
            b0 = max(0, int(round(t_start * 1000 / bin_size_ms)))
            b1 = min(T_binned, int(round(t_end * 1000 / bin_size_ms)))
            if b1 <= b0:
                continue
            spikes_list.append(sc_smooth[b0:b1].astype(np.float32))
            curs_list.append(cp_binned[b0:b1].astype(np.float32))
            n_ok += 1

        print(f"  extracted {n_ok} successful trials (R)")
        if n_ok == 0:
            raise RuntimeError("no successful trials found!")
        return spikes_list, curs_list


def concat_trials(spikes_list, curs_list):
    spikes = np.concatenate(spikes_list, axis=0)
    curs   = np.concatenate(curs_list,   axis=0)
    print(f"Concatenated: spikes={spikes.shape}  cursor={curs.shape}")
    return spikes, curs


def get_torch_model(model):
    torch_model = model.solver_.model
    torch_model.split_outputs = False
    torch_model.to(device)
    torch_model.eval()
    return torch_model


def compute_decoder_score(model, train_data, test_data,
                           train_label, test_label):
    train_latent  = torch.tensor(model.transform(train_data),
                                  dtype=torch.float32).to(device)
    test_latent   = torch.tensor(model.transform(test_data),
                                  dtype=torch.float32).to(device)
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


# ── Silencing functions ──────────────────────────────────────
def zero_out_random_neurons(test_data, n_silence=N_SILENCE,
                             fake_positions=None):
    n_neurons   = test_data.shape[1]
    all_indices = list(range(n_neurons))
    if fake_positions is not None and len(fake_positions) > 0:
        fake_set    = set(fake_positions.tolist())
        all_indices = [i for i in all_indices if i not in fake_set]

    rng     = np.random.default_rng(42)
    dropped = rng.choice(all_indices, size=n_silence, replace=False)

    modified = test_data.copy()
    modified[:, dropped] = 0.0
    print(f"  [Random silencing] {n_silence} neurons: {sorted(dropped)}")
    return modified, dropped


def zero_out_top_neurons(test_data, attribution_result,
                          n_silence=N_SILENCE, fake_positions=None):
                            
    importance = attribution_result["jfinv"].mean(axis=0)  # (n_neurons,)

    candidate_mask = np.ones(len(importance), dtype=bool)
    if fake_positions is not None and len(fake_positions) > 0:
        candidate_mask[fake_positions] = False

    importance_masked = importance.copy()
    importance_masked[~candidate_mask] = -np.inf 

    top_idx = np.argsort(importance_masked)[-n_silence:][::-1]

    modified = test_data.copy()
    modified[:, top_idx] = 0.0
    print(f"  [Targeted silencing] top-{n_silence} neurons by attribution: "
          f"{sorted(top_idx.tolist())}")
    print(f"  importance values: "
          f"{[f'{importance[i]:.4f}' for i in top_idx]}")
    return modified, top_idx


def plot_silencing_comparison(session_name, silencing_results):
    save_dir = os.path.join(RESULT_DIR, session_name)
    os.makedirs(save_dir, exist_ok=True)

    modes    = ["clean", "adv"]
    colors   = {"clean": "steelblue", "adv": "tomato"}
    x        = np.arange(3)
    labels   = ["Normal", f"Random\n({N_SILENCE} neurons)",
                f"Targeted\n({N_SILENCE} top neurons)"]
    width    = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mode in enumerate(modes):
        vals = [
            silencing_results[mode]["normal"],
            silencing_results[mode]["random"],
            silencing_results[mode]["targeted"],
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=mode,
                      color=colors[mode], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R²")
    ax.set_title(f"{session_name}\nR² under neuron silencing (test only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "targeted_silencing.png"), dpi=300)
    plt.close()
    print(f"[{session_name}] targeted_silencing.png saved.")


# ============================================================
# MAIN PIPELINE
# ============================================================
for SESSION_FILE in sessions:
    session_name = Path(SESSION_FILE).stem

    print(f"\n{'='*80}")
    print(f"Session: {session_name}")
    print(f"{'='*80}")

    spikes_list, curs_list = load_monkey_session(SESSION_FILE)
    spikes, cursor         = concat_trials(spikes_list, curs_list)

    split       = int(0.8 * len(spikes))
    train_data  = spikes[:split]
    test_data   = spikes[split:]
    train_label = cursor[:split, :2]
    test_label  = cursor[split:, :2]

    print(f"Train: {train_data.shape}  Test: {test_data.shape}")

    silencing_results = {"clean": {}, "adv": {}}
    fake_store        = {}

    for training_mode, adv in [("clean", False), ("adversarial", True)]:

        print(f"\n{'='*60}")
        print(f"Training mode: {training_mode}")
        print(f"{'='*60}")

        key    = f"{session_name}_adv" if adv else session_name
        mode   = "adv" if adv else "clean"

        fake_positions = np.array([], dtype=int)
        fake_store[key] = fake_positions
        tr_data, te_data = train_data.copy(), test_data.copy()

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

        r2_normal = compute_decoder_score(
            model, tr_data, te_data, train_label, test_label)
        print(f"Normal R² = {r2_normal:.4f}")

        print("Computing attribution...")
        attr_result = compute_attribution(model, tr_data)

        # ── ۳. Random silencing (only test) ───────────────────
        te_random, random_idx = zero_out_random_neurons(
            te_data, n_silence=N_SILENCE, fake_positions=fake_positions)
        r2_random = compute_decoder_score(
            model, tr_data, te_random, train_label, test_label)
        print(f"Random silencing R² = {r2_random:.4f}")

        te_targeted, top_idx = zero_out_top_neurons(
            te_data, attr_result,
            n_silence=N_SILENCE, fake_positions=fake_positions)
        r2_targeted = compute_decoder_score(
            model, tr_data, te_targeted, train_label, test_label)
        print(f"Targeted silencing R² = {r2_targeted:.4f}")

        silencing_results[mode]["normal"]   = r2_normal
        silencing_results[mode]["random"]   = r2_random
        silencing_results[mode]["targeted"] = r2_targeted

        del model
        torch.cuda.empty_cache()

    plot_silencing_comparison(session_name, silencing_results)

    print()
    print("=" * 60)
    print(f"Summary — {session_name}")
    print("=" * 60)
    print(f"{'':30s} {'clean':>10s} {'adv':>10s}")
    print(f"{'Normal R²':30s} "
          f"{silencing_results['clean']['normal']:>10.4f} "
          f"{silencing_results['adv']['normal']:>10.4f}")
    print(f"{'Random silencing R²':30s} "
          f"{silencing_results['clean']['random']:>10.4f} "
          f"{silencing_results['adv']['random']:>10.4f}")
    print(f"{'Targeted silencing R²':30s} "
          f"{silencing_results['clean']['targeted']:>10.4f} "
          f"{silencing_results['adv']['targeted']:>10.4f}")

    drop_clean_rand = silencing_results['clean']['normal'] - silencing_results['clean']['random']
    drop_clean_targ = silencing_results['clean']['normal'] - silencing_results['clean']['targeted']
    drop_adv_rand   = silencing_results['adv']['normal']   - silencing_results['adv']['random']
    drop_adv_targ   = silencing_results['adv']['normal']   - silencing_results['adv']['targeted']

    print()
    print(f"{'R² drop (random)':30s} "
          f"{drop_clean_rand:>10.4f} {drop_adv_rand:>10.4f}")
    print(f"{'R² drop (targeted)':30s} "
          f"{drop_clean_targ:>10.4f} {drop_adv_targ:>10.4f}")

print()
print("Finished.")
