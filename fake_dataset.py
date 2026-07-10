import sys
import os
import shutil
import subprocess
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIG
# ============================================================
T = 100_000

D1 = 3
D2 = 3
D_LATENT = D1 + D2

N1 = 25
N2 = 25
D_OBS = N1 + N2

N_MLP_LAYERS = 4
SIGMA_EPS = 0.03

OUTPUT_DIM = D_LATENT
BATCH_SIZE = 5000
MAX_ITER = 50
adv_epsilon = 0.1

REPO_DIR = "CEBRA"
RESULT_DIR = "results_synthetic"
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PATCH CEBRA
# ============================================================
if not os.path.exists(REPO_DIR):
    subprocess.run([
        "git", "clone",
        "https://github.com/AdaptiveMotorControlLab/CEBRA.git",
    ], check=True)

shutil.copy("base.py", os.path.join(REPO_DIR, "cebra/solver/base.py"))
shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py"))
shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/cebra.py"))

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
print("Device:", device)


def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
    torch.manual_seed(seed)
    layers = []
    d_in = in_dim
    hidden = in_dim * 10

    for i in range(n_layers - 1):
        d_h = in_dim * 30 if i < n_layers - 2 else hidden
        lin = nn.Linear(d_in, d_h)
        nn.init.orthogonal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.GELU()]
        d_in = d_h

    lin = nn.Linear(d_in, out_dim)
    nn.init.orthogonal_(lin.weight)
    nn.init.zeros_(lin.bias)
    layers.append(lin)

    mlp = nn.Sequential(*layers)

    for p in mlp.parameters():
        p.requires_grad_(False)

    return mlp.eval()


def brownian_motion_box(T, d, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, d), dtype=np.float32)
    x[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

    for t in range(T - 1):
        step = rng.normal(loc=0.0, scale=sigma, size=d).astype(np.float32)
        x[t + 1] = np.clip(x[t] + step, -1.0, 1.0)

    return x


def make_binary_ground_truth(D1, D2, N1, N2):
    """
    Figure 5 paper

    x1 (0:N1)     <- z1
    x2 (N1:end)   <- z1 + z2
    """

    D_LATENT = D1 + D2
    D_OBS = N1 + N2

    gt = np.zeros((D_LATENT, D_OBS), dtype=bool)

    # z1 -> x1 and x2
    gt[:D1, :] = True

    # z2 -> only x2
    gt[D1:, N1:] = True

    return gt

def generate_synthetic_data(T=T, seed=42):

    z1 = brownian_motion_box(T, D1, sigma=SIGMA_EPS, seed=seed)
    z2 = brownian_motion_box(T, D2, sigma=SIGMA_EPS, seed=seed + 1)

    g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
    g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

    z1_t = torch.tensor(z1, dtype=torch.float32)
    z2_t = torch.tensor(z2, dtype=torch.float32)

    with torch.no_grad():
        x1 = g1(z1_t).cpu().numpy()
        x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

    x = np.concatenate([x1, x2], axis=1).astype(np.float32)
    latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

    gt_bool = make_binary_ground_truth(D1, D2, N1, N2)

    gt_attr = gt_bool.astype(np.float32)

    return x, latent, gt_attr, gt_bool, g1, g2

def get_torch_model(model):
    torch_model = model.solver_.model
    torch_model.split_outputs = False
    torch_model.to(device)
    torch_model.eval()
    return torch_model


def compute_attribution(model, neural_np, gt_attr_bool, batch_size=256, num_samples=2000):
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

    attr = method.compute_attribution_map(batch_size=batch_size)

    jf = np.abs(attr["jf"]).mean(axis=0)
    jfinv = np.abs(attr["jf-inv-svd"]).mean(axis=0)

    auc_jf = method.compute_attribution_score(jf, gt_attr_bool)
    auc_jfinv = method.compute_attribution_score(jfinv, gt_attr_bool)

    del method
    torch.cuda.empty_cache()

    return {
        "jf": jf,
        "jfinv": jfinv,
        "auc_jf": auc_jf,
        "auc_jfinv": auc_jfinv,
    }


import sys
import os
import shutil
import subprocess
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import zscore

y_obs, x_latent, gt_attr, gt_attr_bool, _, _ = generate_synthetic_data()

label = x_latent

print(gt_attr.shape)
print(gt_attr_bool.shape)

results = {}

for training_mode, adv in [("clean", False), ("adversarial", True)]:
    print("\n" + "=" * 60)
    print(f"Training: {training_mode.upper()}")
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
        jacobian_weight=0.5,
        adv_aggregate=False,
    )

    model.fit(y_obs, label)


    print("\nComputing attribution...")
    mode_key = "adv" if adv else "clean"
    results[mode_key] = compute_attribution(model, y_obs, gt_attr_bool)
    print(f"AUC jf    = {results[mode_key]['auc_jf']:.4f}")
    print(f"AUC jfinv = {results[mode_key]['auc_jfinv']:.4f}")

    del model
    torch.cuda.empty_cache()


# from typing_extensions import List
# import sys
# import os
# import shutil
# import subprocess
# import random
# import pickle
# import itertools
# import requests

# import numpy as np
# import torch
# import matplotlib.pyplot as plt

# from scipy.stats import zscore

# REPO_DIR = "CEBRA"

# if not os.path.exists(REPO_DIR):
#     subprocess.run(
#         [
#             "git",
#             "clone",
#             "https://github.com/AdaptiveMotorControlLab/CEBRA.git",
#         ],
#         check=True,
#     )
    
# shutil.copy(
#     "base.py",
#     os.path.join(
#         REPO_DIR,
#         "cebra/solver/base.py",
#     ),
# )

# shutil.copy(
#     "cebra.py",
#     os.path.join(
#         REPO_DIR,
#         "cebra/integrations/sklearn/cebra.py",
#     ),
# )

# shutil.copy(
#     "cebra.py",
#     os.path.join(
#         REPO_DIR,
#         "cebra/cebra.py",
#     ),
# )


# base_path = os.path.join(
#     REPO_DIR,
#     "cebra/solver/base.py",
# )

# sys.path.insert(0, REPO_DIR)

# if "cebra" in sys.modules:
#     del sys.modules["cebra"]

# import cebra
# import cebra.attribution

# from cebra import CEBRA


# with open(base_path, "r") as f:
#     content = f.read()

# if "AuxiliaryVariableSolver" not in content:

#     with open(base_path, "a") as f:

#         f.write(
#             "\nclass AuxiliaryVariableSolver(Solver):\n    pass\n"
#         )

#         f.write(
#             "\nclass DiscreteAuxiliaryVariableSolver(Solver):\n    pass\n"
#         )

# print("Patch applied.")

# print("CEBRA:", cebra.__version__)

# device = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )

# adv_epsilon = 0.1

# RESULT_DIR = "results_fake"

# os.makedirs(
#     RESULT_DIR,
#     exist_ok=True,
# )

# ############################################################
# ################ DOWNLOAD SYNTHETIC DATASET ################
# ############################################################

# FAKE_DATASET_DIR = "fake_dataset"

# os.makedirs(
#     FAKE_DATASET_DIR,
#     exist_ok=True,
# )

# FAKE_DATASET_FILE = os.path.join(
#     FAKE_DATASET_DIR,
#     "cynthi_neurons90.p",
# )

# FAKE_DATASET_URL = (
#     "https://zenodo.org/records/15267195/files/"
#     "cynthi_neurons90_gridbase0.5_gridmodules3_"
#     "grid_head_direction_place_speed_duration2000_"
#     "noise0.25_bs100_seed231209234.p?download=1"
# )


# def download_fake_dataset():

#     if os.path.exists(FAKE_DATASET_FILE):

#         print("Synthetic dataset already exists.")

#         return

#     print("Downloading synthetic dataset...")

#     response = requests.get(
#         FAKE_DATASET_URL,
#         stream=True,
#     )

#     response.raise_for_status()

#     total = int(
#         response.headers.get(
#             "content-length",
#             0,
#         )
#     )

#     downloaded = 0

#     with open(
#         FAKE_DATASET_FILE,
#         "wb",
#     ) as f:

#         for chunk in response.iter_content(
#             chunk_size=1024 * 1024,
#         ):

#             if chunk:

#                 f.write(chunk)

#                 downloaded += len(chunk)

#                 if total > 0:

#                     print(
#                         f"\r{downloaded*100/total:6.2f}% "
#                         f"({downloaded/1024**2:.1f}/"
#                         f"{total/1024**2:.1f} MB)",
#                         end="",
#                     )

#     print("\nDownload complete.")


# download_fake_dataset()
# def load_synthetic_dataset():

#     with open(
#         FAKE_DATASET_FILE,
#         "rb",
#     ) as f:

#         synthetic_dataset = pickle.load(f)

#     spikes = synthetic_dataset["spikes"].astype(
#         np.float32,
#     )

#     position = synthetic_dataset["position"].astype(
#         np.float32,
#     )

#     speed = synthetic_dataset["speed"].astype(
#         np.float32,
#     )

#     if speed.ndim == 1:

#         speed = speed[:, None]

#     labels = np.concatenate(
#         [
#             position,
#             speed,
#         ],
#         axis=1,
#     )

#     print(
#         f"spikes={spikes.shape}  labels={labels.shape}"
#     )

#     return spikes, labels


# ############################################################
# ###################### GROUND TRUTH #########################
# ############################################################

# cells = [
#     ["position"] * 100,
#     ["hd"] * 100,
#     ["position"] * 100,
#     ["grid"] * 60,
# ]

# cells = np.array(
#     list(
#         itertools.chain.from_iterable(
#             cells,
#         )
#     )
# )

# latents = [
#     (
#         ["position", "grid"],
#         3,
#     ),
#     (
#         ["speed"],
#         11,
#     ),
# ]

# latents = [
#     group
#     for group, repeats in latents
#     for _ in range(repeats)
# ]

# ground_truth_attribution = np.zeros(
#     (
#         len(latents),
#         len(cells),
#     ),
#     dtype=bool,
# )

# for i, latent in enumerate(latents):

#     for j, cell_type in enumerate(cells):

#         ground_truth_attribution[i, j] = (
#             cell_type in latent
#         )

# print(
#     "Ground truth:",
#     ground_truth_attribution.shape,
# )


# ############################################################
# ######################## HELPERS ############################
# ############################################################

# def setup_seed(seed=42):

#     torch.manual_seed(seed)

#     np.random.seed(seed)

#     random.seed(seed)

#     if torch.cuda.is_available():

#         torch.cuda.manual_seed_all(seed)


# def get_torch_model(model):
#     torch_model = model.solver_.model
#     torch_model.split_outputs = False
#     torch_model.to(device)
#     torch_model.eval()
#     return torch_model


# def compute_attribution(
#     model,
#     neural,
#     batch_size=256,
#     num_samples=2000,
# ):

#     neural = torch.from_numpy(
#         neural,
#     ).float().to(device)

#     neural.requires_grad_(True)

#     torch_model = get_torch_model(
#         model,
#     )

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=torch_model,
#         input_data=neural,
#         output_dimension=torch_model.num_output,
#         num_samples=num_samples,
#     )

#     attribution = method.compute_attribution_map(
#         batch_size=batch_size,
#     )

#     jf = np.abs(
#         attribution["jf"],
#     ).mean(axis=0)

#     jfinv = np.abs(
#         attribution["jf-inv-svd"],
#     ).mean(axis=0)

#     jfconvabsinv = np.abs(
#         attribution["jf-convabs-inv-svd"],
#     ).mean(axis=0)

#     del method

#     torch.cuda.empty_cache()

#     return {
#         "jf": jf,
#         "jfinv": jfinv,
#         "jfconvabsinv": jfconvabsinv,
#     }

# def save_ground_truth_plot(
#     ground_truth,
#     clean_result,
#     adv_result,
# ):

#     clean_bin = (
#         zscore(
#             clean_result["jfinv"],
#             axis=None,
#         ) > 0
#     ).astype(int)

#     adv_bin = (
#         zscore(
#             adv_result["jfinv"],
#             axis=None,
#         ) > 0
#     ).astype(int)

#     fig, axs = plt.subplots(
#         1,
#         3,
#         figsize=(18,5),
#     )

#     ########################################################

#     im = axs[0].imshow(
#         ground_truth,
#         aspect="auto",
#         cmap="Greys",
#         vmin=0,
#         vmax=1,
#     )

#     axs[0].set_title(
#         "Ground Truth",
#     )

#     axs[0].set_xlabel(
#         "Neurons",
#     )

#     axs[0].set_ylabel(
#         "Latent Dimensions",
#     )

#     plt.colorbar(
#         im,
#         ax=axs[0],
#     )

#     ########################################################

#     im = axs[1].imshow(
#         clean_bin,
#         aspect="auto",
#         cmap="Greys",
#         vmin=0,
#         vmax=1,
#     )

#     axs[1].set_title(
#         "Clean",
#     )

#     axs[1].set_xlabel(
#         "Neurons",
#     )

#     axs[1].set_ylabel(
#         "Latent Dimensions",
#     )

#     plt.colorbar(
#         im,
#         ax=axs[1],
#     )

#     ########################################################

#     im = axs[2].imshow(
#         adv_bin,
#         aspect="auto",
#         cmap="Greys",
#         vmin=0,
#         vmax=1,
#     )

#     axs[2].set_title(
#         "Adversarial",
#     )

#     axs[2].set_xlabel(
#         "Neurons",
#     )

#     axs[2].set_ylabel(
#         "Latent Dimensions",
#     )

#     plt.colorbar(
#         im,
#         ax=axs[2],
#     )

#     ########################################################

#     plt.tight_layout()

#     plt.savefig(
#         os.path.join(
#             RESULT_DIR,
#             "ground_truth_vs_clean_adv.png",
#         ),
#         dpi=300,
#     )

#     plt.close()

#     print(
#         "Figure saved ->",
#         os.path.join(
#             RESULT_DIR,
#             "ground_truth_vs_clean_adv.png",
#         ),
#     )

# ##############################################################
# ###################### MAIN PIPELINE ##########################
# ##############################################################

# spikes, labels = load_synthetic_dataset()

# model_results = {}

# for training_mode in [

#     "clean",

#     "adversarial",

# ]:

#     print("=" * 80)
#     print(training_mode)
#     print("=" * 80)

#     setup_seed(0)

#     model = CEBRA(
#         batch_size=512,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=1000,
#         output_dimension=len(latents),
#         verbose=True,
#         training_mode=training_mode,
#         adv_alpha=adv_epsilon / 5,
#         adv_epsilon=adv_epsilon,
#         adv_steps=10,
#         attack_norm="l2",
#         jacobian_weight=0,
#         adv_aggregate=False,
#     )

#     model.fit(
#         spikes,
#         labels,
#     )

#     ########################################################

#     result = compute_attribution(
#         model,
#         spikes,
#     )

#     ########################################################

#     torch_model = get_torch_model(model,)

#     method = cebra.attribution.init(

#         name="jacobian-based-batched",

#         model=torch_model,

#         input_data=torch.from_numpy(
#             spikes,
#         ).float().to(device),

#         output_dimension=torch_model.num_output,

#         num_samples=2000,

#     )

#     ########################################################

#     auc_jf = method.compute_attribution_score(

#         result["jf"],

#         ground_truth_attribution,

#     )

#     auc_jfinv = method.compute_attribution_score(

#         result["jfinv"],

#         ground_truth_attribution,

#     )

#     auc_jfconvabsinv = method.compute_attribution_score(

#         result["jfconvabsinv"],

#         ground_truth_attribution,

#     )

#     ########################################################

#     model_results[training_mode] = {

#         "result": result,

#         "auc_jf": auc_jf,

#         "auc_jfinv": auc_jfinv,

#         "auc_jfconvabsinv": auc_jfconvabsinv,

#     }

#     ########################################################

#     del method

#     del model

#     torch.cuda.empty_cache()

# ##############################################################
# ########################## RESULTS ############################
# ##############################################################

# print()

# print("=== Clean ===")

# print(
#     f"AUC jf            = "
#     f"{model_results['clean']['auc_jf']:.4f}"
# )

# print(
#     f"AUC jf-inv         = "
#     f"{model_results['clean']['auc_jfinv']:.4f}"
# )

# print(
#     f"AUC jf-convabs-inv = "
#     f"{model_results['clean']['auc_jfconvabsinv']:.4f}"
# )

# print()

# print("=== Adversarial ===")

# print(
#     f"AUC jf            = "
#     f"{model_results['adversarial']['auc_jf']:.4f}"
# )

# print(
#     f"AUC jf-inv         = "
#     f"{model_results['adversarial']['auc_jfinv']:.4f}"
# )

# print(
#     f"AUC jf-convabs-inv = "
#     f"{model_results['adversarial']['auc_jfconvabsinv']:.4f}"
# )

# ##############################################################
# ######################## SAVE FIGURE ##########################
# ##############################################################

# save_ground_truth_plot(

#     ground_truth_attribution,

#     model_results["clean"]["result"],

#     model_results["adversarial"]["result"],

# )

# print()

# print("=" * 80)

# print("Finished.")

# print("=" * 80)













# #import requests
# # import os
# # import sys
# # import shutil
# # import subprocess
# # import random

# # import joblib
# # import numpy as np
# # import torch
# # import matplotlib.pyplot as plt

# # from scipy.stats import zscore


# # REPO_DIR = "CEBRA"

# # FAKE_DATASET_DIR = "fake_dataset"
# # os.makedirs(FAKE_DATASET_DIR, exist_ok=True)

# # FAKE_DATASET_FILE = os.path.join(
# #     FAKE_DATASET_DIR,
# #     "cynthi_neurons90.p",
# # )

# # FAKE_DATASET_URL = (
# #     "https://zenodo.org/records/15267195/files/"
# #     "cynthi_neurons90_gridbase0.5_gridmodules3_"
# #     "grid_head_direction_place_speed_duration2000_"
# #     "noise0.25_bs100_seed231209234.p?download=1"
# # )

# # if not os.path.exists(REPO_DIR):
# #     subprocess.run([
# #         "git",
# #         "clone",
# #         "https://github.com/AdaptiveMotorControlLab/CEBRA.git"
# #     ], check=True)

# # shutil.copy(
# #     "base.py",
# #     os.path.join(REPO_DIR, "cebra/solver/base.py")
# # )

# # shutil.copy(
# #     "cebra.py",
# #     os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py")
# # )

# # shutil.copy(
# #     "cebra.py",
# #     os.path.join(REPO_DIR, "cebra/cebra.py")
# # )

# # import cebra
# # import cebra.attribution

# # from cebra import CEBRA

# # def download_fake_dataset():

# #     if os.path.exists(FAKE_DATASET_FILE):
# #         print("Synthetic dataset already exists.")
# #         return

# #     print("Downloading synthetic dataset...")

# #     response = requests.get(
# #         FAKE_DATASET_URL,
# #         stream=True,
# #     )

# #     response.raise_for_status()

# #     total_size = int(
# #         response.headers.get(
# #             "content-length",
# #             0,
# #         )
# #     )

# #     downloaded = 0

# #     with open(
# #         FAKE_DATASET_FILE,
# #         "wb",
# #     ) as f:

# #         for chunk in response.iter_content(
# #             chunk_size=1024 * 1024,
# #         ):

# #             if chunk:
# #                 f.write(chunk)
# #                 downloaded += len(chunk)
# #                 if total_size > 0:
# #                     percent = downloaded * 100 / total_size

# #                     print(
# #                         f"\r{percent:6.2f}% "
# #                         f"({downloaded/1024**2:.1f}/"
# #                         f"{total_size/1024**2:.1f} MB)",
# #                         end="",
# #                     )

# #     print("\nDownload complete.")

# # download_fake_dataset()

# # import pickle

# # def load_synthetic_dataset():

# #     with open(FAKE_DATASET_FILE, "rb") as f:
# #         synthetic_dataset = pickle.load(f)

# #     spikes = synthetic_dataset["spikes"].astype(np.float32)

# #     position = synthetic_dataset["position"].astype(np.float32)

# #     speed = synthetic_dataset["speed"].astype(np.float32)

# #     if speed.ndim == 1:
# #         speed = speed[:, None]

# #     labels = np.concatenate(
# #         [
# #             position,
# #             speed,
# #         ],
# #         axis=1,
# #     )

# #     print(
# #         f"spikes={spikes.shape}  labels={labels.shape}"
# #     )

# #     return spikes, labels

# # import itertools

# # cells = [
# #     ["position"] * 100,
# #     ["hd"] * 100,
# #     ["position"] * 100,
# #     ["grid"] * 60,
# # ]

# # cells = np.array(
# #     list(
# #         itertools.chain.from_iterable(cells)
# #     )
# # )

# # latents = [
# #     (["position", "grid"], 3),
# #     (["speed"], 11),
# # ]

# # latents = [
# #     group
# #     for group, repeats in latents
# #     for _ in range(repeats)
# # ]

# # ground_truth_attribution = np.zeros(
# #     (
# #         len(latents),
# #         len(cells),
# #     ),
# #     dtype=bool,
# # )

# # for i, latent in enumerate(latents):

# #     for j, cell_type in enumerate(cells):

# #         ground_truth_attribution[i, j] = (
# #             cell_type in latent
# #         )

# # print(
# #     "Ground truth:",
# #     ground_truth_attribution.shape,
# # )

# # def compute_attribution(
# #     model,
# #     neural,
# #     batch_size=256,
# #     num_samples=2000,
# # ):

# #     neural = torch.from_numpy(neural).float().to(device)

# #     neural.requires_grad_(True)

# #     torch_model = get_torch_model(model)

# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=torch_model,
# #         input_data=neural,
# #         output_dimension=torch_model.num_output,
# #         num_samples=num_samples,
# #     )

# #     attribution = method.compute_attribution_map(
# #         batch_size=batch_size,
# #     )

# #     jf = np.abs(
# #         attribution["jf"]
# #     ).mean(axis=0)

# #     jfinv = np.abs(
# #         attribution["jf-inv-svd"]
# #     ).mean(axis=0)

# #     jfconvabsinv = np.abs(
# #         attribution["jf-convabs-inv-svd"]
# #     ).mean(axis=0)

# #     del method

# #     torch.cuda.empty_cache()

# #     return {
# #         "jf": jf,
# #         "jfinv": jfinv,
# #         "jfconvabsinv": jfconvabsinv,
# #     }

# # def setup_seed(seed):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False

# # def save_ground_truth_plot(
# #     ground_truth,
# #     clean_result,
# #     adv_result,
# # ):

# #     clean_bin = (
# #         zscore(
# #             clean_result["jfinv"],
# #             axis=None,
# #         ) > 0
# #     ).astype(int)

# #     adv_bin = (
# #         zscore(
# #             adv_result["jfinv"],
# #             axis=None,
# #         ) > 0
# #     ).astype(int)

# #     fig, axs = plt.subplots(
# #         1,
# #         3,
# #         figsize=(18,5),
# #     )

# #     im = axs[0].imshow(
# #         ground_truth,
# #         aspect="auto",
# #         cmap="Greys",
# #         vmin=0,
# #         vmax=1,
# #     )

# #     axs[0].set_title(
# #         "Ground Truth",
# #     )

# #     plt.colorbar(
# #         im,
# #         ax=axs[0],
# #     )

# #     im = axs[1].imshow(
# #         clean_bin,
# #         aspect="auto",
# #         cmap="Greys",
# #         vmin=0,
# #         vmax=1,
# #     )

# #     axs[1].set_title(
# #         "Clean",
# #     )

# #     plt.colorbar(
# #         im,
# #         ax=axs[1],
# #     )

# #     im = axs[2].imshow(
# #         adv_bin,
# #         aspect="auto",
# #         cmap="Greys",
# #         vmin=0,
# #         vmax=1,
# #     )

# #     axs[2].set_title(
# #         "Adversarial",
# #     )

# #     plt.colorbar(
# #         im,
# #         ax=axs[2],
# #     )

# #     plt.tight_layout()

# #     plt.savefig(
# #         os.path.join(
# #             RESULT_DIR,
# #             "ground_truth_vs_clean_adv.png",
# #         ),
# #         dpi=300,
# #     )

# #     plt.close()


# # RESULT_DIR = "results_fake"
# # os.makedirs(RESULT_DIR, exist_ok=True)

# # ##############################################################
# # ###################### MAIN PIPELINE ##########################
# # ##############################################################
# # spikes, labels = load_synthetic_dataset()

# # model_results = {}

# # for training_mode in [
# #     "clean",
# #     "adversarial",
# # ]:

# #     print("=" * 80)
# #     print(training_mode)
# #     print("=" * 80)

# #     setup_seed(0)

# #     model = CEBRA(
# #         batch_size=512,
# #         temperature=0.4,
# #         model_architecture="offset36-model-more-dropout",
# #         time_offsets=4,
# #         max_iterations=1000,
# #         output_dimension=len(latents),
# #         verbose=True,
# #         training_mode=training_mode,
# #         adv_alpha=adv_epsilon / 5,
# #         adv_epsilon=adv_epsilon,
# #         adv_steps=10,
# #         attack_norm="l2",
# #         jacobian_weight=0,#0.01
# #         adv_aggregate=False,
# #     )

# #     model.fit(
# #         spikes,
# #         labels,
# #     )

# #     result = compute_attribution(
# #         model,
# #         spikes,
# #     )

# #     torch_model = get_torch_model(model)

# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=torch_model,
# #         input_data=torch.from_numpy(spikes).float().to(device),
# #         output_dimension=torch_model.num_output,
# #         num_samples=2000,
# #     )

# #     auc_jf = method.compute_attribution_score(
# #         result["jf"],
# #         ground_truth_attribution,
# #     )

# #     auc_jfinv = method.compute_attribution_score(
# #         result["jfinv"],
# #         ground_truth_attribution,
# #     )

# #     auc_jfconvabsinv = method.compute_attribution_score(
# #         np.abs(
# #             method.compute_attribution_map(
# #                 batch_size=256,
# #             )["jf-convabs-inv-svd"]
# #         ).mean(0),
# #         ground_truth_attribution,
# #     )

# #     model_results[training_mode] = {
# #         "result": result,
# #         "auc_jf": auc_jf,
# #         "auc_jfinv": auc_jfinv,
# #         "auc_jfconvabsinv": auc_jfconvabsinv,
# #     }

# #     del method
# #     del model
# #     torch.cuda.empty_cache()

# # print()

# # print("=== Clean ===")
# # print(f"AUC jf            = {model_results['clean']['auc_jf']:.4f}")
# # print(f"AUC jf-inv         = {model_results['clean']['auc_jfinv']:.4f}")
# # print(f"AUC jf-convabs-inv = {model_results['clean']['auc_jfconvabsinv']:.4f}")

# # print()

# # print("=== Adversarial ===")
# # print(f"AUC jf            = {model_results['adversarial']['auc_jf']:.4f}")
# # print(f"AUC jf-inv         = {model_results['adversarial']['auc_jfinv']:.4f}")
# # print(f"AUC jf-convabs-inv = {model_results['adversarial']['auc_jfconvabsinv']:.4f}")

