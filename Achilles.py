import sys
import os
import shutil
import subprocess
import joblib
import numpy as np
import torch
import random

REPO_DIR = "CEBRA"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone",
                     "https://github.com/AdaptiveMotorControlLab/CEBRA.git"], check=True)

shutil.copy("base.py", os.path.join(REPO_DIR, "cebra/solver/base.py"))
shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py"))
shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/cebra.py"))

with open(os.path.join(REPO_DIR, "cebra/solver/base.py"), "a") as f:
    content = open(os.path.join(REPO_DIR, "cebra/solver/base.py")).read()
    if "AuxiliaryVariableSolver" not in content:
        f.write("\nclass AuxiliaryVariableSolver(Solver):\n    pass\n")
        f.write("\nclass DiscreteAuxiliaryVariableSolver(Solver):\n    pass\n")
print("Patch applied (or already present).")


sys.path.insert(0, REPO_DIR)
if "cebra" in sys.modules:
    del sys.modules["cebra"]

import cebra
from cebra import CEBRA
from decoder import TwoLayerMLP 

print("cebra version:", cebra.__version__)

target_folder = "hippo_models"
os.makedirs(target_folder, exist_ok=True)
rats = ["achilles", "buddy", "cicero", "gatsby"]
adv_epsilon = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_local_rat_dataset(name, dataset_dir="dataset"):
    path = os.path.join(dataset_dir, f"{name}.jl")
    data = joblib.load(path)
    spikes = data["spikes"].astype(np.float32)   
    position = data["position"].astype(np.float32)

    if position.ndim == 1:
        position = position[:, None]

    print(f"[{name}] spikes shape: {spikes.shape}, position shape: {position.shape}")
    return spikes, position
    

for training_mode, adv in [("clean", False), ("adversarial", True)]:
    epochs = 500

    for name in rats:
        model = CEBRA(
            batch_size=2048,
            temperature=0.4,
            model_architecture="offset36-model-more-dropout",
            time_offsets=4,
            max_iterations=epochs,
            output_dimension=48,
            verbose=True,
            training_mode=training_mode,
            adv_alpha=adv_epsilon / 5,
            adv_epsilon=adv_epsilon,
            adv_steps=10,
            attack_norm="l2",
            jacobian_weight=0.01,
        )

        spikes, position = load_local_rat_dataset(name, dataset_dir="dataset")
        train_idx = int(0.8 * len(spikes))
        train_data = spikes[:train_idx]

        train_continuous_label = position[:train_idx, :2]

        setup_seed(0)

        path = name
        if adv:
            path += "_adv"
        path += ".pth"

        model.fit(train_data, train_continuous_label)
        model.save(os.path.join(target_folder, path))
        print(f"saved: {path}")

print("Training done.")

# # !git clone https://github.com/AdaptiveMotorControlLab/CEBRA.git
# # !pip install poyo datasets

# # !unzip -o base.zip

# # !cp base.py CEBRA/cebra/solver/base.py

# # !cp cebra.py CEBRA/cebra/integrations/sklearn/cebra.py
# # !cp cebra.py CEBRA/cebra/cebra.py

# # !rm base.py cebra.py

# # !pip install literate_dataclasses

# with open("CEBRA/cebra/solver/base.py", "a") as f:
#     f.write("\nclass AuxiliaryVariableSolver(Solver):\n    pass\n")
#     f.write("\nclass DiscreteAuxiliaryVariableSolver(Solver):\n    pass\n")
# print("Patch applied successfully!")

# import sys
# if "cebra" in sys.modules:
#     del sys.modules["cebra"]

# CEBRA_DIR = 'CEBRA'
# sys.path.append(str(CEBRA_DIR))
# import cebra
# import itertools
# import torch
# import numpy as np
# import os
# import random
# from cebra import CEBRA
# from utils.decoder import TwoLayerMLP

# target_folder = 'hippo_models'
# os.makedirs(target_folder, exist_ok=True)
# rats = ['achilles']

# adv_epsilon = 0.5
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def setup_seed(seed: int = 42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# fake_neuron_indices = {}

# for training_mode, adv in [('clean', False), ('adversarial', True)]:
#     epochs = 500 if adv else 500

#     for name in rats:
#         model = CEBRA(
#             batch_size=2048,
#             temperature=0.4,
#             model_architecture="offset36-model-more-dropout",
#             time_offsets=4,

#             max_iterations=epochs,
#             output_dimension=48,
#             verbose=True,
#             training_mode=training_mode,
#             adv_alpha=adv_epsilon / 5,
#             adv_epsilon=adv_epsilon,
#             adv_steps=10,
#             attack_norm="l2",
#             jacobian_weight=0.01,
#             adv_aggregate=True,
#         )

#         dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
#         train_idx = int(0.8 * len(dataset.neural))
#         train_data = dataset.neural[:train_idx].numpy()

#         # ---------------- Add fake neurons at random positions ----------------
#         n_fake = 0
#         n_real = train_data.shape[1]
#         n_total = n_real + n_fake

#         rng = np.random.default_rng(0)

#         mu = train_data.mean()
#         sigma = train_data.std()

#         fake_neurons = rng.normal(
#             loc=mu,
#             scale=sigma,
#             size=(train_data.shape[0], n_fake)
#         )

#         fake_positions = rng.choice(n_total, size=n_fake, replace=False)
#         fake_positions = np.sort(fake_positions)

#         combined = np.zeros((train_data.shape[0], n_total), dtype=train_data.dtype)
#         is_fake = np.zeros(n_total, dtype=bool)
#         is_fake[fake_positions] = True

#         combined[:, is_fake] = fake_neurons
#         combined[:, ~is_fake] = train_data

#         train_data = combined

#         key = f"{name}_adv" if adv else name
#         fake_neuron_indices[key] = fake_positions

#         print(f"{name} ({training_mode}): shape={train_data.shape}, fake at {fake_positions}")
#         # ------------------------------------------------------------------------

#         train_continuous_label = dataset.continuous_index.numpy()[:train_idx, :2]
#         setup_seed(0)

#         path = name
#         if adv:
#             path += '_adv'
#         path += '.pth'

#         model.fit(train_data, train_continuous_label)
#         model.save(os.path.join(target_folder, path))
#         print(f"saved: {path}")

# # """#ADditinoal"""

# # def evaluate_robustness_to_neuron_drop(model_path, name, n_delete_list=[0, 5, 10, 15, 20], n_repeats=1):
# #     dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
# #     test_idx = int(0.8 * len(dataset.neural))
# #     test_data = dataset.neural[test_idx:].numpy()
# #     test_label = dataset.continuous_index.numpy()[test_idx:, :2]
# #     train_data_full = dataset.neural[:test_idx].numpy()
# #     train_label = dataset.continuous_index.numpy()[:test_idx, :2]

# #     model = cebra.CEBRA.load(model_path, weights_only=False)
# #     n_total_neurons = test_data.shape[1]

# #     results = {}
# #     for n_delete in n_delete_list:
# #         r2_scores = []
# #         for seed in range(n_repeats):
# #             rng = np.random.default_rng(seed)
# #             if n_delete > 0:
# #                 drop_idx = rng.choice(n_total_neurons, size=n_delete, replace=False)
# #                 keep_mask = np.ones(n_total_neurons, dtype=bool)
# #                 keep_mask[drop_idx] = False
# #             else:
# #                 keep_mask = np.ones(n_total_neurons, dtype=bool)

# #             # نورون‌های حذف‌شده را با صفر جایگزین می‌کنیم (نه این‌که بعدشان را کم کنیم)
# #             # چون مدل از قبل train شده و انتظار دارد همان تعداد نورون ورودی را ببیند
# #             train_perturbed = train_data_full.copy()
# #             test_perturbed = test_data.copy()
# #             train_perturbed[:, ~keep_mask] = 0
# #             test_perturbed[:, ~keep_mask] = 0

# #             train_embedding = model.transform(train_perturbed)
# #             test_embedding = model.transform(test_perturbed)

# #             decoder = TwoLayerMLP(input_dim=train_embedding.shape[1], output_dim=2)
# #             decoder.fit(torch.tensor(train_embedding), torch.tensor(train_label))
# #             with torch.no_grad():
# #                 r2 = decoder.score(torch.tensor(test_embedding), torch.tensor(test_label), device)
# #             r2_scores.append(r2)

# #         results[n_delete] = {"mean": np.mean(r2_scores), "std": np.std(r2_scores)}
# #         print(f"  n_delete={n_delete}: R² = {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# #     return results


# # for name in rats:
# #     print(f"\n=== {name} — Clean ===")
# #     r2_clean_drop = evaluate_robustness_to_neuron_drop(
# #         os.path.join(target_folder, f"{name}.pth"), name
# #     )

# #     print(f"\n=== {name} — Adversarial ===")
# #     r2_adv_drop = evaluate_robustness_to_neuron_drop(
# #         os.path.join(target_folder, f"{name}_adv.pth"), name
# #     )

# # # for rat, score in scores.items():
# # #     print(f'{rat}: R\u00b2 = {score:.4f}')
# # for mode_key, mode_scores in scores.items():
# #     print(f"--- {mode_key} ---")
# #     for rat, score in mode_scores.items():
# #         print(f'{rat}: R² = {score:.4f}')

# """continue

# """
# scores = {}

# for adv in [False, True]:
#     mode_key = 'adv' if adv else 'clean'
#     scores[mode_key] = {}

#     for name in rats:

#         # ---------------- Load model ----------------
#         path = name + ('_adv' if adv else '') + '.pth'
#         model = CEBRA.load(os.path.join(target_folder, path), weights_only=False)

#         # ---------------- Dataset ----------------
#         dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')

#         test_idx = int(0.8 * len(dataset.neural))

#         train_data = dataset.neural[:test_idx].numpy()
#         test_data = dataset.neural[test_idx:].numpy()

#         train_label = dataset.continuous_index.numpy()[:test_idx, :2]
#         test_label = dataset.continuous_index.numpy()[test_idx:, :2]

#         # ---------------- FAKE NEURONS (same positions used in training) ----------------
#         key = f"{name}_adv" if adv else name
#         fake_positions = fake_neuron_indices[key]
      
#         n_fake = len(fake_positions)
#         n_real = train_data.shape[1]
#         n_total = n_real + n_fake

#         rng = np.random.default_rng(0)
#         mu = train_data.mean()
#         sigma = train_data.std()

#         def insert_fake_at_positions(x, positions, n_total, rng, mu, sigma):
#             is_fake = np.zeros(n_total, dtype=bool)
#             is_fake[positions] = True

#             fake_values = rng.normal(loc=mu, scale=sigma, size=(x.shape[0], len(positions)))

#             combined = np.zeros((x.shape[0], n_total), dtype=x.dtype)
#             combined[:, is_fake] = fake_values
#             combined[:, ~is_fake] = x
#             return combined

#         train_data = insert_fake_at_positions(train_data, fake_positions, n_total, rng, mu, sigma)
#         test_data = insert_fake_at_positions(test_data, fake_positions, n_total, rng, mu, sigma)

#         # ---------------- FIX: dtype consistency ----------------
#         train_latent = torch.tensor(
#             model.transform(train_data),
#             dtype=torch.float32
#         ).to(device)

#         test_latent = torch.tensor(
#             model.transform(test_data),
#             dtype=torch.float32
#         ).to(device)

#         train_label_t = torch.tensor(
#             train_label,
#             dtype=torch.float32
#         ).to(device)

#         test_label_t = torch.tensor(
#             test_label,
#             dtype=torch.float32
#         ).to(device)

#         # ---------------- Decoder ----------------
#         setup_seed(0) 
#         decoder = TwoLayerMLP(input_dim=48, output_dim=2)

#         decoder.fit(train_latent, train_label_t)

#         with torch.no_grad():
#             r2 = decoder.score(test_latent, test_label_t, device)

#         scores[mode_key][name] = r2

#         print(f"[{mode_key} | {name}] R2 = {r2:.4f}")

# # for rat, score in scores.items():
# #     print(f'{rat}: R\u00b2 = {score:.4f}')
# for mode_key, mode_scores in scores.items():
#     print(f"--- {mode_key} ---")
#     for rat, score in mode_scores.items():
#         print(f'{rat}: R² = {score:.4f}')

# """XCEBRA"""

# # !pip install captum

# import cebra
# import cebra.attribution
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def get_torch_model(cebra_model):
#     torch_model = cebra_model.solver_.model
#     torch_model.split_outputs = False
#     return torch_model

# def compute_attribution_for_rat(name, target_folder="hippo_models",
#                                  split="train", device=None,
#                                  num_samples=2000, batch_size=256):

#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
#     n = len(dataset.neural)
#     train_idx = int(0.8 * n)

#     if split == "train":
#         neural = dataset.neural[:train_idx].clone()
#     elif split == "test":
#         neural = dataset.neural[train_idx:].clone()
#     else:
#         neural = dataset.neural.clone()

#     # ---------------- ADD FAKE NEURONS (IMPORTANT FIX) ----------------
#     n_fake = 5
#     rng = np.random.default_rng(0)

#     mu = neural.mean()
#     sigma = neural.std()

#     fake_neurons = rng.normal(
#         loc=mu,
#         scale=sigma,
#         size=(neural.shape[0], n_fake)
#     )

#     neural = np.concatenate([neural, fake_neurons], axis=1)
#     # -------------------------------------------------------------------
#     neural = torch.from_numpy(neural).float().to(device)
#     neural.requires_grad_(True)

#     results = {}
#     for tag, suffix in [("clean", ""), ("adv", "_adv")]:
#         path = os.path.join(target_folder, f"{name}{suffix}.pth")
#         if not os.path.exists(path):
#             print(f"[skip] {path} not found")
#             continue

#         model = cebra.CEBRA.load(path, weights_only=False)
#         torch_model = get_torch_model(model).to(device)
#         torch_model.eval()

#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=torch_model,
#             input_data=neural,
#             output_dimension=torch_model.num_output,
#             num_samples=num_samples,
#         )
#         attribution_result = method.compute_attribution_map(batch_size=batch_size)

#         jf = np.abs(attribution_result['jf']).mean(0)
#         jfinv = np.abs(attribution_result['jf-inv-svd']).mean(0)

#         results[tag] = {"jf": jf, "jfinv": jfinv, "raw": attribution_result}
#         print(f"[{name} / {tag}] jf shape: {jf.shape}, jfinv shape: {jfinv.shape}")

#         del torch_model, model, method, attribution_result
#         torch.cuda.empty_cache()

#     return results

# attribution_results = {}

# for name in rats:
#     print(f"\n=== Computing attribution for {name} ===")
#     attribution_results[name] = compute_attribution_for_rat(
#         name, target_folder=target_folder, split="train"
#     )

# for name in rats:
#     res = attribution_results[name]
#     if "clean" not in res or "adv" not in res:
#         continue

#     fig, axs = plt.subplots(2, 2, figsize=(12, 10))

#     im0 = axs[0, 0].matshow(res["clean"]["jf"], aspect="auto")
#     axs[0, 0].set_title(f"{name} — clean — JF")
#     plt.colorbar(im0, ax=axs[0, 0])

#     im1 = axs[0, 1].matshow(res["adv"]["jf"], aspect="auto")
#     axs[0, 1].set_title(f"{name} — adversarial — JF")
#     plt.colorbar(im1, ax=axs[0, 1])

#     im2 = axs[1, 0].matshow(res["clean"]["jfinv"], aspect="auto")
#     axs[1, 0].set_title(f"{name} — clean — JF-inv")
#     plt.colorbar(im2, ax=axs[1, 0])

#     im3 = axs[1, 1].matshow(res["adv"]["jfinv"], aspect="auto")
#     axs[1, 1].set_title(f"{name} — adversarial — JF-inv")
#     plt.colorbar(im3, ax=axs[1, 1])

#     for ax in axs.flat:
#         ax.set_xlabel("Input neurons")
#         ax.set_ylabel("Latent dims")

#     plt.suptitle(f"xCEBRA attribution comparison — {name}", y=1.02, fontsize=14)
#     plt.tight_layout()
#     plt.show()

# from scipy.stats import zscore

# for name in rats:
#     res = attribution_results[name]
#     if "clean" not in res or "adv" not in res:
#         continue

#     jf_clean_bin = (zscore(res["clean"]["jf"], axis=None) > 0).astype(int)
#     jf_adv_bin = (zscore(res["adv"]["jf"], axis=None) > 0).astype(int)
#     jfinv_clean_bin = (zscore(res["clean"]["jfinv"], axis=None) > 0).astype(int)
#     jfinv_adv_bin = (zscore(res["adv"]["jfinv"], axis=None) > 0).astype(int)

#     # ---------------- CHECK FAKE NEURONS (random positions, not last 5) ----------------
#     fake_idx_clean = fake_neuron_indices[name]
#     fake_idx_adv = fake_neuron_indices[f"{name}_adv"]

#     fake_clean = res["clean"]["jfinv"][:, fake_idx_clean]
#     fake_adv = res["adv"]["jfinv"][:, fake_idx_adv]

#     print(f"\n[{name}] Fake neuron positions (clean): {fake_idx_clean}")
#     print(f"[{name}] JF-inv fake neurons (clean):")
#     print(fake_clean.mean(axis=0))

#     print(f"[{name}] Fake neuron positions (adv): {fake_idx_adv}")
#     print(f"[{name}] JF-inv fake neurons (adv):")
#     print(fake_adv.mean(axis=0))
#     # -------------------------------------------------------------------------------------

#     fig, axs = plt.subplots(2, 2, figsize=(12, 10))

#     im0 = axs[0, 0].matshow(jf_clean_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     axs[0, 0].set_title(f"{name} — clean — JF (binary)")
#     plt.colorbar(im0, ax=axs[0, 0])

#     im1 = axs[0, 1].matshow(jf_adv_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     axs[0, 1].set_title(f"{name} — adversarial — JF (binary)")
#     plt.colorbar(im1, ax=axs[0, 1])

#     im2 = axs[1, 0].matshow(jfinv_clean_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     axs[1, 0].set_title(f"{name} — clean — JF-inv (binary)")
#     plt.colorbar(im2, ax=axs[1, 0])

#     im3 = axs[1, 1].matshow(jfinv_adv_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     axs[1, 1].set_title(f"{name} — adversarial — JF-inv (binary)")
#     plt.colorbar(im3, ax=axs[1, 1])

#     for fpos in fake_idx_clean:
#         axs[1, 0].axvline(x=fpos, color="red", linestyle="--", linewidth=0.8)
#     for fpos in fake_idx_adv:
#         axs[1, 1].axvline(x=fpos, color="red", linestyle="--", linewidth=0.8)

#     for ax in axs.flat:
#         ax.set_xlabel("Input neurons")
#         ax.set_ylabel("Latent dims")

#     plt.suptitle(f"xCEBRA attribution comparison (binary, z-score>0) — {name}", y=1.02, fontsize=14)
#     plt.tight_layout()
#     plt.show()

#     fake_clean_bin = (zscore(fake_clean, axis=None) > 0).astype(int)
#     plt.figure(figsize=(6, 4))
#     plt.imshow(fake_clean_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     plt.title(f"{name} — clean JF-inv (fake neurons, binary)")
#     plt.colorbar()
#     plt.xlabel(f"Fake neuron index: {list(fake_idx_clean)}")
#     plt.ylabel("Latent dims")
#     plt.show()

#     fake_adv_bin = (zscore(fake_adv, axis=None) > 0).astype(int)
#     plt.figure(figsize=(6, 4))
#     plt.imshow(fake_adv_bin, aspect="auto", cmap="Greys", vmin=0, vmax=1)
#     plt.title(f"{name} — adv JF-inv (fake neurons, binary)")
#     plt.colorbar()
#     plt.xlabel(f"Fake neuron index: {list(fake_idx_adv)}")
#     plt.ylabel("Latent dims")
#     plt.show()
