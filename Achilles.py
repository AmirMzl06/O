import sys
import os
import shutil
import subprocess
import random

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import zscore

REPO_DIR = "CEBRA"

if not os.path.exists(REPO_DIR):
    subprocess.run([
        "git",
        "clone",
        "https://github.com/AdaptiveMotorControlLab/CEBRA.git"
    ], check=True)

shutil.copy(
    "base.py",
    os.path.join(REPO_DIR, "cebra/solver/base.py")
)

shutil.copy(
    "cebra.py",
    os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py")
)

shutil.copy(
    "cebra.py",
    os.path.join(REPO_DIR, "cebra/cebra.py")
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
import cebra.attribution

from cebra import CEBRA
from decoder import TwoLayerMLP

print("CEBRA:", cebra.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rats = [
    "achilles"
    # "buddy",
    # "cicero",
    # "gatsby"
]

adv_epsilon = 0.5

N_FAKE = 8

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

scores = {
    "clean": {},
    "adv": {}
}

fake_neuron_indices = {}
def setup_seed(seed=42):
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

    print(f"{name}: spikes={spikes.shape}  position={position.shape}")

    return spikes, position


def insert_fake_at_positions(
    x,
    positions,
    rng,
    mu,
    sigma,
):

    n_total = x.shape[1] + len(positions)

    is_fake = np.zeros(n_total, dtype=bool)
    is_fake[positions] = True

    fake_values = rng.normal(
        loc=mu,
        scale=sigma,
        size=(x.shape[0], len(positions)),
    )

    combined = np.zeros(
        (x.shape[0], n_total),
        dtype=np.float32,
    )

    combined[:, is_fake] = fake_values
    combined[:, ~is_fake] = x

    return combined


def add_fake_neurons(train_data, test_data, key):

    rng = np.random.default_rng(0)

    n_real = train_data.shape[1]

    positions = np.sort(
        rng.choice(
            n_real + N_FAKE,
            size=N_FAKE,
            replace=False,
        )
    )

    fake_neuron_indices[key] = positions

    mu = train_data.mean()
    sigma = train_data.std()

    train_data = insert_fake_at_positions(
        train_data,
        positions,
        rng,
        mu,
        sigma,
    )

    test_data = insert_fake_at_positions(
        test_data,
        positions,
        rng,
        mu,
        sigma,
    )

    return train_data, test_data, positions


def get_torch_model(model):

    torch_model = model.solver_.model

    torch_model.split_outputs = False

    torch_model.to(device)
    torch_model.eval()

    return torch_model


def compute_decoder_score(
    model,
    train_data,
    test_data,
    train_label,
    test_label,
):

    train_latent = torch.tensor(
        model.transform(train_data),
        dtype=torch.float32,
    ).to(device)
    test_latent = torch.tensor(
        model.transform(test_data),
        dtype=torch.float32,
    ).to(device)

    train_label = torch.tensor(

        train_label,

        dtype=torch.float32,

    ).to(device)

    test_label = torch.tensor(

        test_label,

        dtype=torch.float32,

    ).to(device)

    decoder = TwoLayerMLP(

        input_dim=48,

        output_dim=2,

    )

    setup_seed(0)

    decoder.fit(

        train_latent,

        train_label,

    )

    with torch.no_grad():
        r2 = decoder.score(
            test_latent,
            test_label,
            device,
        )

    return r2

def compute_attribution(

    model,
    neural,
    batch_size=256,
    num_samples=2000,

):

    neural = torch.from_numpy(neural).float().to(device)

    neural.requires_grad_(True)

    torch_model = get_torch_model(model)

    method = cebra.attribution.init(

        name="jacobian-based-batched",

        model=torch_model,

        input_data=neural,

        output_dimension=torch_model.num_output,

        num_samples=num_samples,

    )

    attribution = method.compute_attribution_map(

        batch_size=batch_size

    )

    jf = np.abs(
        attribution["jf"]
    ).mean(axis=0)

    jfinv = np.abs(
        attribution["jf-inv-svd"]
    ).mean(axis=0)

    del method
    torch.cuda.empty_cache()

    return {

        "jf": jf,

        "jfinv": jfinv,

    }


def binary_maps(jf, jfinv):

    jf_bin = (

        zscore(jf, axis=None) > 0

    ).astype(np.int32)

    jfinv_bin = (

        zscore(jfinv, axis=None) > 0

    ).astype(np.int32)

    return jf_bin, jfinv_bin


def save_heatmaps(

    rat,

    mode,

    result,

    fake_positions,

):

    save_dir = os.path.join(
        RESULT_DIR,
        rat,
        mode,
    )
    os.makedirs(
        save_dir,
        exist_ok=True,
    )
    ########################################################

    fig, axs = plt.subplots(2,2,figsize=(12,10),)

    im = axs[0,0].imshow(
        result["jf"],
        aspect="auto",
    )

    axs[0,0].set_title("JF")

    plt.colorbar(im, ax=axs[0,0])

    im = axs[0,1].imshow(

        result["jfinv"],

        aspect="auto",

    )

    axs[0,1].set_title("JF-inv")

    plt.colorbar(im, ax=axs[0,1])

    jf_bin, jfinv_bin = binary_maps(

        result["jf"],

        result["jfinv"],

    )

    im = axs[1,0].imshow(

        jf_bin,

        aspect="auto",

        cmap="Greys",

        vmin=0,

        vmax=1,

    )

    axs[1,0].set_title("JF binary")

    plt.colorbar(im, ax=axs[1,0])

    im = axs[1,1].imshow(

        jfinv_bin,

        aspect="auto",

        cmap="Greys",

        vmin=0,

        vmax=1,

    )

    axs[1,1].set_title("JF-inv binary")

    plt.colorbar(im, ax=axs[1,1])

    for f in fake_positions:

        axs[0,1].axvline(

            f,

            color="red",

            linestyle="--",

            linewidth=0.8,

        )

        axs[1,1].axvline(

            f,

            color="red",

            linestyle="--",

            linewidth=0.8,

        )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            save_dir,

            "heatmaps.png",

        ),

        dpi=300,

    )

    plt.close()

    ########################################################

    fake = result["jfinv"][:, fake_positions]

    fake_bin = (

        zscore(fake, axis=None) > 0

    ).astype(np.int32)

    plt.figure(

        figsize=(6,4),

    )

    plt.imshow(

        fake_bin,

        aspect="auto",

        cmap="Greys",

        vmin=0,

        vmax=1,

    )

    plt.colorbar()

    plt.title(

        "Fake neurons"

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            save_dir,

            "fake_neurons.png",

        ),

        dpi=300,

    )
    plt.close()
    print(
        f"Figures saved -> {save_dir}"
    )

##############################################################
###################### MAIN PIPELINE ##########################
##############################################################

for training_mode, adv in [

    ("clean", False),

    ("adversarial", True),

]:

    print("=" * 80)
    print(training_mode)
    print("=" * 80)

    for name in rats:

        print(f"\nTraining {name} ...")

        ###########################################################
        # Load dataset
        ###########################################################

        spikes, position = load_local_rat_dataset(name)

        split = int(0.8 * len(spikes))

        train_data = spikes[:split]
        test_data = spikes[split:]

        train_label = position[:split, :2]
        test_label = position[split:, :2]

        ###########################################################
        # Add fake neurons
        ###########################################################

        key = f"{name}_adv" if adv else name

        train_data, test_data, fake_positions = add_fake_neurons(

            train_data,

            test_data,

            key,

        )

        print("Fake neurons:", fake_positions)

        ###########################################################
        # Train
        ###########################################################

        setup_seed(0)

        model = CEBRA(
            batch_size=2048,
            temperature=0.4,
            model_architecture="offset36-model-more-dropout",
            time_offsets=4,
            max_iterations=500,
            output_dimension=48,
            verbose=True,
            training_mode=training_mode,
            adv_alpha=adv_epsilon / 5,
            adv_epsilon=adv_epsilon,
            adv_steps=10,
            attack_norm="l2",
            jacobian_weight=0.01,
            adv_aggregate = True,

        )

        model.fit(

            train_data,

            train_label,

        )

        ###########################################################
        # Decoder
        ###########################################################

        r2 = compute_decoder_score(
            model,
            train_data,
            test_data,
            train_label,
            test_label,
        )

        mode = "adv" if adv else "clean"
        scores[mode][name] = r2
        print(f"R² = {r2:.4f}")
        ###########################################################
        # Attribution
        ###########################################################
        result = compute_attribution(model,train_data,
        )

        ###########################################################
        # Save figures
        ###########################################################

        save_heatmaps(

            rat=name,

            mode=mode,

            result=result,

            fake_positions=fake_positions,

        )

        ###########################################################
        # Fake neurons importance
        ###########################################################

        fake_score = result["jfinv"][:, fake_positions]

        print()

        print("Fake neuron importance:")

        print(fake_score.mean(axis=0))

        ###########################################################
        # Free GPU
        ###########################################################

        del model

        torch.cuda.empty_cache()

##############################################################

print()

print("=" * 60)

print("Decoder Results")

print("=" * 60)

for mode in scores:

    print()

    print(mode)

    for rat in rats:

        print(

            f"{rat:10s}: {scores[mode][rat]:.4f}"

        )

print()

print("Finished.")
#import sys
# import os
# import shutil
# import subprocess
# import joblib
# import numpy as np
# import torch
# import random

# REPO_DIR = "CEBRA"
# if not os.path.exists(REPO_DIR):
#     subprocess.run(["git", "clone",
#                      "https://github.com/AdaptiveMotorControlLab/CEBRA.git"], check=True)

# shutil.copy("base.py", os.path.join(REPO_DIR, "cebra/solver/base.py"))
# shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/integrations/sklearn/cebra.py"))
# shutil.copy("cebra.py", os.path.join(REPO_DIR, "cebra/cebra.py"))

# with open(os.path.join(REPO_DIR, "cebra/solver/base.py"), "a") as f:
#     content = open(os.path.join(REPO_DIR, "cebra/solver/base.py")).read()
#     if "AuxiliaryVariableSolver" not in content:
#         f.write("\nclass AuxiliaryVariableSolver(Solver):\n    pass\n")
#         f.write("\nclass DiscreteAuxiliaryVariableSolver(Solver):\n    pass\n")
# print("Patch applied (or already present).")


# sys.path.insert(0, REPO_DIR)
# if "cebra" in sys.modules:
#     del sys.modules["cebra"]

# import cebra
# from cebra import CEBRA
# from decoder import TwoLayerMLP 

# print("cebra version:", cebra.__version__)

# target_folder = "hippo_models"
# os.makedirs(target_folder, exist_ok=True)
# rats = ["achilles", "buddy", "cicero", "gatsby"]
# adv_epsilon = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def setup_seed(seed: int = 42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def load_local_rat_dataset(name, dataset_dir="dataset"):
#     path = os.path.join(dataset_dir, f"{name}.jl")
#     data = joblib.load(path)
#     spikes = data["spikes"].astype(np.float32)   
#     position = data["position"].astype(np.float32)

#     if position.ndim == 1:
#         position = position[:, None]

#     print(f"[{name}] spikes shape: {spikes.shape}, position shape: {position.shape}")
#     return spikes, position
    

# for training_mode, adv in [("clean", False), ("adversarial", True)]:
#     epochs = 500

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
#         )

#         spikes, position = load_local_rat_dataset(name, dataset_dir="dataset")
#         train_idx = int(0.8 * len(spikes))
#         train_data = spikes[:train_idx]

#         train_continuous_label = position[:train_idx, :2]

#         setup_seed(0)

#         path = name
#         if adv:
#             path += "_adv"
#         path += ".pth"

#         model.fit(train_data, train_continuous_label)
#         model.save(os.path.join(target_folder, path))
#         print(f"saved: {path}")

# print("Training done.")
