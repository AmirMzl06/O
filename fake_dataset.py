import requests
import os

FAKE_DATASET_DIR = "fake_dataset"
os.makedirs(FAKE_DATASET_DIR, exist_ok=True)

FAKE_DATASET_FILE = os.path.join(
    FAKE_DATASET_DIR,
    "cynthi_neurons90.p",
)

FAKE_DATASET_URL = (
    "https://zenodo.org/records/15267195/files/"
    "cynthi_neurons90_gridbase0.5_gridmodules3_"
    "grid_head_direction_place_speed_duration2000_"
    "noise0.25_bs100_seed231209234.p?download=1"
)

def download_fake_dataset():

    if os.path.exists(FAKE_DATASET_FILE):
        print("Synthetic dataset already exists.")
        return

    print("Downloading synthetic dataset...")

    response = requests.get(
        FAKE_DATASET_URL,
        stream=True,
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get(
            "content-length",
            0,
        )
    )

    downloaded = 0

    with open(
        FAKE_DATASET_FILE,
        "wb",
    ) as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024,
        ):

            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded * 100 / total_size

                    print(
                        f"\r{percent:6.2f}% "
                        f"({downloaded/1024**2:.1f}/"
                        f"{total_size/1024**2:.1f} MB)",
                        end="",
                    )

    print("\nDownload complete.")

download_fake_dataset()

import pickle

def load_synthetic_dataset():

    with open(FAKE_DATASET_FILE, "rb") as f:
        synthetic_dataset = pickle.load(f)

    spikes = synthetic_dataset["spikes"].astype(np.float32)

    position = synthetic_dataset["position"].astype(np.float32)

    speed = synthetic_dataset["speed"].astype(np.float32)

    if speed.ndim == 1:
        speed = speed[:, None]

    labels = np.concatenate(
        [
            position,
            speed,
        ],
        axis=1,
    )

    print(
        f"spikes={spikes.shape}  labels={labels.shape}"
    )

    return spikes, labels

import itertools

cells = [
    ["position"] * 100,
    ["hd"] * 100,
    ["position"] * 100,
    ["grid"] * 60,
]

cells = np.array(
    list(
        itertools.chain.from_iterable(cells)
    )
)

latents = [
    (["position", "grid"], 3),
    (["speed"], 11),
]

latents = [
    group
    for group, repeats in latents
    for _ in range(repeats)
]

ground_truth_attribution = np.zeros(
    (
        len(latents),
        len(cells),
    ),
    dtype=bool,
)

for i, latent in enumerate(latents):

    for j, cell_type in enumerate(cells):

        ground_truth_attribution[i, j] = (
            cell_type in latent
        )

print(
    "Ground truth:",
    ground_truth_attribution.shape,
)

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
        batch_size=batch_size,
    )

    jf = np.abs(
        attribution["jf"]
    ).mean(axis=0)

    jfinv = np.abs(
        attribution["jf-inv-svd"]
    ).mean(axis=0)

    jfconvabsinv = np.abs(
        attribution["jf-convabs-inv-svd"]
    ).mean(axis=0)

    del method

    torch.cuda.empty_cache()

    return {
        "jf": jf,
        "jfinv": jfinv,
        "jfconvabsinv": jfconvabsinv,
    }

def save_ground_truth_plot(
    ground_truth,
    clean_result,
    adv_result,
):

    clean_bin = (
        zscore(
            clean_result["jfinv"],
            axis=None,
        ) > 0
    ).astype(int)

    adv_bin = (
        zscore(
            adv_result["jfinv"],
            axis=None,
        ) > 0
    ).astype(int)

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(18,5),
    )

    im = axs[0].imshow(
        ground_truth,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    axs[0].set_title(
        "Ground Truth",
    )

    plt.colorbar(
        im,
        ax=axs[0],
    )

    im = axs[1].imshow(
        clean_bin,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    axs[1].set_title(
        "Clean",
    )

    plt.colorbar(
        im,
        ax=axs[1],
    )

    im = axs[2].imshow(
        adv_bin,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    axs[2].set_title(
        "Adversarial",
    )

    plt.colorbar(
        im,
        ax=axs[2],
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "ground_truth_vs_clean_adv.png",
        ),
        dpi=300,
    )

    plt.close()


RESULT_DIR = "results_fake"
os.makedirs(RESULT_DIR, exist_ok=True)

##############################################################
###################### MAIN PIPELINE ##########################
##############################################################
spikes, labels = load_synthetic_dataset()

model_results = {}

for training_mode in [
    "clean",
    "adversarial",
]:

    print("=" * 80)
    print(training_mode)
    print("=" * 80)

    setup_seed(0)

    model = CEBRA(
        batch_size=512,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=1000,
        output_dimension=len(latents),
        verbose=True,
        training_mode=training_mode,
        adv_alpha=adv_epsilon / 5,
        adv_epsilon=adv_epsilon,
        adv_steps=10,
        attack_norm="l2",
        jacobian_weight=0.01,
        adv_aggregate=True,
    )

    model.fit(
        spikes,
        labels,
    )

    result = compute_attribution(
        model,
        spikes,
    )

    torch_model = get_torch_model(model)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=torch_model,
        input_data=torch.from_numpy(spikes).float().to(device),
        output_dimension=torch_model.num_output,
        num_samples=2000,
    )

    auc_jf = method.compute_attribution_score(
        result["jf"],
        ground_truth_attribution,
    )

    auc_jfinv = method.compute_attribution_score(
        result["jfinv"],
        ground_truth_attribution,
    )

    auc_jfconvabsinv = method.compute_attribution_score(
        np.abs(
            method.compute_attribution_map(
                batch_size=256,
            )["jf-convabs-inv-svd"]
        ).mean(0),
        ground_truth_attribution,
    )

    model_results[training_mode] = {
        "result": result,
        "auc_jf": auc_jf,
        "auc_jfinv": auc_jfinv,
        "auc_jfconvabsinv": auc_jfconvabsinv,
    }

    del method
    del model
    torch.cuda.empty_cache()

print()

print("=== Clean ===")
print(f"AUC jf            = {model_results['clean']['auc_jf']:.4f}")
print(f"AUC jf-inv         = {model_results['clean']['auc_jfinv']:.4f}")
print(f"AUC jf-convabs-inv = {model_results['clean']['auc_jfconvabsinv']:.4f}")

print()

print("=== Adversarial ===")
print(f"AUC jf            = {model_results['adversarial']['auc_jf']:.4f}")
print(f"AUC jf-inv         = {model_results['adversarial']['auc_jfinv']:.4f}")
print(f"AUC jf-convabs-inv = {model_results['adversarial']['auc_jfconvabsinv']:.4f}")

# def compute_attribution(
#     model,
#     neural,
#     batch_size=256,
#     num_samples=2000,
# ):

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

#     attribution = method.compute_attribution_map(
#         batch_size=batch_size,
#     )

#     jf = np.abs(
#         attribution["jf"]
#     ).mean(axis=0)

#     jfinv = np.abs(
#         attribution["jf-inv-svd"]
#     ).mean(axis=0)

#     jfconvabsinv = np.abs(
#         attribution["jf-convabs-inv-svd"]
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

#     plt.colorbar(
#         im,
#         ax=axs[0],
#     )

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

#     plt.colorbar(
#         im,
#         ax=axs[1],
#     )

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

#     plt.colorbar(
#         im,
#         ax=axs[2],
#     )

#     plt.tight_layout()

#     plt.savefig(
#         os.path.join(
#             RESULT_DIR,
#             "ground_truth_vs_clean_adv.png",
#         ),
#         dpi=300,
#     )

#     plt.close()

print(f"AUC jf-convabs-inv = {model_results['adversarial']['auc_jfconvabsinv']:.4f}")

save_ground_truth_plot(
    ground_truth_attribution,
    model_results["clean"]["result"],
    model_results["adversarial"]["result"],
)

print("\nFigure saved -> results_fake/ground_truth_vs_clean_adv.png")

