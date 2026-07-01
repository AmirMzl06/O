from torch_brain.datasets import PerichMillerPopulation2018

PerichMillerPopulation2018.download(
    root="perich_data"
)

from torch_brain.datasets import PerichMillerPopulation2018

dataset = PerichMillerPopulation2018(
    root="data"
)

dataset = PerichMillerPopulation2018(
    root="data",
    recording_ids=[
        "c_20131003_center_out_reaching"
    ]
)

print(dataset.recording_ids)
