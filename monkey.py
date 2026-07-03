from pathlib import Path
import h5py

for file in sorted(Path("monkey_dataset").glob("*.mat")):
    with h5py.File(file, "r") as f:
        xds = f["xds"]

        print(
            file.name,
            xds["spike_counts"].shape,
            xds["EMG"].shape,
            xds["force"].shape,
        )
