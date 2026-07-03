import h5py

f = h5py.File("monkey_dataset/Jango_20150730_001.mat", "r")

xds = f["xds"]

for key in [
    "spike_counts",
    "EMG",
    "force",
    "curs_p",
    "trial_target_dir",
    "unit_names",
]:
    obj = xds[key]
    print(f"\n{key}")
    print(type(obj))
    print(obj.shape)
    print(obj.dtype)
