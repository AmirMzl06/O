import h5py

f = h5py.File("monkey_dataset/Jango_20150730_001.mat", "r")

print("Top-level keys:")
for k in f.keys():
    print(k)
