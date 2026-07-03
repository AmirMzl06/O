import h5py

f = h5py.File("monkey_dataset/Jango_20150730_001.mat", "r")

xds = f["xds"]

print(type(xds))
print(xds.keys())
