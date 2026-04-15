import xarray as xr
from virtualizarr.manifests import ManifestArray
from zarr.core.metadata.v3 import ArrayV3Metadata

file_path = "/home/ibb000/zarr-mycodec/eccc-data/sample_safe.fst"

with open(file_path, "rb") as f:
    raw_bytes = f.read()
    offset = raw_bytes.find(b"XDF0")
    if offset == -1:
        offset = raw_bytes.find(b"RSF0")
        fst_format = "RSF"
    else:
        fst_format = "XDF"
    
    length = len(raw_bytes) - offset
    
print(offset, length, fst_format)
chunk_manifest = {
    "0.0.0": {
        "path": "/home/ibb000/zarr-mycodec/eccc-data/sample_safe.fst",
        "offset": offset,  #4096,
        "length": length #8192,
    },
}

zarr_v3_metadata = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (8, 8, 1),
    "data_type": "float32",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (8, 8, 1)}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "."}},
    "codecs": [{"name": "eccc-rpn-fst", "configuration": {"fst_format": "XDF"}}],
    "fill_value": "NaN",
    "dimension_names": ["x", "y", "z"],
    "attributes": {},
}

parsed_metadata = ArrayV3Metadata.from_dict(zarr_v3_metadata)
fst_virtual_array = ManifestArray(
    chunkmanifest=chunk_manifest, metadata=parsed_metadata
)

# Création du dataset initial
ds = xr.Dataset(data_vars={"temperature": (["x", "y", "z"], fst_virtual_array)})

print("\n--- 🚀 Déclenchement du décodage via Kerchunk ---")

# Transformation en Kerchunk
# virtualizarr peut parfois renommer ou grouper différemment
m_dict = ds.virtualize.to_kerchunk()

import fsspec
from fsspec.implementations.reference import ReferenceFileSystem

fs = fsspec.filesystem("reference", fo=m_dict, remote_protocol="file")
mapper = fs.get_mapper("")

# On ouvre avec le moteur zarr (car VirtualiZarr génère des refs Zarr)
real_ds = xr.open_dataset(mapper, engine="zarr", chunks={})

# 3. Maintenant on peut accéder à la variable
print(f"Variables détectées : {list(real_ds.data_vars)}")
actual_data = real_ds["temperature"].values

print("\n--- ✅ Output Final ---")
print(f"Shape: {actual_data.shape}")
print(actual_data[:, :, 0])
